import logging
from typing import Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from src.agent.policy import requires_human_approval
from src.agent.state import AgentState
from src.core.audit import log_action
from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)


class CandidateScore(BaseModel):
    """Оценка соответствия вакансии профилю пользователя."""

    score: float  # 0..1
    reason: str


class ResumeChoice(BaseModel):
    """Выбранное резюме для отклика (имя из списка RESUMES)."""

    resume: str


# Порог: вакансии с оценкой выше идут в отклик
APPLY_THRESHOLD = 0.6


async def _score_candidate(
    gateway: LLMGateway, candidate: dict, profile: dict, chunks: list[str] | None = None
) -> CandidateScore:
    """Оценить соответствие вакансии профилю через LLM (структурированный вывод).

    :param gateway: единая точка доступа к LLM
    :param candidate: словарь с данными вакансии
    :param profile: профиль соискателя (навыки/опыт) для сопоставления
    :param chunks: релевантные фрагменты резюме (RAG), подмешиваемые в промпт
    :return: оценка соответствия (score + reason)
    """
    chunks = chunks or []
    context = f"Профиль: {profile}"
    if chunks:
        fragments = "\n".join(f"- {chunk}" for chunk in chunks)
        context += f"\nРелевантные фрагменты резюме:\n{fragments}"
    return await gateway.invoke_structured(
        gateway.text_model,
        [
            (
                "system",
                "Оцени релевантность вакансии профилю соискателя от 0 до 1. "
                "score — число, reason — краткое обоснование.",
            ),
            ("human", f"{context}\nВакансия: {candidate}"),
        ],
        CandidateScore,
    )


async def _retrieve_chunks(
    user_id: int,
    candidate: dict,
    retriever: Callable[[int, str], Awaitable[list[str]]] | None,
) -> list[str]:
    """Извлечь релевантные фрагменты резюме; при сбое RAG — пустой список."""
    if retriever is None:
        return []
    query = f"{candidate.get('title', '')} {candidate.get('description', '')}".strip()
    if not query:
        return []
    try:
        return await retriever(user_id, query)
    except Exception:
        # RAG недоступен (нет эмбеддингов) — скорим без фрагментов
        return []


async def _pick_resume(
    gateway: LLMGateway,
    resumes: dict[str, str],
    job: dict,
    full_text: str,
) -> str:
    """Выбрать резюме из списка, наиболее подходящее вакансии.

    :param gateway: единая точка доступа к LLM
    :param resumes: словарь имя резюме -> описание
    :param job: данные вакансии (короткое описание)
    :param full_text: полный текст вакансии
    :return: имя выбранного резюме (или "" если список пуст)
    """
    if not resumes:
        return ""
    if len(resumes) == 1:
        return next(iter(resumes))
    options = "\n".join(f"- {name}: {desc}" for name, desc in resumes.items())
    try:
        choice = await gateway.invoke_structured(
            gateway.text_model,
            [
                (
                    "system",
                    "Выбери из списка резюме то, которое лучше всего подходит "
                    "для вакансии. Верни поле resume с точным именем из списка.",
                ),
                (
                    "human",
                    f"Доступные резюме:\n{options}\n\nВакансия: {job}\n"
                    f"Описание вакансии:\n{full_text}",
                ),
            ],
            ResumeChoice,
        )
        if choice.resume in resumes:
            return choice.resume
        return next(iter(resumes))
    except Exception as exc:  # noqa: BLE001 - при сбое берём первое резюме
        logger.warning("Выбор резюме не удался: %s", exc)
        return next(iter(resumes))


def build_graph(
    gateway: LLMGateway,
    searcher: Callable[[int, str], Awaitable[list[dict]]],
    checkpointer: BaseCheckpointSaver | None = None,
    applier: Callable[[int, dict], Awaitable[None]] | None = None,
    profile_provider: Callable[[int], Awaitable[dict]] | None = None,
    retriever: Callable[[int, str], Awaitable[list[str]]] | None = None,
    vacancy_reader: Callable[[int, str], Awaitable[str]] | None = None,
    resumes: dict[str, str] | None = None,
):
    """Собрать граф поиска: search → match → decision → apply → report + HITL.

    :param gateway: LLMGateway (скоринг, письма)
    :param searcher: async callable(user_id, query) -> list[dict]
    :param checkpointer: checkpointer LangGraph (для HITL)
    :param applier: async callable(user_id, decision) — реальный отклик
    :param profile_provider: async callable(user_id) -> dict — профиль соискателя
    :param retriever: async callable(user_id, query) -> list[str] — фрагменты резюме
    :param vacancy_reader: async callable(user_id, url) -> str — полный текст вакансии
    :param resumes: словарь имя резюме -> описание (для выбора под вакансию)
    :return: скомпилированный граф
    """

    async def search_node(state: AgentState) -> dict:
        """Найти вакансии через внедрённый searcher по запросу пользователя."""
        query = state.get("user_message", "")
        logger.info("Поиск вакансий: query=%r", query)
        candidates = await searcher(state["user_id"], query)
        logger.info("Найдено вакансий: %s", len(candidates))
        return {"candidates": candidates}

    async def match_node(state: AgentState) -> dict:
        """Оценить каждую вакансию через LLM-скоринг с учётом профиля и RAG."""
        profile = await profile_provider(state["user_id"]) if profile_provider else {}
        decisions = []
        for c in state.get("candidates", []):
            chunks = await _retrieve_chunks(state["user_id"], c, retriever)
            scored = await _score_candidate(gateway, c, profile, chunks)
            logger.info(
                "Скоринг вакансии «%s»: %.2f (%s)",
                c.get("title", "?"),
                scored.score,
                scored.reason,
            )
            decisions.append({"job": c, "score": scored.score, "reason": scored.reason})
        return {"decisions": decisions}

    async def decision_node(state: AgentState) -> dict:
        """Вынести решение apply/skip по порогу релевантности и набросать письмо."""
        final = []
        for d in state.get("decisions", []):
            decision = "apply" if d["score"] >= APPLY_THRESHOLD else "skip"
            logger.info(
                "Решение по «%s»: %s (score %.2f)",
                d.get("job", {}).get("title", "?"),
                decision,
                d["score"],
            )
            d["decision"] = decision
            if decision == "apply":
                # Полный текст вакансии: учитываем требования работодателя к письму
                url = d.get("job", {}).get("url", "")
                full_text = ""
                if vacancy_reader is not None and url:
                    try:
                        full_text = await vacancy_reader(state["user_id"], url)
                    except Exception as exc:  # noqa: BLE001 - письмо пишем и без текста
                        logger.warning("Чтение вакансии %s не удалось: %s", url, exc)
                d["full_text"] = full_text
                # Выбор резюме под вакансию (если задано несколько)
                d["resume"] = await _pick_resume(
                    gateway, resumes or {}, d.get("job", {}), full_text
                )
                # Черновик сопроводительного письма под конкретную вакансию
                msg = await gateway.text_model.ainvoke(
                    [
                        (
                            "system",
                            "Ты пишешь краткое сопроводительное письмо на русском "
                            "(3-5 предложений) под конкретную вакансию. Если в "
                            "описании вакансии есть конкретные требования к письму "
                            "(например, указать что-то определённое) — выполни их.",
                        ),
                        (
                            "human",
                            f"Вакансия: {d['job']}\n"
                            f"Полное описание вакансии:\n{full_text}",
                        ),
                    ]
                )
                d["cover_letter"] = msg.content
            else:
                d["cover_letter"] = ""
            final.append(d)
        return {"decisions": final}

    async def apply_node(state: AgentState) -> dict:
        """Откликнуться на подходящие вакансии с HITL-подтверждением."""
        applied = []
        for d in state.get("decisions", []):
            if d.get("decision") != "apply":
                continue
            # Высокорисковое действие: пауза до подтверждения человека
            if requires_human_approval("apply"):
                approval = interrupt(
                    {
                        "action": "apply",
                        "job": d.get("job"),
                        "cover_letter": d.get("cover_letter", ""),
                        "resume": d.get("resume", ""),
                    }
                )
                # Аудит: что агент хотел сделать и какое решение принято
                log_action(
                    state["user_id"],
                    "apply",
                    "approved" if approval is True else "rejected",
                    {"job": d.get("job", {})},
                )
                if approval is not True:
                    continue
            # Реальное выполнение отклика через внедрённый side-effect
            if applier is not None:
                await applier(state["user_id"], d)
            applied.append(d)
        return {"decisions": applied, "needs_human": False, "pending_action": None}

    async def report_node(state: AgentState) -> dict:
        """Сформировать итоговую статистику и текст ответа пользователю."""
        applied_count = len(state.get("decisions", []))
        report = {"applied_count": applied_count, "replied_count": 0}
        reply = f"Откликнулся на {applied_count} вакансий"
        return {"report": report, "reply": reply}

    graph = StateGraph(AgentState)
    graph.add_node("search", search_node)
    graph.add_node("match", match_node)
    graph.add_node("decision", decision_node)
    graph.add_node("apply", apply_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "match")
    graph.add_edge("match", "decision")
    graph.add_edge("decision", "apply")
    graph.add_edge("apply", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
