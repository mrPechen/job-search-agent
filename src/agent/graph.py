from typing import Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from src.agent.policy import requires_human_approval
from src.agent.router import IntentRouter
from src.agent.state import AgentState
from src.llm.gateway import LLMGateway


class CandidateScore(BaseModel):
    """Оценка соответствия вакансии профилю пользователя."""

    score: float  # 0..1
    reason: str


# Порог: вакансии с оценкой выше идут в отклик
APPLY_THRESHOLD = 0.6


async def _score_candidate(gateway: LLMGateway, candidate: dict) -> CandidateScore:
    """Оценить соответствие вакансии профилю через LLM (структурированный вывод).

    :param gateway: единая точка доступа к LLM
    :param candidate: словарь с данными вакансии
    :return: оценка соответствия (score + reason)
    """
    return await gateway.invoke_structured(
        gateway.text_model,
        [
            (
                "system",
                "Оцени релевантность вакансии профилю соискателя от 0 до 1. "
                "score — число, reason — краткое обоснование.",
            ),
            ("human", str(candidate)),
        ],
        CandidateScore,
    )


def build_graph(
    gateway: LLMGateway,
    searcher: Callable[[int], Awaitable[list[dict]]],
    router: IntentRouter | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Собрать граф агента с внедрёнными зависимостями.

    :param gateway: LLMGateway (для классификации, скоринга, чата)
    :param searcher: async callable(user_id) -> list[dict] — поиск вакансий
    :param router: IntentRouter (по умолчанию создаётся из gateway)
    :param checkpointer: checkpointer LangGraph (для HITL; обязателен в тестах)
    :return: скомпилированный граф
    """
    router = router or IntentRouter(gateway)

    async def router_node(state: AgentState) -> dict:
        """Классифицировать намерение пользователя."""
        intent = await router.classify(state.get("user_message", ""))
        return {"intent": intent.intent}

    async def search_node(state: AgentState) -> dict:
        """Найти вакансии через внедрённый searcher."""
        candidates = await searcher(state["user_id"])
        return {"candidates": candidates}

    async def match_node(state: AgentState) -> dict:
        """Оценить каждую вакансию через LLM-скоринг."""
        decisions = []
        for c in state.get("candidates", []):
            scored = await _score_candidate(gateway, c)
            decisions.append({"job": c, "score": scored.score, "reason": scored.reason})
        return {"decisions": decisions}

    async def decision_node(state: AgentState) -> dict:
        """Вынести решение apply/skip по порогу релевантности."""
        final = []
        for d in state.get("decisions", []):
            decision = "apply" if d["score"] >= APPLY_THRESHOLD else "skip"
            d["decision"] = decision
            # Черновик сопроводительного генерируется на этапе apply
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
                    }
                )
                if approval is not True:
                    continue
            applied.append(d)
        return {"decisions": applied, "needs_human": False, "pending_action": None}

    async def report_node(state: AgentState) -> dict:
        """Сформировать итоговую статистику и текст ответа пользователю."""
        applied_count = len(state.get("decisions", []))
        report = {"applied_count": applied_count, "replied_count": 0}
        reply = f"Откликнулся на {applied_count} вакансий"
        return {"report": report, "reply": reply}

    async def chat_node(state: AgentState) -> dict:
        """Ответить пользователю в свободном режиме чата."""
        reply_msg = await gateway.text_model.ainvoke(
            [
                (
                    "system",
                    "Ты — дружелюбный ассистент по поиску работы. "
                    "Отвечай кратко на русском.",
                ),
                ("human", state.get("user_message", "")),
            ]
        )
        return {"reply": reply_msg.content}

    def route_by_intent(state: AgentState) -> str:
        """Выбрать следующую ветку графа по намерению."""
        intent = state.get("intent", "chat")
        if intent == "search_job":
            return "search"
        if intent == "stats":
            return "report"
        if intent == "confirm":
            return "report"  # подтверждения обрабатываются через resume HITL
        return "chat"

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("search", search_node)
    graph.add_node("match", match_node)
    graph.add_node("decision", decision_node)
    graph.add_node("apply", apply_node)
    graph.add_node("report", report_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"search": "search", "report": "report", "chat": "chat"},
    )
    graph.add_edge("search", "match")
    graph.add_edge("match", "decision")
    graph.add_edge("decision", "apply")
    graph.add_edge("apply", "report")
    graph.add_edge("report", END)
    graph.add_edge("chat", END)

    return graph.compile(checkpointer=checkpointer)
