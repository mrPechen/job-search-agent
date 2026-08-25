import base64
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class BrowserAction(BaseModel):
    """Одно действие браузера, выбранное VLM."""

    tool: Literal["navigate", "click", "type", "scroll", "back", "done"]
    args: dict = Field(default_factory=dict)


class SearchOutcome(BaseModel):
    """Результат поиска: найденные вакансии."""

    candidates: list[dict] = Field(default_factory=list)
    error: str = ""


class ApplyOutcome(BaseModel):
    """Результат отклика."""

    applied: bool = False
    detail: str = ""
    error: str = ""


_SYSTEM_PROMPT = (
    "Ты управляешь браузером для поиска работы. На каждом шаге получаешь "
    "скриншот страницы и список интерактивных элементов с CSS-селекторами. "
    "Верни одно действие: tool из [navigate, click, type, scroll, back, done]. "
    "Для click/type используй selector ТОЛЬКО из списка элементов. "
    "Когда цель достигнута, верни tool=done и результат в args."
)


class BrowserLoop:
    """Цикл «скриншот → VLM → действие», пока цель не достигнута."""

    def __init__(self, executor, gateway, max_steps: int | None = None) -> None:
        self._executor = executor
        self._gateway = gateway
        self._max_steps = (
            max_steps if max_steps is not None else settings.BROWSER_MAX_STEPS
        )

    async def run(
        self,
        user_id: int,
        goal: str,
        result_schema: type,
        allowed_domains: list[str],
        start_url: str | None = None,
    ):
        if start_url:
            await self._executor.navigate(
                user_id, start_url, allowed_domains=allowed_domains
            )
        for _ in range(self._max_steps):
            page = await self._executor.extract(user_id)
            shot = await self._executor.screenshot(user_id)
            action = await self._decide(goal, page, shot)
            if action is None:
                return result_schema(error="vlm decision failed")
            if action.tool == "done":
                try:
                    return result_schema.model_validate(action.args)
                except Exception as exc:
                    logger.warning("Invalid done args: %s", exc)
                    return result_schema(error="invalid result")
            await self._execute(user_id, action, allowed_domains)
        return result_schema(error="step limit exceeded")

    async def _decide(self, goal: str, page: dict, shot: bytes) -> BrowserAction | None:
        image = base64.b64encode(shot).decode()
        elements = page.get("elements", [])
        text = (page.get("text") or "")[:4000]
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Цель: {goal}\n\nТекст страницы:\n{text}\n\n"
                            f"Элементы (селекторы):\n{elements}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ]
            ),
        ]
        try:
            return await self._gateway.invoke_structured(
                self._gateway.vision_model, messages, BrowserAction
            )
        except Exception as exc:
            logger.warning("VLM decision failed, retrying: %s", exc)
            try:
                return await self._gateway.invoke_structured(
                    self._gateway.vision_model, messages, BrowserAction
                )
            except Exception as exc2:
                logger.warning("VLM decision failed again, giving up: %s", exc2)
                return None

    async def _execute(
        self, user_id: int, action: BrowserAction, allowed_domains: list[str]
    ) -> None:
        tool = action.tool
        args = action.args or {}
        try:
            if tool == "navigate":
                await self._executor.navigate(
                    user_id, args["url"], allowed_domains=allowed_domains
                )
            elif tool == "click":
                await self._executor.click(user_id, args["selector"])
            elif tool == "type":
                await self._executor.type_text(user_id, args["selector"], args["text"])
            elif tool == "scroll":
                await self._executor.scroll(user_id, args.get("delta", 800))
            elif tool == "back":
                await self._executor.back(user_id)
        except Exception as exc:
            logger.warning("Browser action %s failed: %s", tool, exc)
