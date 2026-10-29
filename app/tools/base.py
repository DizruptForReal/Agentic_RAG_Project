"""Tool protocol, execution context, and a registry that validates args, retries and wraps errors."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol

from pydantic import BaseModel, ValidationError

from app.agent.models import Action, Observation, ObsStatus

log = logging.getLogger(__name__)


class ToolError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ToolContext:
    investigation_id: str
    attachments: list[str] = field(default_factory=list)


class Tool(Protocol):
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    def run(self, args: Any, ctx: ToolContext) -> Observation: ...


class ToolRegistry:
    def __init__(self, tools: list[Tool], *, retries: int = 2, backoff_s: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep):
        self.tools = {t.name: t for t in tools}
        self.retries, self.backoff_s, self.sleep = retries, backoff_s, sleep

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "args": t.Args.model_json_schema()}
                for t in self.tools.values()]

    def run(self, action: Action, ctx: ToolContext) -> Observation:
        name = action.type.value
        tool = self.tools.get(name)
        if tool is None:
            return Observation(tool=name, status=ObsStatus.ERROR, error=f"unknown tool '{name}'")
        try:
            args = tool.Args.model_validate(action.args)
        except ValidationError as exc:
            # The model produced bad arguments: a malformed-output problem, not a backend failure.
            return Observation(tool=name, status=ObsStatus.ERROR, retryable=False,
                               error=f"invalid arguments: {exc.errors(include_url=False)}")
        last_error = ""
        for attempt in range(1, self.retries + 2):
            try:
                obs = tool.run(args, ctx)
                obs.attempts = attempt
                return obs
            except ToolError as exc:
                last_error = str(exc)
                if exc.retryable and attempt <= self.retries:
                    log.warning("%s failed (attempt %d), retrying: %s", name, attempt, exc)
                    self.sleep(self.backoff_s * 2 ** (attempt - 1))
                    continue
                return Observation(tool=name, status=ObsStatus.ERROR, error=last_error,
                                   retryable=exc.retryable, attempts=attempt)
            except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the investigation
                log.exception("tool %s crashed", name)
                return Observation(tool=name, status=ObsStatus.ERROR, attempts=attempt,
                                   error=f"{type(exc).__name__}: {exc}")
        return Observation(tool=name, status=ObsStatus.ERROR, error=last_error, attempts=self.retries + 1)
