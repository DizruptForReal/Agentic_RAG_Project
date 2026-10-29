"""LLM interface plus a helper that validates structured output and repairs it once."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM provider failed (network, quota, server error)."""


class LLMClient(Protocol):
    def complete_json(
        self, *, purpose: str, system: str, user: str, schema: dict[str, Any], model: str | None = None
    ) -> Any:
        """Return the model's structured output (usually a dict). May return malformed data."""


@dataclass
class StructuredResult:
    value: BaseModel | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    llm_failed: bool = False


def _as_obj(raw: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):] if "{" in text else text
        return json.loads(text)
    return raw


def call_structured(
    llm: LLMClient,
    *,
    purpose: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    model_cls: type[T],
    model: str | None = None,
) -> StructuredResult:
    """One call, validate; on failure one repair call with the error; then give up (caller falls back)."""
    errors: list[str] = []
    prompt = user
    for attempt in (1, 2):
        try:
            raw = llm.complete_json(purpose=purpose, system=system, user=prompt, schema=schema, model=model)
        except LLMError as exc:
            errors.append(f"llm_error: {exc}")
            return StructuredResult(None, attempt, errors, llm_failed=True)
        try:
            return StructuredResult(model_cls.model_validate(_as_obj(raw)), attempt, errors)
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
            msg = str(exc)[:1500]
            errors.append(f"invalid_output: {msg}")
            log.warning("malformed %s output (attempt %d): %s", purpose, attempt, msg[:200])
            if attempt == 1:
                from app.agent import progress

                progress.report("The model returned malformed output; asking it to correct it", "warn")
            prompt = (
                f"{user}\n\n---\nYour previous output was INVALID and was rejected.\n"
                f"Validation error:\n{msg}\n\nPrevious output:\n{str(raw)[:2000]}\n\n"
                "Return a corrected output that matches the schema exactly."
            )
    return StructuredResult(None, 2, errors)
