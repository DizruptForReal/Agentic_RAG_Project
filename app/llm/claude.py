"""Anthropic Claude implementation of LLMClient using forced tool-use for structured output."""
from __future__ import annotations

from typing import Any

from app.llm.base import LLMError


class ClaudeClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 2048, timeout_s: float = 60.0):
        import anthropic  # imported lazily so tests never need it

        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens

    def complete_json(
        self, *, purpose: str, system: str, user: str, schema: dict[str, Any], model: str | None = None
    ) -> Any:
        try:
            resp = self.client.messages.create(
                model=model or self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{
                    "name": purpose,
                    "description": f"Submit the {purpose.replace('_', ' ')} result.",
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": purpose},
            )
        except self._anthropic.APIError as exc:  # network, 5xx, rate limit after SDK retries
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input
        # No tool call: hand back the text so the validator can reject / repair it.
        return "".join(getattr(b, "text", "") for b in resp.content)
