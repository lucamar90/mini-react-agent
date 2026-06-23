"""Language-model backends.

Every backend exposes ``complete(prompt: str) -> str``.

- ``ReplayLLM``  — returns a recorded list of outputs in order. Lets the whole
  agent run deterministically offline (for the demo and tests): the tool calls
  and observations are executed for real; only the "thinking" is pre-recorded.
- ``OpenAILLM`` / ``AnthropicLLM`` — real models, with a stop sequence so the
  model halts before fabricating its own Observation.
"""

from __future__ import annotations

import os
from typing import Protocol

from .prompt import STOP_SEQUENCE, SYSTEM_PROMPT


class LLM(Protocol):
    name: str
    model: str

    def complete(self, prompt: str) -> str: ...


class ReplayLLM:
    """Deterministic, offline backend that replays recorded model outputs."""

    name = "replay"

    def __init__(self, outputs: list[str], model: str = "replay-1"):
        self._outputs = list(outputs)
        self._i = 0
        self.model = model

    def complete(self, prompt: str) -> str:
        if self._i >= len(self._outputs):
            raise RuntimeError(
                "ReplayLLM ran out of recorded outputs — the agent took more "
                "steps than the script provides."
            )
        out = self._outputs[self._i]
        self._i += 1
        return out


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install 'mini-react-agent[openai]'"
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI backend.")
        self._client = OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stop=[STOP_SEQUENCE],
        )
        return resp.choices[0].message.content or ""


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Anthropic SDK not installed. Run: pip install 'mini-react-agent[anthropic]'"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the Anthropic backend.")
        self._client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=[STOP_SEQUENCE],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def get_llm(name: str, model: str | None = None, outputs: list[str] | None = None) -> LLM:
    """Factory: ``replay`` | ``openai`` | ``anthropic``."""
    key = name.lower()
    if key == "replay":
        return ReplayLLM(outputs or [], model=model or "replay-1")
    if key == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini")
    if key == "anthropic":
        return AnthropicLLM(model=model or "claude-haiku-4-5-20251001")
    raise ValueError(f"Unknown LLM backend {name!r}. Choose: replay, openai, anthropic.")
