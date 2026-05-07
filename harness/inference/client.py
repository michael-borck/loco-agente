"""Inference layer — talks to OpenAI-compatible HTTP endpoints.

The harness is inference-engine-agnostic. This module wraps the OpenAI SDK
to talk to Ollama, llama.cpp server, vLLM, SGLang, or any other OpenAI-
compatible backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI


class InferenceClient(Protocol):
    """A minimal completion interface — one prompt in, one string out."""

    def complete(self, prompt: str, **sampling_params: Any) -> str: ...


@dataclass
class OpenAICompatibleClient:
    """Talks to any OpenAI-compatible chat-completions endpoint."""

    base_url: str
    api_key: str
    model: str

    def complete(self, prompt: str, **sampling_params: Any) -> str:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        for key, value in sampling_params.items():
            kwargs[key] = value
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


@dataclass
class FakeInferenceClient:
    """Test double that returns canned responses in order.

    Records every call for assertion against. Raises RuntimeError when the
    canned-response list is exhausted (so tests fail loudly on miswiring).
    """

    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursor: int = 0

    def complete(self, prompt: str, **sampling_params: Any) -> str:
        self.calls.append({"prompt": prompt, "sampling_params": sampling_params})
        if self._cursor >= len(self.responses):
            raise RuntimeError(
                f"FakeInferenceClient responses exhausted "
                f"(received {self._cursor + 1} calls, "
                f"only {len(self.responses)} responses configured)"
            )
        response = self.responses[self._cursor]
        self._cursor += 1
        return response
