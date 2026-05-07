"""Tests for harness.inference.client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harness.inference.client import (
    FakeInferenceClient,
    InferenceClient,
    OpenAICompatibleClient,
)


def test_fake_client_returns_canned_responses_in_order() -> None:
    client = FakeInferenceClient(responses=["one", "two", "three"])
    assert client.complete("p1") == "one"
    assert client.complete("p2") == "two"
    assert client.complete("p3") == "three"


def test_fake_client_raises_when_exhausted() -> None:
    client = FakeInferenceClient(responses=["only one"])
    client.complete("p1")
    with pytest.raises(RuntimeError, match="exhausted"):
        client.complete("p2")


def test_fake_client_records_calls() -> None:
    client = FakeInferenceClient(responses=["a", "b"])
    client.complete("first prompt", temperature=0.7)
    client.complete("second prompt", temperature=0.3)
    assert len(client.calls) == 2
    assert client.calls[0]["prompt"] == "first prompt"
    assert client.calls[0]["sampling_params"] == {"temperature": 0.7}
    assert client.calls[1]["prompt"] == "second prompt"


@patch("harness.inference.client.OpenAI")
def test_openai_compatible_client_calls_completions_api(mock_openai_cls) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hello world"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
        model="qwen3:4b",
    )
    result = client.complete("test prompt", temperature=0.5)

    assert result == "hello world"
    mock_openai_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="not-needed"
    )
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "qwen3:4b"
    assert kwargs["temperature"] == 0.5
    assert kwargs["messages"] == [{"role": "user", "content": "test prompt"}]


@patch("harness.inference.client.OpenAI")
def test_openai_compatible_client_omits_temperature_when_not_supplied(
    mock_openai_cls,
) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="x"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = OpenAICompatibleClient(
        base_url="http://x", api_key="x", model="m"
    )
    client.complete("p")

    _, kwargs = mock_client.chat.completions.create.call_args
    assert "temperature" not in kwargs
