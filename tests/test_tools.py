"""Tests for harness.tools — Phase 1 stub."""
from __future__ import annotations

import pytest

from harness.tools.client import StubToolClient


def test_stub_tool_client_lists_no_tools() -> None:
    client = StubToolClient()
    assert client.list_tools() == []


def test_stub_tool_client_invoke_raises() -> None:
    client = StubToolClient()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        client.invoke(name="search", args={"q": "x"})


def test_stub_tool_client_with_advertised_tools() -> None:
    """A stub may advertise tool names without implementing them — useful
    for testing that orchestration patterns surface tool specs to prompts."""
    client = StubToolClient(advertised=["search_corpus", "fetch_paper"])
    assert client.list_tools() == ["search_corpus", "fetch_paper"]
    with pytest.raises(NotImplementedError):
        client.invoke(name="search_corpus", args={"q": "x"})
