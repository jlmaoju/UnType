from __future__ import annotations

import concurrent.futures
import threading

import pytest

from untype.llm import LLMClient


def test_insert_builds_default_payload(monkeypatch) -> None:
    client = LLMClient("https://example.com/v1", "secret", "demo-model")
    seen: dict = {}

    def fake_request(payload: dict) -> str:
        seen.update(payload)
        return "ok"

    monkeypatch.setattr(client, "_do_request", fake_request)

    result = client.insert("raw speech")

    assert result == "ok"
    assert seen["model"] == "demo-model"
    assert seen["temperature"] == 0.3
    assert seen["max_tokens"] == 2048
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][1]["content"] == "<transcription>\nraw speech\n</transcription>"


def test_polish_allows_per_call_overrides(monkeypatch) -> None:
    client = LLMClient("https://example.com/v1", "secret", "demo-model")
    seen: dict = {}

    def fake_request(payload: dict) -> str:
        seen.update(payload)
        return "ok"

    monkeypatch.setattr(client, "_do_request", fake_request)

    client.polish(
        "before",
        "make it shorter",
        system_prompt="custom-system",
        model="override-model",
        temperature=0.9,
        max_tokens=99,
    )

    assert seen["model"] == "override-model"
    assert seen["temperature"] == 0.9
    assert seen["max_tokens"] == 99
    assert seen["messages"][0]["content"] == "custom-system"
    assert "<original_text>\nbefore\n</original_text>" in seen["messages"][1]["content"]
    assert (
        "<voice_instruction>\nmake it shorter\n</voice_instruction>"
        in seen["messages"][1]["content"]
    )


def test_cancelled_request_recreates_http_client(monkeypatch) -> None:
    created_clients: list[FakeClient] = []

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeFuture:
        def cancel(self) -> None:
            return None

        def done(self) -> bool:
            return False

        def result(self, timeout: float | None = None) -> str:
            raise concurrent.futures.TimeoutError

    class FakeExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.future = FakeFuture()

        def submit(self, fn):
            return self.future

        def shutdown(self, wait: bool = False) -> None:
            return None

    def fake_create_http_client(self) -> FakeClient:
        client = FakeClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(LLMClient, "_create_http_client", fake_create_http_client)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", FakeExecutor)

    client = LLMClient("https://example.com/v1", "secret", "demo-model")
    first_client = client._client
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(KeyboardInterrupt):
        client.insert("raw speech", cancel_event=cancel_event)

    assert first_client.closed is True
    assert client._client is not first_client
    assert len(created_clients) == 2
