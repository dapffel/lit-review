from lit_review.tracing import trace_async, tracing_enabled


def test_tracing_disabled_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert tracing_enabled() is False


def test_tracing_respects_explicit_disable(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing_enabled() is False


def test_tracing_enabled_with_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing_enabled() is True


async def test_trace_async_is_transparent_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    calls: list[tuple] = []

    @trace_async("demo")
    async def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    result = await add(2, 3)
    assert result == 5
    assert calls == [(2, 3)]
