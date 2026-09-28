import asyncio

from poc.llm import rate_limit


def test_limiter_disabled_by_default(monkeypatch):
    monkeypatch.delenv("POC_LLM_RPM", raising=False)
    monkeypatch.delenv("POC_GROQ_RPM", raising=False)
    rate_limit.reset()
    assert rate_limit.limiter_for("groq") is None


def test_provider_specific_rpm_wins(monkeypatch):
    monkeypatch.setenv("POC_LLM_RPM", "10")
    monkeypatch.setenv("POC_GEMINI_RPM", "600")
    rate_limit.reset()
    limiter = rate_limit.limiter_for("gemini")
    assert limiter is not None
    assert limiter._interval == 0.1
    rate_limit.reset()


async def test_limiter_spaces_calls():
    # tests/unit/llm/conftest.py replaces asyncio.sleep with an AsyncMock, so assert the
    # requested waits instead of wall-clock time.
    limiter = rate_limit.RateLimiter(rpm=1200)  # 50 ms apart
    for _ in range(4):
        await limiter.acquire()
    waits = [call.args[0] for call in asyncio.sleep.await_args_list]
    assert len(waits) == 3  # the first call goes straight through
    assert waits[-1] > waits[0] > 0.04


async def test_pacing_meter_accumulates_waits_from_child_tasks():
    limiter = rate_limit.RateLimiter(rpm=1200)  # 50 ms apart
    meter = rate_limit.start_pacing_meter()
    await asyncio.gather(*(asyncio.create_task(limiter.acquire()) for _ in range(3)))
    assert meter[0] > 0.1  # waits of ~50 ms + ~100 ms recorded via child tasks
