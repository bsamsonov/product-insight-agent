from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from poc.audit.logger import AuditEvent
from poc.audit.store import PostgresAuditStore

pytestmark = pytest.mark.asyncio


def make_pool_with_conn(conn: AsyncMock) -> AsyncMock:
    """Return an asyncpg-like pool mock whose .acquire() is an async context manager."""
    pool = AsyncMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


@pytest.fixture
def mock_pool():
    conn = AsyncMock()
    pool = make_pool_with_conn(conn)
    return pool, conn


async def test_log_event_inserts_row(mock_pool):
    pool, conn = mock_pool
    store = PostgresAuditStore(dsn="postgresql://fake/fake")
    store._pool = pool  # inject mock

    event = AuditEvent(event_type="ask", tenant="default", data={"question": "test"})
    await store.log_event(event)

    conn.execute.assert_called_once()
    call_args = conn.execute.call_args[0]
    assert "INSERT INTO audit_log" in call_args[0]


async def test_log_event_never_raises(mock_pool):
    pool, conn = mock_pool
    conn.execute.side_effect = Exception("DB error")
    store = PostgresAuditStore(dsn="postgresql://fake/fake")
    store._pool = pool

    # Should NOT raise — log_event is fire-and-forget safe
    event = AuditEvent(event_type="ask", tenant="default", data={})
    await store.log_event(event)  # must not raise


async def test_get_recent_returns_records(mock_pool):
    pool, conn = mock_pool
    conn.fetch.return_value = [
        {
            "id": 1,
            "tenant": "default",
            "event_type": "ask",
            "data": '{"question": "test"}',
            "timestamp_ms": 1000,
            "created_at": "2026-01-01",
        }
    ]
    store = PostgresAuditStore(dsn="postgresql://fake/fake")
    store._pool = pool

    records = await store.get_recent(tenant="default", limit=10)
    assert len(records) == 1
    conn.fetch.assert_called_once()


async def test_log_event_no_pool_does_not_raise():
    """log_event with no pool should log error but never raise."""
    store = PostgresAuditStore(dsn="postgresql://fake/fake")
    # _pool is None by default

    event = AuditEvent(event_type="ask", tenant="t1", data={})
    await store.log_event(event)  # must not raise


async def test_get_recent_no_pool_raises():
    """get_recent with no pool should raise RuntimeError."""
    store = PostgresAuditStore(dsn="postgresql://fake/fake")

    with pytest.raises(RuntimeError, match="pool not initialised"):
        await store.get_recent(tenant="t1")


async def test_connect_close_lifecycle():
    """connect() and close() interact with asyncpg.create_pool."""
    with patch("poc.audit.store.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        mock_pool_instance = AsyncMock()
        mock_create.return_value = mock_pool_instance

        store = PostgresAuditStore(dsn="postgresql://fake/fake")
        await store.connect()
        mock_create.assert_called_once_with("postgresql://fake/fake")
        assert store._pool is mock_pool_instance

        await store.close()
        mock_pool_instance.close.assert_called_once()
        assert store._pool is None


async def test_hundred_events_produce_hundred_inserts(mock_pool):
    """AC (S5.T3): 100 logged requests → 100 INSERT rows.

    Unit-level proof that every log_event issues exactly one INSERT (the
    DB-backed row count + append-only DELETE check live in the integration
    test tests/integration/audit/test_postgres_store_integration.py).
    """
    pool, conn = mock_pool
    store = PostgresAuditStore(dsn="postgresql://fake/fake")
    store._pool = pool

    for i in range(100):
        await store.log_event(AuditEvent(event_type="ask", tenant="default", data={"i": i}))

    assert conn.execute.call_count == 100
    assert all("INSERT INTO audit_log" in call[0][0] for call in conn.execute.call_args_list)


async def test_log_event_passes_correct_fields(mock_pool):
    """Verify tenant, event_type, and timestamp_ms are forwarded to SQL."""
    pool, conn = mock_pool
    store = PostgresAuditStore(dsn="postgresql://fake/fake")
    store._pool = pool

    event = AuditEvent(event_type="search", tenant="acme", data={"q": "revenue"})
    await store.log_event(event)

    args = conn.execute.call_args[0]
    # args: (sql, tenant, event_type, data_json, timestamp_ms)
    assert args[1] == "acme"
    assert args[2] == "search"
    assert args[4] == event.timestamp_ms
