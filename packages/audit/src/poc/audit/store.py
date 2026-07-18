from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

import asyncpg
from poc.audit.logger import AuditEvent

_log = logging.getLogger(__name__)


class PostgresAuditStore:
    """Append-only audit log backed by Postgres.

    The audit_log table has an immutability trigger blocking UPDATE/DELETE.
    Use connect() before any operations, close() on shutdown.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create asyncpg connection pool."""
        self._pool = await asyncpg.create_pool(self._dsn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def log(self, event: AuditEvent) -> None:
        """Alias for log_event — matches AuditLogger's interface so the two backends
        are duck-type interchangeable from the API's point of view."""
        await self.log_event(event)

    async def log_event(self, event: AuditEvent) -> None:
        """Insert one audit event. Never raises — logs errors instead.

        The insert-only pattern: we NEVER update or delete audit records.
        Any attempt to do so raises an exception from the Postgres trigger.
        """
        if self._pool is None:
            _log.error("audit_store: pool not initialised, dropping event %s", event.event_type)
            return

        sql = """
            INSERT INTO audit_log (tenant, event_type, data, timestamp_ms)
            VALUES ($1, $2, $3, $4)
        """
        data_json = json.dumps(asdict(event).get("data", {}), ensure_ascii=False)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql, event.tenant, event.event_type, data_json, event.timestamp_ms
                )
        except Exception:
            _log.exception(
                "audit_store: failed to insert event type=%s tenant=%s",
                event.event_type,
                event.tenant,
            )

    async def get_recent(self, *, tenant: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent `limit` events for a tenant, newest first."""
        if self._pool is None:
            raise RuntimeError("audit_store: pool not initialised")

        sql = """
            SELECT id, tenant, event_type, data, timestamp_ms, created_at
            FROM audit_log
            WHERE tenant = $1
            ORDER BY timestamp_ms DESC
            LIMIT $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, tenant, limit)
        return [dict(row) for row in rows]
