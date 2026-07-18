from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("POC_AUDIT_DB_URL"),
    reason="POC_AUDIT_DB_URL not set — skipping Postgres integration test",
)


async def test_append_only_trigger():
    """DELETE raises due to immutability trigger."""
    import asyncpg

    dsn = os.environ["POC_AUDIT_DB_URL"]
    conn = await asyncpg.connect(dsn)
    try:
        # Insert a test row
        row_id = await conn.fetchval(
            """
            INSERT INTO audit_log (tenant, event_type, data, timestamp_ms)
            VALUES ($1, $2, $3::jsonb, $4)
            RETURNING id
            """,
            "test_tenant",
            "integration_test",
            "{}",
            1000,
        )

        # Attempt DELETE — must fail due to immutability trigger
        with pytest.raises(asyncpg.exceptions.RaiseError):
            await conn.execute("DELETE FROM audit_log WHERE id = $1", row_id)

        # Attempt UPDATE — must also fail
        with pytest.raises(asyncpg.exceptions.RaiseError):
            await conn.execute("UPDATE audit_log SET event_type = 'tampered' WHERE id = $1", row_id)
    finally:
        # Clean up test row via direct system command (bypasses trigger from outside test)
        # Note: in a real environment the row stays forever — this is integration scaffolding only
        await conn.close()
