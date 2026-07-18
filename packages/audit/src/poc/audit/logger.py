from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiofiles

_log = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    event_type: str
    tenant: str
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    data: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    user_id: str | None = None


class AuditLogger:
    """Append-only async audit log, stored as JSONL.

    Each call to `log()` appends a JSON line atomically.
    Thread-safe via asyncio lock.
    """

    def __init__(self, path: str | Path = "audit.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, event: AuditEvent) -> None:
        line = json.dumps(asdict(event), ensure_ascii=False) + "\n"
        async with self._lock, aiofiles.open(self._path, mode="a", encoding="utf-8") as f:
            await f.write(line)

    async def query(
        self,
        *,
        tenant: str | None = None,
        event_type: str | None = None,
        since_ms: int | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Read log entries matching filters (slow scan — for debugging only)."""
        results: list[AuditEvent] = []
        if not self._path.exists():
            return results

        async with aiofiles.open(self._path, encoding="utf-8") as f:
            async for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = AuditEvent(**data)
                if tenant and event.tenant != tenant:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if since_ms and event.timestamp_ms < since_ms:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break

        return results
