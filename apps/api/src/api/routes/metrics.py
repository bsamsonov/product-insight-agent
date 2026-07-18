from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter(prefix="/metrics", tags=["metrics"])

_AUDIT_PATH = Path("audit.jsonl")


def _since_ms(days: int) -> int:
    """Return epoch-milliseconds for `days` ago."""
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    return now_ms - days * 86_400_000


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a pre-sorted list."""
    if not sorted_values:
        return 0.0
    idx = (p / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@router.get("/summary")
async def metrics_summary(days: int = Query(default=1, ge=1, le=90)) -> dict:
    """Aggregate metrics from the audit log for the last N days.

    Returns:
        - total_requests: all ``ask`` events in the window
        - requests_by_event_type: count per event_type
        - avg_latency_ms: arithmetic mean latency (ask events only)
        - p95_latency_ms: 95th-percentile latency (ask events only)
        - total_cost_usd: sum of cost_usd across ask events
        - avg_cost_per_request: mean cost per ask event
        - refusal_rate: fraction of events that are ``input_rejected``
        - requests_by_tenant: count of ask events per tenant
    """
    since = _since_ms(days)

    event_type_counts: dict[str, int] = {}
    latencies: list[float] = []
    costs: list[float] = []
    tenant_counts: dict[str, int] = {}
    ask_count = 0
    rejected_count = 0

    if _AUDIT_PATH.exists():
        with _AUDIT_PATH.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                ts = rec.get("timestamp_ms", 0)
                if ts < since:
                    continue

                etype = rec.get("event_type", "unknown")
                event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

                if etype == "ask":
                    ask_count += 1
                    data = rec.get("data", {})
                    lat = data.get("latency_ms")
                    if isinstance(lat, (int, float)):
                        latencies.append(float(lat))
                    cost = data.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        costs.append(float(cost))
                    tenant = rec.get("tenant", "unknown")
                    tenant_counts[tenant] = tenant_counts.get(tenant, 0) + 1

                elif etype == "input_rejected":
                    rejected_count += 1

    total_events = sum(event_type_counts.values())
    latencies_sorted = sorted(latencies)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies_sorted, 95)
    total_cost = sum(costs)
    avg_cost = total_cost / ask_count if ask_count else 0.0
    refusal_rate = rejected_count / total_events if total_events else 0.0

    return {
        "total_requests": ask_count,
        "requests_by_event_type": event_type_counts,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_request": round(avg_cost, 6),
        "refusal_rate": round(refusal_rate, 4),
        "requests_by_tenant": tenant_counts,
    }
