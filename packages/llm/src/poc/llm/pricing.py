from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)
_PRICING: dict[str, Any] | None = None
_PRICING_PATH = Path(__file__).parents[3] / "data" / "pricing.yaml"


def _load_pricing() -> dict[str, Any]:
    global _PRICING
    if _PRICING is None:
        with _PRICING_PATH.open() as f:
            _PRICING = yaml.safe_load(f)
    return _PRICING


def estimate_cost(provider: str, model: str, in_tok: int, out_tok: int) -> float | None:
    """Return estimated USD cost or None when pricing is unknown."""
    pricing = _load_pricing()
    provider_data = pricing.get("providers", {}).get(provider)

    if provider_data is None:
        _log.warning("Unknown provider in pricing table: %s", provider)
        return None

    if provider_data.get("free"):
        return 0.0

    model_data = provider_data.get("models", {}).get(model)
    if model_data is None:
        _log.warning("Unknown model in pricing table: provider=%s model=%s", provider, model)
        return None

    in_price: float = model_data.get("input_per_1m_usd", 0.0)
    out_price: float = model_data.get("output_per_1m_usd", 0.0)
    return (in_tok * in_price + out_tok * out_price) / 1_000_000
