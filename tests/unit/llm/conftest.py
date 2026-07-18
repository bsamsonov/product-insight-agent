from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _instant_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch asyncio.sleep so tenacity backoff doesn't slow tests down."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
