from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def configure(level: str | None = None, json_format: bool | None = None) -> None:
    """Configure structlog-style logging.

    Uses JSON format in prod, human-readable in dev.
    """
    env = os.getenv("ENV", "dev").lower()

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    if json_format is None:
        json_format = env == "prod"

    if json_format:
        import json
        import sys
        import time

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                data = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    data["exc"] = self.formatException(record.exc_info)
                return json.dumps(data, ensure_ascii=False)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
        )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
