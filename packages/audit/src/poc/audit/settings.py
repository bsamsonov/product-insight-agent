from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class AuditSettings(BaseSettings):
    db_url: str = Field(
        default="postgresql://poc:poc@localhost:5432/poc",
        alias="POC_AUDIT_DB_URL",
    )

    model_config = {"env_prefix": "POC_AUDIT_", "populate_by_name": True}
