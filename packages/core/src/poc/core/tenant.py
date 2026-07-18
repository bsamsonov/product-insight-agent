from __future__ import annotations

from contextvars import ContextVar

_tenant_var: ContextVar[str] = ContextVar("tenant_id", default="default")


def get_tenant() -> str:
    """Get the current tenant ID from context."""
    return _tenant_var.get()


def set_tenant(tenant: str) -> None:
    """Set the current tenant ID (use only at request boundary)."""
    _tenant_var.set(tenant)


def current_tenant_token(tenant: str):
    """Context manager / token for scoped tenant setting.

    Usage::

        token = current_tenant_token("acme")
        try:
            ...
        finally:
            _tenant_var.reset(token)
    """
    return _tenant_var.set(tenant)


def reset_tenant(token) -> None:
    """Reset tenant to previous value using the token returned by set."""
    _tenant_var.reset(token)
