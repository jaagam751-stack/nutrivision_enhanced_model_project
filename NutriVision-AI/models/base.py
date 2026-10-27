"""Declarative base and shared column helpers for all NutriVision AI models.

Kept in its own module so that ``models/*.py`` can import the base without
creating a circular import through the ``database`` package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    """Timezone-aware current UTC time.

    Used as a Python-side default so timestamps work identically on PostgreSQL
    (``timestamptz``) and on the SQLite development fallback.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for every ORM model in the project."""

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a plain dictionary (handy for charts and tests)."""
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        identity = getattr(self, "user_id", None) or getattr(self, "meal_id", None)
        return f"<{self.__class__.__name__} id={identity}>"
