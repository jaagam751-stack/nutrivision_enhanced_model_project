"""Database connection handling (Phase 2).

Design notes
------------
* **PostgreSQL (Neon) is the project's database.** The connection string is read
  from Streamlit Secrets (cloud) or ``DATABASE_URL`` in ``.env`` (local) via
  ``config.get_secret``.
* **SQLite is an explicit, opt-in development fallback.** It exists so the app
  can be tested without a Neon account. It never activates silently: you must
  set ``DATABASE_URL=sqlite:///nutrivision.db`` (or run ``init_db.py --sqlite``),
  and every screen that shows database status reports the active backend
  honestly.
* Streamlit caches the engine. If the configuration changes, the cache is
  cleared so the new URL is picked up immediately.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

import config as _config
from config import BASE_DIR

logger = logging.getLogger(__name__)

_postgres_driver_error: Optional[str] = None


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when no database URL is configured (or its driver is missing)."""


@dataclass(frozen=True)
class DatabaseStatus:
    """Result of a database connectivity + schema check."""

    ok: bool
    backend: str            # "postgresql" | "sqlite" | "not configured"
    label: str              # "Neon PostgreSQL" | "SQLite (development fallback)" | ...
    location: str           # host/database or file path — never contains credentials
    tables: tuple[str, ...]
    missing_tables: tuple[str, ...]
    error: str = ""


#: The tables Phase 2 must create.
EXPECTED_TABLES: Final[tuple[str, ...]] = ("users", "meals", "predictions", "recommendations")

#: Local SQLite file used by the development fallback.
SQLITE_PATH: Final[Path] = BASE_DIR / "nutrivision.db"
SQLITE_URL: Final[str] = f"sqlite:///{SQLITE_PATH}"


# --------------------------------------------------------------------------- #
# URL resolution
# --------------------------------------------------------------------------- #
def _normalise_postgres_url(url: str) -> str:
    """Adapt provider-style URLs to the drivers this project installs.

    Neon, Heroku and Render hand out ``postgres://`` or plain ``postgresql://``
    URLs. SQLAlchemy needs an explicit driver, so both are rewritten to
    ``postgresql+psycopg2://``.
    """
    clean = url.strip().strip('"').strip("'")
    if clean.startswith("postgres://"):
        clean = "postgresql+psycopg2://" + clean[len("postgres://"):]
    elif clean.startswith("postgresql://"):
        clean = "postgresql+psycopg2://" + clean[len("postgresql://"):]
    return clean


def get_database_url() -> str:
    """Return the configured database URL.

    The environment variable is checked first so the app can switch backends at
    runtime (the "Use SQLite for this session" button); the value loaded from
    ``.env`` / Streamlit secrets through ``config`` is the fallback.

    Raises:
        DatabaseNotConfiguredError: if nothing is configured. The message tells
            the user exactly what to do (Neon URL, or the SQLite fallback).
    """
    raw = (os.environ.get("DATABASE_URL") or _config.DATABASE_URL or "").strip()
    if not raw.strip():
        raise DatabaseNotConfiguredError(
            "No database is configured. Create a free Neon PostgreSQL project "
            "(https://neon.tech), then set DATABASE_URL in your local .env file or in "
            "Streamlit secrets. For UI development without a database you can "
            "deliberately opt into the local fallback by setting "
            "DATABASE_URL=sqlite:///nutrivision.db"
        )
    if raw.startswith("sqlite"):
        return raw
    if not raw.startswith("postgres"):
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must be a PostgreSQL connection string "
            f"(it starts with '{raw.split(':', 1)[0]}'). "
            "Use sqlite:///nutrivision.db for the development fallback."
        )
    return _normalise_postgres_url(raw)


def describe_database(url: str) -> dict[str, str]:
    """Return human-readable backend details. Never returns credentials."""
    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1] or str(SQLITE_PATH)
        return {
            "backend": "sqlite",
            "label": "SQLite (development fallback)",
            "location": path,
        }
    try:
        engine = create_engine(url)
        return {
            "backend": "postgresql",
            "label": "Neon PostgreSQL",
            "location": f"{engine.url.host or 'host'}/{engine.url.database or 'database'}",
        }
    except SQLAlchemyError as exc:  # malformed URL
        return {"backend": "postgresql", "label": "Neon PostgreSQL", "location": f"invalid ({exc.__class__.__name__})"}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def _build_engine(url: str) -> Engine:
    global _postgres_driver_error

    if url.startswith("sqlite"):
        engine = create_engine(url, future=True)

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _record):  # pragma: no cover
            """SQLite ignores foreign keys unless they are switched on per connection."""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        logger.info("SQLite development fallback in use: %s", url)
        return engine

    try:
        engine = create_engine(
            url,
            future=True,
            pool_pre_ping=True,   # reconnects automatically if Neon wakes from idle
            pool_size=5,
            max_overflow=10,
        )
    except ModuleNotFoundError as exc:  # psycopg2 missing
        _postgres_driver_error = (
            f"PostgreSQL driver unavailable ({exc}). Install it with: "
            "pip install psycopg2-binary"
        )
        raise DatabaseNotConfiguredError(_postgres_driver_error) from exc

    logger.info("PostgreSQL engine created for %s", describe_database(url)["location"])
    return engine


def get_engine(force_new: bool = False) -> Engine:
    """Return the shared SQLAlchemy engine, creating it once per process."""
    global _ENGINE
    if force_new or _ENGINE is None:
        url = get_database_url()
        _ENGINE = _build_engine(url)
        _URL_IN_USE = url
    return _ENGINE


def reset_engine() -> None:
    """Dispose and forget the cached engine (used after a configuration change)."""
    global _ENGINE, _URL_IN_USE
    if _ENGINE is not None:
        _ENGINE.dispose()
        logger.info("Database engine disposed")
    _ENGINE = None
    _URL_IN_USE = ""


_ENGINE: Optional[Engine] = None
_URL_IN_USE: str = ""


def get_session_factory() -> sessionmaker[Session]:
    """Return a session factory bound to the active engine."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def get_session() -> Session:
    """Return a new ORM session. Callers are responsible for closing it.

    Prefer :func:`session_scope` when possible, or ``repository.session_scope``
    in application code.
    """
    return get_session_factory()()


# --------------------------------------------------------------------------- #
# Health / schema check
# --------------------------------------------------------------------------- #
def check_database() -> DatabaseStatus:
    """Connect, list tables and report what (if anything) is missing.

    Never raises: returns ``ok=False`` with a readable ``error`` instead, so the
    Streamlit UI can stay usable while the database is unavailable.
    """
    try:
        url = get_database_url()
    except DatabaseNotConfiguredError as exc:
        return DatabaseStatus(False, "not configured", "Not configured", "—", (), EXPECTED_TABLES, str(exc))

    meta = describe_database(url)
    try:
        engine = get_engine()
        with engine.connect() as connection:
            existing = set(connection.dialect.get_table_names(connection))
        tables = tuple(name for name in EXPECTED_TABLES if name in existing)
        missing = tuple(name for name in EXPECTED_TABLES if name not in existing)
        return DatabaseStatus(
            ok=not missing,
            backend=meta["backend"],
            label=meta["label"],
            location=meta["location"],
            tables=tables,
            missing_tables=missing,
        )
    except (SQLAlchemyError, ModuleNotFoundError, DatabaseNotConfiguredError) as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return DatabaseStatus(
            ok=False,
            backend=meta["backend"],
            label=meta["label"],
            location=meta["location"],
            tables=(),
            missing_tables=EXPECTED_TABLES,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def ping() -> bool:
    """Return ``True`` if a trivial query succeeds."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        logger.warning("Database ping failed: %s", exc)
        return False
