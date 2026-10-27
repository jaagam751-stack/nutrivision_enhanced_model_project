"""Makes the project root importable for tests run via ``pytest`` from any folder."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Shared fixtures (Phase 8 cleanup)
# --------------------------------------------------------------------------- #
# These used to live in tests/test_database.py and were re-imported by the PNIE,
# coach and profile test modules. Living in conftest.py makes pytest discover
# them automatically in every test file — no imports, no redefinition warnings.

import pytest  # noqa: E402

from database import repository as repo  # noqa: E402
from database.connection import reset_engine  # noqa: E402
from models import Base, User  # noqa: E402
from utils.helpers import build_profile  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sqlite_db(tmp_path, monkeypatch):
    """Point the app at a throw-away SQLite file and create the schema."""
    db_file = tmp_path / "test_nutrivision.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    reset_engine()
    Base.metadata.create_all(repo.get_session_factory().kw["bind"])
    yield db_file
    reset_engine()


def _alpha_suffix(index: int) -> str:
    """1 -> 'a', 2 -> 'b', ... 27 -> 'aa' (names must not contain digits)."""
    suffix = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        suffix = chr(ord("a") + remainder) + suffix
    return suffix


@pytest.fixture()
def user_factory(sqlite_db):
    """Create real users on demand. Names use letter suffixes: the profile
    validator deliberately rejects digits in names."""
    counter = {"n": 0}

    def _make(name: str = "Test User", is_sample: bool = False, **overrides) -> User:
        counter["n"] += 1
        payload = {
            "name": f"{name} {_alpha_suffix(counter['n'])}",
            "age": 30,
            "gender": "Female",
            "height_cm": 162.0,
            "weight_kg": 60.0,
            "activity_level": "Lightly active (light exercise 1-3 days/week)",
            "diet_preference": "Vegetarian",
            "fitness_goal": "Weight Loss",
        }
        payload.update(overrides)
        user = User.from_profile(build_profile(payload))
        user.is_sample = is_sample
        return repo.save_user(user)

    return _make
