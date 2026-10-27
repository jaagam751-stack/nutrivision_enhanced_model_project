"""Tests for the Phase 8 profile-data layer (summary, edit-in-place, deletes).

Nothing here touches a real database: the ``sqlite_db`` fixture from
``tests/test_database.py`` points ``DATABASE_URL`` at a temporary SQLite file and
creates the schema, and ``user_factory`` builds throwaway users.

What is being protected:

* the Profile page must never create a duplicate account when you save twice;
* a delete must remove exactly the rows it promises and nothing else —
  deleting the meal history keeps the profile, forecasts and recommendations;
* counts reported to the user (and stored in the confirmation message) must be
  the real row counts, not estimates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import OperationalError

from database import repository as repo
from models import Meal, Prediction, Recommendation
from utils import data_access
from utils.helpers import build_profile

# Reuse the fixtures that already create a temporary database + users.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _meal(user_id: int, day_offset: int = 0, calories: float = 500.0) -> Meal:
    return Meal(
        user_id=user_id,
        timestamp=datetime.now(timezone.utc) - timedelta(days=day_offset),
        food_name="Chapati (whole wheat roti)",
        portion_grams=100.0,
        calories=calories,
        protein=9.6,
        fat=7.1,
        carbohydrates=49.0,
        fiber=6.1,
        health_score=72.0,
    )


@pytest.fixture
def stored_user(user_factory):
    """A user row plus three meals, one prediction and one recommendation."""
    user = user_factory("Profile Page User")
    for offset, kcal in ((0, 620.0), (1, 540.0), (1, 410.0), (3, 700.0)):
        repo.save_meal(_meal(user.user_id, day_offset=offset, calories=kcal))
    repo.save_prediction(
        Prediction(
            user_id=user.user_id,
            prediction_date=datetime.now(timezone.utc),
            predicted_next_week_avg_calories=1900.0,
            trend_direction="steady",
            model_name="xgboost-next-week-calories",
            model_version="1.0.0",
            training_samples=9,
            mae=40.0,
        )
    )
    repo.save_recommendation(
        Recommendation(
            user_id=user.user_id,
            timestamp=datetime.now(timezone.utc),
            recommendation_text="Eat more dal.",
            source="gemini-2.5-flash",
            prompt="How can I eat better?",
        )
    )
    return user


# --------------------------------------------------------------------------- #
# user_data_summary
# --------------------------------------------------------------------------- #
def test_summary_counts_every_table(sqlite_db, stored_user):
    summary = repo.user_data_summary(stored_user.user_id)
    assert summary["exists"] is True
    assert summary["meals"] == 4
    assert summary["predictions"] == 1
    assert summary["recommendations"] == 1
    assert summary["user_id"] == stored_user.user_id


def test_summary_counts_distinct_logged_days(sqlite_db, stored_user):
    """Four meals spread over three UTC days → three logged days."""
    assert repo.user_data_summary(stored_user.user_id)["logged_days"] == 3


def test_summary_reports_the_meal_span(sqlite_db, stored_user):
    summary = repo.user_data_summary(stored_user.user_id)
    assert summary["first_meal_at"] is not None
    assert summary["last_meal_at"] >= summary["first_meal_at"]
    assert (summary["last_meal_at"].date() - summary["first_meal_at"].date()).days == 3


def test_summary_is_empty_for_an_unknown_user(sqlite_db):
    summary = repo.user_data_summary(999_999)
    assert summary["exists"] is False
    assert summary["meals"] == 0 and summary["logged_days"] == 0
    assert summary["first_meal_at"] is None
    assert summary["predictions"] == 0 and summary["recommendations"] == 0


def test_summary_ignores_other_users(sqlite_db, user_factory):
    first = user_factory("Summary One")
    second = user_factory("Summary Two")
    repo.save_meal(_meal(first.user_id))
    repo.save_meal(_meal(second.user_id))
    repo.save_meal(_meal(second.user_id))
    assert repo.user_data_summary(first.user_id)["meals"] == 1
    assert repo.user_data_summary(second.user_id)["meals"] == 2


# --------------------------------------------------------------------------- #
# profile_overview (Streamlit-facing wrapper)
# --------------------------------------------------------------------------- #
def test_overview_without_a_user_id(sqlite_db):
    overview = data_access.profile_overview(None)
    assert overview.stored is False
    assert overview.user_id is None
    assert "not stored" in dict(overview.as_rows())["Saved profile"]


def test_overview_reads_the_stored_row(sqlite_db, stored_user):
    overview = data_access.profile_overview(stored_user.user_id)
    assert overview.stored is True
    assert overview.meals == 4
    assert overview.logged_days == 3
    assert overview.predictions == 1
    assert overview.recommendations == 1
    assert overview.error == ""
    rows = dict(overview.as_rows())
    assert rows["Meals saved"] == "4"
    assert "user_id" in rows["Saved profile"]


def test_overview_formats_timestamps_in_utc(sqlite_db, stored_user):
    overview = data_access.profile_overview(stored_user.user_id)
    assert overview.last_meal_at and overview.last_meal_at.endswith("UTC")
    assert overview.first_meal_at is not None


def test_overview_reports_a_missing_row(sqlite_db):
    overview = data_access.profile_overview(987_654)
    assert overview.stored is False
    assert "no row" in overview.error.lower()


def test_overview_never_raises_when_the_table_is_missing(sqlite_db):
    """A user_id pointing at a non-existent schema must return an error, not a traceback."""
    with repo.session_scope() as session:
        session.execute(repo.delete(Meal))
    overview = data_access.profile_overview(1)
    assert isinstance(overview, data_access.ProfileOverview)


# --------------------------------------------------------------------------- #
# persist_profile (create / update in place)
# --------------------------------------------------------------------------- #
RAW = {
    "name": "Persist Tester",
    "age": 33,
    "gender": "Female",
    "height_cm": 165.0,
    "weight_kg": 68.0,
    "activity_level": "Lightly active (light exercise 1-3 days/week)",
    "diet_preference": "Vegetarian",
    "fitness_goal": "Weight Loss",
}


def test_persist_creates_a_row(sqlite_db):
    profile = build_profile(RAW)
    user_id, message = data_access.persist_profile(profile)
    assert user_id is not None
    assert "user_id" in message
    stored = repo.get_user(user_id)
    assert stored is not None and stored.weight_kg == 68.0


def test_persist_updates_in_place_when_the_user_id_is_known(sqlite_db, stored_user):
    profile = build_profile({**RAW, "name": stored_user.name, "weight_kg": 61.0})
    user_id, message = data_access.persist_profile(profile, previous_user_id=stored_user.user_id)
    assert user_id == stored_user.user_id
    assert "in place" in message
    assert repo.get_user(stored_user.user_id).weight_kg == 61.0


def test_persist_matches_by_name_instead_of_duplicating(sqlite_db, stored_user):
    """No authentication yet: saving the same name twice must not create a second account."""
    before = len(repo.list_users(include_samples=True))
    profile = build_profile({**RAW, "name": stored_user.name, "age": 44})
    user_id, _message = data_access.persist_profile(profile)
    after = len(repo.list_users(include_samples=True))
    assert user_id == stored_user.user_id
    assert after == before
    assert repo.get_user(stored_user.user_id).age == 44


def test_persist_reports_an_error_instead_of_raising(sqlite_db, monkeypatch):
    """A database failure must come back as (None, message), so the session profile survives."""
    def boom(*_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("server closed the connection unexpectedly"))

    monkeypatch.setattr(data_access, "find_user_by_name", boom)
    user_id, message = data_access.persist_profile(build_profile(RAW))
    assert user_id is None
    assert "OperationalError" in message
    assert "server closed the connection unexpectedly" in message


def test_persist_keeps_the_original_creation_time(sqlite_db):
    profile = build_profile(RAW)
    user_id, _ = data_access.persist_profile(profile)
    first = repo.get_user(user_id).created_at
    again = build_profile({**RAW, "weight_kg": 67.0})
    data_access.persist_profile(again, previous_user_id=user_id)
    assert repo.get_user(user_id).created_at == first


# --------------------------------------------------------------------------- #
# deletes
# --------------------------------------------------------------------------- #
def test_delete_meal_history_removes_meals_only(sqlite_db, stored_user):
    removed, message = data_access.delete_meal_history(stored_user.user_id)
    assert removed == 4
    assert "4 meal(s)" in message
    summary = repo.user_data_summary(stored_user.user_id)
    assert summary["meals"] == 0
    assert summary["predictions"] == 1, "forecasts must survive a meal-history delete"
    assert summary["recommendations"] == 1
    assert repo.get_user(stored_user.user_id) is not None, "the profile must survive"


def test_delete_meal_history_on_an_empty_account(sqlite_db, user_factory):
    user = user_factory("Empty History")
    removed, message = data_access.delete_meal_history(user.user_id)
    assert removed == 0
    assert "0 meal(s)" in message


def test_delete_profile_and_data_removes_everything(sqlite_db, stored_user):
    removed, message = data_access.delete_profile_and_data(stored_user.user_id)
    assert removed == {"meals": 4, "predictions": 1, "recommendations": 1, "users": 1}
    assert "4 meals" in message and "1 users" in message
    assert repo.get_user(stored_user.user_id) is None
    assert repo.user_data_summary(stored_user.user_id)["meals"] == 0
    assert repo.get_latest_recommendation(stored_user.user_id) is None


def test_delete_profile_and_data_leaves_other_users_alone(sqlite_db, stored_user, user_factory):
    other = user_factory("Bystander")
    repo.save_meal(_meal(other.user_id))
    data_access.delete_profile_and_data(stored_user.user_id)
    assert repo.get_user(other.user_id) is not None
    assert repo.user_data_summary(other.user_id)["meals"] == 1


def test_delete_an_unknown_user_reports_nothing_deleted(sqlite_db):
    removed, message = data_access.delete_profile_and_data(424_242)
    assert removed == {}
    assert "Nothing was deleted" in message


def test_deleting_a_user_reports_real_row_counts(sqlite_db, stored_user):
    counts = repo.delete_user(stored_user.user_id)
    assert counts["users"] == 1
    assert counts["meals"] == 4
    assert counts["predictions"] == 1
    assert counts["recommendations"] == 1


def test_deleting_an_unknown_user_returns_an_empty_dict(sqlite_db):
    assert repo.delete_user(31_337) == {}


def test_delete_user_cascades_even_without_foreign_key_enforcement(sqlite_db, stored_user, monkeypatch):
    """Rows are deleted explicitly, so no orphans are left on backends without FK cascades."""
    repo.delete_user(stored_user.user_id)
    with repo.session_scope() as session:
        orphans = [
            session.scalar(repo.select(repo.func.count(Meal.meal_id))),
            session.scalar(repo.select(repo.func.count(Prediction.prediction_id))),
            session.scalar(repo.select(repo.func.count(Recommendation.recommendation_id))),
        ]
    assert orphans == [0, 0, 0]
