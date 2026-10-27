"""Phase 2 tests: schema, constraints, relationships and the repository layer.

Every test runs against a temporary SQLite file (created by the fixture), so the
suite never touches a real Neon database. The production path — PostgreSQL via
``DATABASE_URL`` — is exercised by ``python database/init_db.py --check``.

Run with:

    pytest -v
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

import config
from database import repository as repo
from database.connection import (
    EXPECTED_TABLES,
    DatabaseNotConfiguredError,
    check_database,
    get_database_url,
    reset_engine,
)
from models import ALL_MODELS, Base, Meal, Prediction, Recommendation, User


def _meal(user_id: int, day_offset: int = 0, calories: float = 500.0, **overrides) -> Meal:
    stamp = datetime.now(timezone.utc) - timedelta(days=day_offset)
    values = {
        "user_id": user_id,
        "timestamp": stamp,
        "food_name": "Chapati (whole wheat roti)",
        "portion_grams": 100.0,
        "calories": calories,
        "protein": 9.6,
        "fat": 7.1,
        "carbohydrates": 49.0,
        "fiber": 6.1,
        "health_score": 72.0,
    }
    values.update(overrides)
    return Meal(**values)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_all_four_tables_are_created(sqlite_db) -> None:
    assert set(Base.metadata.tables) == set(EXPECTED_TABLES)
    status = check_database()
    assert status.ok is True
    assert status.backend == "sqlite"
    assert set(status.tables) == set(EXPECTED_TABLES)
    assert status.missing_tables == ()


def test_prediction_and_recommendation_are_registered() -> None:
    assert set(ALL_MODELS) == {User, Meal, Prediction, Recommendation}


def test_missing_schema_is_reported(sqlite_db, monkeypatch, tmp_path) -> None:
    """An empty database file must be reported as 'tables missing', not crash."""
    empty_file = tmp_path / "empty.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{empty_file}")
    reset_engine()
    status = check_database()
    assert status.ok is False
    assert set(status.missing_tables) == set(EXPECTED_TABLES)


def test_unconfigured_database_raises_helpful_error(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "DATABASE_URL", "", raising=False)
    reset_engine()
    with pytest.raises(DatabaseNotConfiguredError) as excinfo:
        get_database_url()
    message = str(excinfo.value)
    assert "neon" in message.lower()
    assert "sqlite" in message.lower()
    status = check_database()
    assert status.ok is False and status.backend == "not configured"


def test_postgres_urls_are_normalised(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@ep-cool-123.eu-central-1.aws.neon.tech/neondb")
    assert get_database_url().startswith("postgresql+psycopg2://")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host/db")
    assert get_database_url().startswith("postgresql+psycopg2://")


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
def test_meal_check_constraints_reject_bad_values(user_factory) -> None:
    user = user_factory()
    with pytest.raises(IntegrityError):
        repo.save_meal(_meal(user.user_id, portion_grams=-5))
    repo.clear_diagnostics()


def test_health_score_out_of_range_is_rejected(user_factory) -> None:
    user = user_factory()
    with pytest.raises(IntegrityError):
        repo.save_meal(_meal(user.user_id, health_score=140.0))
    repo.clear_diagnostics()


def test_user_age_constraint(sqlite_db) -> None:
    with pytest.raises(IntegrityError):
        repo.save_user(
            User(
                name="Impossible Age",
                age=0,
                gender="Other",
                height_cm=170.0,
                weight_kg=70.0,
                activity_level="Sedentary (little or no exercise)",
                diet_preference="Vegetarian",
                fitness_goal="Maintain Weight",
            )
        )
    repo.clear_diagnostics()


# --------------------------------------------------------------------------- #
# Relationships & cascade
# --------------------------------------------------------------------------- #
def test_relationships_are_wired_both_ways(user_factory) -> None:
    user = user_factory()
    meal = repo.save_meal(_meal(user.user_id))
    assert meal.user.user_id == user.user_id

    with repo.session_scope() as session:
        session.add(Prediction(user_id=user.user_id, predicted_next_week_avg_calories=1700.0))
        session.add(Recommendation(user_id=user.user_id, recommendation_text="Eat more dal."))
    with repo.session_scope() as session:
        refreshed = session.get(User, user.user_id)
        assert len(refreshed.meals) == 1
        assert len(refreshed.predictions) == 1
        assert len(refreshed.recommendations) == 1
        assert refreshed.meals[0].user is refreshed


def test_deleting_a_user_cascades_to_child_rows(user_factory) -> None:
    user = user_factory()
    repo.save_meal(_meal(user.user_id))
    repo.save_prediction(Prediction(user_id=user.user_id, predicted_next_week_avg_calories=1650.0))
    repo.save_recommendation(Recommendation(user_id=user.user_id, recommendation_text="Hydrate."))

    assert repo.table_counts()["meals"] == 1

    with repo.session_scope() as session:
        session.delete(session.get(User, user.user_id))

    counts = repo.table_counts()
    assert counts["meals"] == 0
    assert counts["predictions"] == 0
    assert counts["recommendations"] == 0


def test_foreign_key_rejects_unknown_user(user_factory) -> None:
    with pytest.raises(IntegrityError):
        repo.save_meal(_meal(user_id=999_999))
    repo.clear_diagnostics()


# --------------------------------------------------------------------------- #
# Repository: users
# --------------------------------------------------------------------------- #
def test_save_user_assigns_primary_key(user_factory) -> None:
    first = user_factory()
    second = user_factory()
    assert first.user_id is not None
    assert second.user_id == first.user_id + 1
    assert first.is_sample is False


def test_find_user_by_name_is_case_insensitive(user_factory) -> None:
    user = user_factory("Meera Nair")
    assert repo.find_user_by_name(f"  {user.name.swapcase()}  ").user_id == user.user_id
    assert repo.find_user_by_name("Nobody Here") is None


def test_update_user_whitelist_blocks_protected_columns(user_factory) -> None:
    user = user_factory()
    with pytest.raises(ValueError):
        repo.update_user(user.user_id, user_id=1234)
    with pytest.raises(ValueError):
        repo.update_user(user.user_id, created_at="2020-01-01")
    assert repo.update_user(user.user_id, weight_kg=58.5).weight_kg == 58.5


def test_update_unknown_user_returns_none(user_factory) -> None:
    assert repo.update_user(424_242, weight_kg=70.0) is None


def test_sample_users_are_hidden_from_the_user_list(user_factory) -> None:
    real = user_factory("Real Person")
    user_factory("Demo Person", is_sample=True)
    visible = repo.list_users()
    assert [u.user_id for u in visible] == [real.user_id]
    assert len(repo.list_users(include_samples=True)) == 2


def test_delete_sample_data_removes_only_samples(user_factory) -> None:
    real = user_factory("Real Person")
    sample = user_factory("Sample Demo User", is_sample=True)
    repo.save_meal(_meal(sample.user_id))
    repo.save_meal(_meal(real.user_id))

    assert repo.delete_sample_data() == 1
    assert repo.get_user(sample.user_id) is None
    assert repo.count_meals(real.user_id) == 1


def test_get_latest_user_id_prefers_recent_activity(user_factory) -> None:
    first = user_factory("Older")
    second = user_factory("Newer")
    repo.save_meal(_meal(first.user_id, day_offset=0))   # meal today for the older profile
    assert repo.get_latest_user_id() == first.user_id
    repo.save_meal(_meal(second.user_id, day_offset=-1))  # future-dated meal, newest timestamp
    assert repo.get_latest_user_id() == second.user_id


# --------------------------------------------------------------------------- #
# Repository: meals and daily aggregation
# --------------------------------------------------------------------------- #
def test_daily_totals_aggregate_by_calendar_day(user_factory) -> None:
    user = user_factory()
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)

    repo.save_meal(_meal(user.user_id, calories=400.0, protein=10.0, fiber=3.0, health_score=60.0,
                         timestamp=today.replace(hour=8, minute=0)))
    repo.save_meal(_meal(user.user_id, calories=600.0, protein=20.0, fiber=5.0, health_score=80.0,
                         timestamp=today.replace(hour=14, minute=30)))
    repo.save_meal(_meal(user.user_id, calories=300.0, timestamp=yesterday.replace(hour=20, minute=0)))

    rows = repo.get_daily_totals(user.user_id)
    assert len(rows) == 2

    _yday, _today = rows
    assert _today["meals"] == 2
    assert _today["calories"] == 1000.0
    assert _today["protein"] == 30.0
    assert _today["fiber"] == 8.0
    assert _today["avg_health_score"] == 70.0        # (60 + 80) / 2
    assert _yday["date"] < _today["date"]
    assert _today["date"] in (date.today(), (today - timedelta(seconds=1)).date())


def test_daily_totals_respect_date_range(user_factory) -> None:
    user = user_factory()
    for offset in range(5):
        repo.save_meal(_meal(user.user_id, day_offset=offset, calories=100.0 * (offset + 1)))

    rows = repo.get_daily_totals(user.user_id, start=date.today() - timedelta(days=1))
    assert len(rows) <= 2
    assert all(row["date"] >= date.today() - timedelta(days=1) for row in rows)


def test_daily_totals_for_user_without_meals_is_empty(user_factory) -> None:
    assert repo.get_daily_totals(user_factory().user_id) == []


def test_list_and_count_meals(user_factory) -> None:
    user = user_factory()
    for offset in range(3):
        repo.save_meal(_meal(user.user_id, day_offset=offset))
    assert repo.count_meals(user.user_id) == 3
    assert len(repo.list_recent_meals(user.user_id, limit=2)) == 2
    assert len(repo.list_meals(user.user_id, limit=1)) == 1


def test_delete_meal_reports_what_happened(user_factory) -> None:
    meal = repo.save_meal(_meal(user_factory().user_id))
    assert repo.delete_meal(meal.meal_id) is True
    assert repo.delete_meal(meal.meal_id) is False


def test_meal_calories_per_100g_helper(user_factory) -> None:
    meal = repo.save_meal(_meal(user_factory().user_id, portion_grams=150.0, calories=297.0))
    assert meal.calories_per_100g == pytest.approx(198.0, abs=0.1)


# --------------------------------------------------------------------------- #
# Repository: predictions & recommendations
# --------------------------------------------------------------------------- #
def test_latest_prediction_is_ordered_by_data_date(user_factory) -> None:
    user = user_factory()
    older = repo.save_prediction(
        Prediction(user_id=user.user_id,
                   prediction_date=datetime.now(timezone.utc) - timedelta(days=2),
                   predicted_next_week_avg_calories=1500.0)
    )
    newer = repo.save_prediction(
        Prediction(user_id=user.user_id, prediction_date=datetime.now(timezone.utc),
                   predicted_next_week_avg_calories=1650.0, model_name="xgboost",
                   training_samples=14, mae=90.0, feature_metadata={"features": ["lag_1"]})
    )
    latest = repo.get_latest_prediction(user.user_id)
    assert latest.prediction_id == newer.prediction_id
    assert latest.model_name == "xgboost"
    assert latest.feature_metadata == {"features": ["lag_1"]}
    assert [p.prediction_id for p in repo.list_predictions(user.user_id)] == [newer.prediction_id, older.prediction_id]


def test_latest_recommendation_roundtrip(user_factory) -> None:
    user = user_factory()
    repo.save_recommendation(Recommendation(user_id=user.user_id, recommendation_text="First", prompt="q1"))
    repo.save_recommendation(Recommendation(user_id=user.user_id, recommendation_text="Second"))
    latest = repo.get_latest_recommendation(user.user_id)
    assert latest.recommendation_text == "Second"
    assert latest.source == "gemini-2.5-flash"
    assert len(repo.list_recommendations(user.user_id)) == 2


def test_predictions_are_isolated_per_user(user_factory) -> None:
    first, second = user_factory(), user_factory()
    repo.save_prediction(Prediction(user_id=first.user_id, predicted_next_week_avg_calories=1600.0))
    assert repo.get_latest_prediction(second.user_id) is None


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #
def test_missing_schema_returns_empty_results_and_records_a_diagnostic(
    monkeypatch, tmp_path
) -> None:
    """Pages opened before init_db.py must not crash the app."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'not_initialised.db'}")
    reset_engine()
    repo.clear_diagnostics()

    assert repo.list_users() == []
    assert repo.count_meals(1) == 0
    assert repo.get_daily_totals(1) == []
    assert repo.table_counts() == {}
    assert repo.get_latest_prediction(1) is None

    diagnostics = repo.get_diagnostics()
    assert diagnostics, "a diagnostic message should be recorded"
    repo.clear_diagnostics()
    assert repo.get_diagnostics() == {}


def test_repository_never_interpolates_input_into_sql(user_factory) -> None:
    """SQLAlchemy binds parameters, so a quoted name is stored verbatim."""
    hostile = "O'Brien); DROP TABLE users;--"
    created = repo.save_user(
        User(
            name=hostile,           # stored through the ORM, bypassing the display validator
            age=30,
            gender="Other",
            height_cm=170.0,
            weight_kg=70.0,
            activity_level="Sedentary (little or no exercise)",
            diet_preference="Vegetarian",
            fitness_goal="Maintain Weight",
        )
    )
    stored = repo.find_user_by_name(hostile)
    assert stored is not None and stored.name == hostile
    assert stored.user_id == created.user_id

    with repo.session_scope() as session:
        assert session.scalar(select(func.count(User.user_id))) == 1
