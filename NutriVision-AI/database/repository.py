"""Repository layer — every database read/write in the app goes through here.

Why a repository?
-----------------
* Pages and future modules never hand-write queries, so SQL stays in one place.
* All statements use SQLAlchemy's parameter binding (never string formatting),
  which protects against SQL injection.
* Local SQLite sessions need a commit, while a bare PostgreSQL connection does
  not — the callbacks below make both behave identically.
* Reading a table that does not exist yet (Phase 3+ pages opened before
  ``init_db.py`` was run) returns empty results instead of crashing the app,
  while still logging a warning.

Session handling
----------------
Two APIs are provided:

* ``session_scope()`` — a context manager for direct ORM work.
* ``db_session(func)`` — a decorator that injects a session and guarantees
  commit/rollback/close, used by every public function in this module.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.exc import NoSuchTableError, OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import Session

from database.connection import DatabaseNotConfiguredError, get_session_factory
from models import Meal, Prediction, Recommendation, User

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Diagnostics keys returned by ``get_diagnostics`` for expected schema issues.
SCHEMA_WARNING_KEYS = frozenset({"last_write_error"})


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session: commit on success, rollback on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _is_schema_error(exc: Exception) -> bool:
    """True if the failure means 'table/column missing' or 'db not created yet'."""
    if isinstance(exc, DatabaseNotConfiguredError):
        return True
    text = str(exc).lower()
    if isinstance(exc, (ProgrammingError, OperationalError)) and (
        "does not exist" in text
        or "undefined table" in text
        or "no such table" in text
        or "cannot operate on a closed database" in text
    ):
        return True
    return isinstance(exc, NoSuchTableError)


_DIAGNOSTICS: dict[str, str] = {}


def _record_error(operation: str, exc: Exception) -> None:
    message = f"{operation} failed: {exc.__class__.__name__}: {exc}"
    _DIAGNOSTICS["last_write_error"] = message
    logger.warning(message)


def get_diagnostics() -> dict[str, str]:
    """Return a copy of the latest non-fatal database problems (for the UI)."""
    return dict(_DIAGNOSTICS)


def clear_diagnostics() -> None:
    _DIAGNOSTICS.clear()


def db_session(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run a repository function inside a managed session.

    The wrapped function receives ``session`` as its first argument. On a schema
    or configuration error the call returns the function's declared
    ``empty_result`` (default ``None``) and records a diagnostic instead of
    raising, so pages keep working with an empty database.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            with session_scope() as session:
                return func(session, *args, **kwargs)
        except SQLAlchemyError as exc:
            if _is_schema_error(exc):
                _record_error(func.__name__, exc)
                fallback = kwargs.pop("_empty", None)
                if fallback is None:
                    fallback = _EMPTY_RESULTS.get(func.__name__, None)
                return fallback
            logger.exception("Database error in %s", func.__name__)
            raise

    return wrapper


#: What each read helper returns when the schema is missing.
_EMPTY_RESULTS: dict[str, Any] = {
    "get_user": None,
    "find_user_by_name": None,
    "list_users": [],
    "get_latest_user_id": None,
    "count_meals": 0,
    "get_daily_totals": [],
    "list_meals": [],
    "list_recent_meals": [],
    "get_latest_prediction": None,
    "list_predictions": [],
    "get_latest_recommendation": None,
    "list_recommendations": [],
    "table_counts": {},
}


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
@db_session
def save_user(session: Session, user: User) -> User:
    """Insert a new user and return it (with ``user_id`` populated)."""
    session.add(user)
    session.flush()          # assigns the primary key before commit
    session.refresh(user)
    logger.info("User saved | user_id=%s name=%s", user.user_id, user.name)
    return user


@db_session
def update_user(session: Session, user_id: int, /, **fields: Any) -> Optional[User]:
    """Update selected columns of an existing user.

    ``user_id`` is positional-only on purpose: passing it as a keyword lands in
    ``fields`` and is rejected by the whitelist below, so the primary key can
    never be overwritten by accident.
    """
    allowed = {
        "name",
        "age",
        "gender",
        "height_cm",
        "weight_kg",
        "activity_level",
        "diet_preference",
        "fitness_goal",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Cannot update unknown user column(s): {sorted(unknown)}")

    user = session.get(User, user_id)
    if user is None:
        logger.warning("update_user: user_id=%s not found", user_id)
        return None
    for key, value in fields.items():
        setattr(user, key, value)
    session.flush()
    session.refresh(user)
    logger.info("User updated | user_id=%s fields=%s", user_id, sorted(fields))
    return user


@db_session
def get_user(session: Session, user_id: int) -> Optional[User]:
    """Fetch one user by primary key."""
    return session.get(User, user_id)


@db_session
def find_user_by_name(session: Session, name: str) -> Optional[User]:
    """Fetch one user by exact name (case-insensitive).

    Phase 2 has no authentication, so this is how a returning user is matched to
    their saved profile. A login system would replace it later.
    """
    statement = select(User).where(func.lower(User.name) == name.strip().lower()).limit(1)
    return session.scalars(statement).first()


@db_session
def list_users(session: Session, limit: int = 50, include_samples: bool = False) -> list[User]:
    """List the most recently created users (for the "continue as" selector).

    Sample rows inserted by ``init_db.py --sample-data`` are excluded unless
    ``include_samples=True`` is passed explicitly.
    """
    statement = select(User)
    if not include_samples:
        statement = statement.where(User.is_sample.is_(False))
    statement = statement.order_by(User.created_at.desc()).limit(limit)
    return list(session.scalars(statement))


@db_session
def get_latest_user_id(session: Session) -> Optional[int]:
    """Return the id of the most recently active user, or ``None``.

    'Most recently active' = newest meal first, falling back to newest profile.
    Used by the sidebar to offer "continue as <name>" when a returning user
    opens the app.
    """
    latest_meal = session.scalars(
        select(Meal).order_by(Meal.timestamp.desc()).limit(1)
    ).first()
    if latest_meal is not None:
        return latest_meal.user_id
    latest_user = session.scalars(select(User).order_by(User.created_at.desc()).limit(1)).first()
    return latest_user.user_id if latest_user else None


# --------------------------------------------------------------------------- #
# Meals
# --------------------------------------------------------------------------- #
@db_session
def save_meal(session: Session, meal: Meal) -> Optional[Meal]:
    """Insert one meal. Only called after the user clicks Save Meal (Phase 5)."""
    session.add(meal)
    session.flush()
    session.refresh(meal)
    logger.info(
        "Meal saved | meal_id=%s user_id=%s food=%s portion=%sg kcal=%.0f",
        meal.meal_id, meal.user_id, meal.food_name, meal.portion_grams, meal.calories,
    )
    return meal


@db_session
def save_meals_bulk(session: Session, meals: Sequence[Meal]) -> int:
    """Insert many meals at once (used by ``init_db.py --sample-data``)."""
    if not meals:
        return 0
    session.add_all(list(meals))
    session.flush()
    logger.info("Bulk inserted %s meal rows", len(meals))
    return len(meals)


@db_session
def list_meals(
    session: Session,
    user_id: int,
    start: Optional[date] = None,
    end: Optional[date] = None,
    limit: Optional[int] = None,
) -> list[Meal]:
    """List a user's meals, optionally filtered by an inclusive date range."""
    statement = select(Meal).where(Meal.user_id == user_id)
    if start is not None:
        statement = statement.where(
            Meal.timestamp >= datetime.combine(start, time.min, tzinfo=timezone.utc)
        )
    if end is not None:
        statement = statement.where(
            Meal.timestamp <= datetime.combine(end, time.max, tzinfo=timezone.utc)
        )
    statement = statement.order_by(Meal.timestamp.desc())
    if limit:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


@db_session
def list_recent_meals(session: Session, user_id: int, limit: int = 8) -> list[Meal]:
    """Return the most recent meals (dashboard 'Recent meals' card)."""
    statement = select(Meal).where(Meal.user_id == user_id).order_by(Meal.timestamp.desc()).limit(limit)
    return list(session.scalars(statement))


@db_session
def get_meal(session: Session, meal_id: int) -> Optional[Meal]:
    """Fetch one meal by primary key (used by delete flows)."""
    return session.get(Meal, meal_id)


@db_session
def delete_meal(session: Session, meal_id: int) -> bool:
    """Delete one meal. Returns ``True`` if a row was removed."""
    result = session.execute(delete(Meal).where(Meal.meal_id == meal_id))
    logger.info("Meal delete requested | meal_id=%s removed=%s", meal_id, result.rowcount)
    return bool(result.rowcount)


@db_session
def count_meals(session: Session, user_id: int) -> int:
    """Number of saved meals for a user (used by the PNIE data-sufficiency check)."""
    return int(session.scalar(select(func.count(Meal.meal_id)).where(Meal.user_id == user_id)) or 0)


@db_session
def find_duplicate_meal(
    session: Session,
    user_id: int,
    food_name: str,
    portion_grams: float,
    calories: float,
    within_seconds: int = 120,
    portion_tolerance: float = 0.5,
) -> Optional[Meal]:
    """Find an identical meal saved moments ago (idempotency guard).

    Streamlit re-executes the script on every interaction, so a double click or a
    rerun racing a click could otherwise insert the same meal twice. A meal counts
    as a duplicate when the same user saved the same food, with the same portion
    (within ``portion_tolerance`` grams) and the same calorie value, inside the
    ``within_seconds`` window.

    Returns the existing row, or ``None`` when this is genuinely a new meal.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, int(within_seconds)))
    statement = (
        select(Meal)
        .where(Meal.user_id == user_id)
        .where(Meal.timestamp >= cutoff)
        .where(func.lower(Meal.food_name) == food_name.strip().lower())
        .where(Meal.portion_grams >= float(portion_grams) - portion_tolerance)
        .where(Meal.portion_grams <= float(portion_grams) + portion_tolerance)
        .where(Meal.calories >= float(calories) - 0.5)
        .where(Meal.calories <= float(calories) + 0.5)
        .order_by(Meal.timestamp.desc())
        .limit(1)
    )
    duplicate = session.scalars(statement).first()
    if duplicate is not None:
        logger.warning(
            "Duplicate meal prevented | user_id=%s food=%s portion=%sg meal_id=%s",
            user_id, food_name, portion_grams, duplicate.meal_id,
        )
    return duplicate


@db_session
def get_daily_totals(
    session: Session,
    user_id: int,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> list[dict[str, Any]]:
    """Aggregate meals per calendar day.

    Returns a list of dicts sorted by date::

        {"date": date(2026, 10, 14), "meals": 3, "calories": 1450.0,
         "protein": 62.0, "fat": 48.0, "carbohydrates": 190.0,
         "fiber": 14.0, "avg_health_score": 68.4}

    This is the exact input the PNIE feature engineering (Phase 6) and the
    dashboard charts (Phase 5) consume, so the aggregation lives here rather
    than being re-implemented per page.

    Implementation note: date bucketing is done in Python instead of with
    ``date_trunc``, because SQLite and PostgreSQL handle timezone conversion
    differently and the row volumes here are small.
    """
    statement = select(Meal).where(Meal.user_id == user_id).order_by(Meal.timestamp)
    if start is not None:
        statement = statement.where(
            Meal.timestamp >= datetime.combine(start, time.min, tzinfo=timezone.utc)
        )
    if end is not None:
        statement = statement.where(
            Meal.timestamp <= datetime.combine(end, time.max, tzinfo=timezone.utc)
        )

    buckets: dict[date, dict[str, Any]] = {}
    for meal in session.scalars(statement):
        stamp = meal.timestamp
        if stamp.tzinfo is None:                      # SQLite returns naive datetimes
            stamp = stamp.replace(tzinfo=timezone.utc)
        day = stamp.date()
        bucket = buckets.setdefault(
            day,
            {
                "date": day,
                "meals": 0,
                "calories": 0.0,
                "protein": 0.0,
                "fat": 0.0,
                "carbohydrates": 0.0,
                "fiber": 0.0,
                "_score_total": 0.0,
            },
        )
        bucket["meals"] += 1
        bucket["calories"] += float(meal.calories or 0)
        bucket["protein"] += float(meal.protein or 0)
        bucket["fat"] += float(meal.fat or 0)
        bucket["carbohydrates"] += float(meal.carbohydrates or 0)
        bucket["fiber"] += float(meal.fiber or 0)
        bucket["_score_total"] += float(meal.health_score or 0)

    rows: list[dict[str, Any]] = []
    for day in sorted(buckets):
        bucket = buckets[day]
        rows.append(
            {
                "date": day,
                "meals": bucket["meals"],
                "calories": round(bucket["calories"], 1),
                "protein": round(bucket["protein"], 1),
                "fat": round(bucket["fat"], 1),
                "carbohydrates": round(bucket["carbohydrates"], 1),
                "fiber": round(bucket["fiber"], 1),
                "avg_health_score": round(bucket["_score_total"] / bucket["meals"], 1),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
@db_session
def save_prediction(session: Session, prediction: Prediction) -> Optional[Prediction]:
    """Store a PNIE forecast together with its model metadata."""
    session.add(prediction)
    session.flush()
    session.refresh(prediction)
    logger.info(
        "Prediction saved | prediction_id=%s user_id=%s value=%s",
        prediction.prediction_id, prediction.user_id, prediction.predicted_next_week_avg_calories,
    )
    return prediction


@db_session
def list_predictions(session: Session, user_id: int, limit: int = 20) -> list[Prediction]:
    """List a user's past predictions, newest first."""
    statement = (
        select(Prediction)
        .where(Prediction.user_id == user_id)
        .order_by(Prediction.prediction_date.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


@db_session
def get_latest_prediction(session: Session, user_id: int) -> Optional[Prediction]:
    """Return the most recent prediction for a user, or ``None``.

    'Latest' means the prediction with the newest *data* date, not the newest
    row: re-running PNIE on the same day updates the existing row instead of
    stacking duplicates.
    """
    statement = (
        select(Prediction)
        .where(Prediction.user_id == user_id)
        .order_by(Prediction.prediction_date.desc(), Prediction.prediction_id.desc())
        .limit(1)
    )
    return session.scalars(statement).first()


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
@db_session
def save_recommendation(session: Session, recommendation: Recommendation) -> Optional[Recommendation]:
    """Store one AI coaching message."""
    session.add(recommendation)
    session.flush()
    session.refresh(recommendation)
    logger.info(
        "Recommendation saved | recommendation_id=%s user_id=%s source=%s",
        recommendation.recommendation_id, recommendation.user_id, recommendation.source,
    )
    return recommendation


@db_session
def list_recommendations(session: Session, user_id: int, limit: int = 20) -> list[Recommendation]:
    """List a user's stored recommendations, newest first."""
    statement = (
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.timestamp.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


@db_session
def get_latest_recommendation(session: Session, user_id: int) -> Optional[Recommendation]:
    """Return the most recent stored recommendation, or ``None``."""
    statement = (
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.timestamp.desc(), Recommendation.recommendation_id.desc())
        .limit(1)
    )
    return session.scalars(statement).first()


# --------------------------------------------------------------------------- #
# Maintenance helpers (used by init_db and the tests)
# --------------------------------------------------------------------------- #
@db_session
def table_counts(session: Session) -> dict[str, int]:
    """Row counts per table — shown in the Streamlit database panel."""
    return {
        "users": int(session.scalar(select(func.count(User.user_id))) or 0),
        "meals": int(session.scalar(select(func.count(Meal.meal_id))) or 0),
        "predictions": int(session.scalar(select(func.count(Prediction.prediction_id))) or 0),
        "recommendations": int(session.scalar(select(func.count(Recommendation.recommendation_id))) or 0),
    }


@db_session
def delete_sample_data(session: Session) -> int:
    """Delete every sample user (detaching their rows via FK cascade).

    Deletion order is explicit so it also works on backends where
    ``ON DELETE CASCADE`` is not applied (older SQLite builds without
    ``PRAGMA foreign_keys=ON``).
    """
    sample_ids = list(
        session.scalars(select(User.user_id).where(User.is_sample.is_(True)))
    )
    if not sample_ids:
        return 0
    session.execute(delete(Meal).where(Meal.user_id.in_(sample_ids)))
    session.execute(delete(Prediction).where(Prediction.user_id.in_(sample_ids)))
    session.execute(delete(Recommendation).where(Recommendation.user_id.in_(sample_ids)))
    session.execute(delete(User).where(User.user_id.in_(sample_ids)))
    logger.warning("Deleted %s sample user(s) and their rows", len(sample_ids))
    return len(sample_ids)


# --------------------------------------------------------------------------- #
# Phase 8 — profile data management (summary + explicit deletes)
# --------------------------------------------------------------------------- #
@db_session
def user_data_summary(session: Session, user_id: int) -> dict[str, Any]:
    """Counts and date span of everything stored for one user.

    Used by the Profile page so the user can see *exactly* what is kept about
    them before deleting anything. ``exists`` says whether the ``users`` row
    itself is still there — the counts alone cannot tell an empty account from a
    deleted one. Day counts are bucketed with SQL ``date()`` (UTC on both
    backends — the same convention as ``get_daily_totals``).
    """
    meals = int(session.scalar(select(func.count(Meal.meal_id)).where(Meal.user_id == user_id)) or 0)
    predictions = int(
        session.scalar(select(func.count(Prediction.prediction_id)).where(Prediction.user_id == user_id)) or 0
    )
    recommendations = int(
        session.scalar(
            select(func.count(Recommendation.recommendation_id)).where(Recommendation.user_id == user_id)
        )
        or 0
    )
    first_meal = session.scalar(select(func.min(Meal.timestamp)).where(Meal.user_id == user_id))
    last_meal = session.scalar(select(func.max(Meal.timestamp)).where(Meal.user_id == user_id))
    logged_days = int(
        session.scalar(
            select(func.count(func.distinct(func.date(Meal.timestamp)))).where(Meal.user_id == user_id)
        )
        or 0
    )
    first_seen = session.scalar(select(func.min(User.created_at)).where(User.user_id == user_id))
    return {
        "user_id": user_id,
        "exists": session.get(User, user_id) is not None,
        "meals": meals,
        "logged_days": logged_days,
        "predictions": predictions,
        "recommendations": recommendations,
        "first_meal_at": first_meal,
        "last_meal_at": last_meal,
        "profile_created_at": first_seen,
    }


@db_session
def delete_meals_for_user(session: Session, user_id: int) -> int:
    """Delete every meal of one user (the profile stays). Returns rows removed.

    Forecasts and recommendations are *kept*: they are historical statements
    about data that used to exist, and the UI labels them with their date.
    """
    result = session.execute(delete(Meal).where(Meal.user_id == user_id))
    removed = int(result.rowcount or 0)
    logger.warning("Deleted %s meal(s) for user_id=%s", removed, user_id)
    return removed


@db_session
def delete_user(session: Session, user_id: int) -> dict[str, int]:
    """Delete one user **and** every row that belongs to them.

    Children are deleted first and explicitly, so the result is identical on
    PostgreSQL (``ON DELETE CASCADE``) and on SQLite, and so the caller can
    report how much was removed. Returns ``{table: rows_deleted}``.
    """
    user = session.get(User, user_id)
    if user is None:
        logger.warning("delete_user: user_id=%s not found", user_id)
        return {}
    removed = {
        "meals": int((session.execute(delete(Meal).where(Meal.user_id == user_id))).rowcount or 0),
        "predictions": int(
            (session.execute(delete(Prediction).where(Prediction.user_id == user_id))).rowcount or 0
        ),
        "recommendations": int(
            (session.execute(delete(Recommendation).where(Recommendation.user_id == user_id))).rowcount or 0
        ),
    }
    session.delete(user)
    session.flush()
    removed["users"] = 1
    logger.warning("Deleted user_id=%s and all related rows: %s", user_id, removed)
    return removed
