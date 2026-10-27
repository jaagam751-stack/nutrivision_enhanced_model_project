"""Streamlit-facing data access helpers (Phases 5-8).

``utils/analytics.py`` stays pure (dataframes and figures only); this module is
the thin layer that talks to the database on behalf of the pages, and caches the
expensive or repeated lookups per session.

Writes happen only from explicit button presses, one function per action:

* :func:`store_meal` — the "Save Meal" button (Phase 5);
* :func:`store_prediction` — "Save this forecast" (Phase 6);
* :func:`request_recommendation` — the coach button (Phase 7);
* :func:`persist_profile`, :func:`delete_meal_history`,
  :func:`delete_profile_and_data` — the Profile page, the last two behind a typed
  confirmation (Phase 8).

Pages never call the repository directly, and every read that can fail returns an
empty/neutral value plus a message instead of raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import streamlit as st

from config import (
    COACH_DUPLICATE_GUARD_SECONDS,
    COACH_STALE_AFTER_DAYS,
    DUPLICATE_GUARD_SECONDS,
    GEMINI_MAX_MEALS_IN_CONTEXT,
    PNIE_DUPLICATE_GUARD_SECONDS,
)
from ai import CoachResult, CoachStatus, GeminiCoach, NutritionContext, build_context
from database import (
    DatabaseStatus,
    check_database,
    count_meals,
    delete_meals_for_user,
    delete_user,
    find_duplicate_meal,
    find_user_by_name,
    get_daily_totals,
    get_latest_prediction,
    get_latest_recommendation,
    list_meals,
    list_predictions,
    list_recent_meals,
    list_recommendations,
    save_meal,
    save_prediction,
    save_recommendation,
    save_user,
    update_user,
    user_data_summary,
)
from database.connection import DatabaseNotConfiguredError
from sqlalchemy.exc import SQLAlchemyError
from models import Meal, Prediction, Recommendation, User
from pnie import PNIEEngine, PNIEOutcome
from utils.analytics import MealSummary, summarise_meals

logger = logging.getLogger(__name__)

SESSION_DB_STATUS: str = "nv_db_available"


# --------------------------------------------------------------------------- #
# Connection / schema status
# --------------------------------------------------------------------------- #
def db_status(refresh: bool = False) -> DatabaseStatus:
    """Return the cached database status for this session.

    The check is done once per session (and again when ``refresh=True``) because
    it costs a round trip and the answer rarely changes.
    """
    if refresh or SESSION_DB_STATUS not in st.session_state:
        st.session_state[SESSION_DB_STATUS] = check_database()
    return st.session_state[SESSION_DB_STATUS]


def db_ready() -> bool:
    """``True`` when the database is reachable and the schema exists."""
    return bool(db_status().ok)


def forget_db_status() -> None:
    """Drop the cached status (used after a configuration change)."""
    st.session_state.pop(SESSION_DB_STATUS, None)


def require_stored_profile(profile) -> bool:
    """Render guidance when the profile has no database row yet.

    Meals reference ``users.user_id`` by foreign key, so a session-only profile
    cannot own meals. Returns ``True`` when it is safe to continue.
    """
    if profile is None:
        st.info("Create your profile on the main page first.")
        return False
    if getattr(profile, "user_id", None):
        return True

    st.warning(
        "Your profile is not stored in the database yet, so meals have no owner to be linked to. "
        "Open the **NutriVision AI** page and press **Save profile to database**, then come back."
    )
    return False


def require_database() -> bool:
    """Render guidance when the database is unavailable. Returns ``True`` if usable."""
    status = db_status()
    if status.ok:
        return True

    st.error(
        "The database is not available, so nothing can be read or written here.\n\n"
        f"**Backend:** {status.label} · **Target:** {status.location}"
        + (f"\n\n**Error:** {status.error}" if status.error else "")
    )
    st.info(
        "Start your Neon database (or the local SQLite fallback), then run "
        "`python database/init_db.py --check` and reload this page."
    )
    return False


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass
class HistoryBundle:
    """Everything the dashboard and history pages need for one date range."""

    meals: list[Any]
    summary: MealSummary
    daily: list[dict[str, Any]]
    start: Optional[date]
    end: Optional[date]


def history_for_range(
    user_id: int,
    start: Optional[date] = None,
    end: Optional[date] = None,
    limit: Optional[int] = None,
) -> HistoryBundle:
    """Fetch meals and their daily aggregates for one range."""
    meals = list_meals(user_id, start=start, end=end, limit=limit) or []
    daily = get_daily_totals(user_id, start=start, end=end) or []
    return HistoryBundle(meals=meals, summary=summarise_meals(meals), daily=daily, start=start, end=end)


def today_summary(user_id: int) -> tuple[MealSummary, list[Any]]:
    """Today's totals (UTC calendar day, matching how meals are aggregated).

    The timezone caveat is documented in the README: timestamps are stored in UTC,
    so "today" is the UTC day, not the user's local day.
    """
    today = datetime.now(timezone.utc).date()
    meals = list_meals(user_id, start=today, end=today) or []
    return summarise_meals(meals), meals


def recent_meals(user_id: int, limit: int = 6) -> list[Any]:
    """The most recent saved meals."""
    return list_recent_meals(user_id, limit=limit) or []


def total_meals(user_id: int) -> int:
    """How many meals the user has saved in total."""
    return int(count_meals(user_id) or 0)


def latest_recommendation(user_id: int):
    """Latest stored AI recommendation, or ``None`` (Phase 7 fills this table)."""
    try:
        return get_latest_recommendation(user_id)
    except Exception:  # noqa: BLE001 - a missing table must not break the dashboard
        logger.warning("Could not read recommendations", exc_info=False)
        return None


# --------------------------------------------------------------------------- #
# Writing (explicit user action only)
# --------------------------------------------------------------------------- #
def store_meal(
    profile,
    food_name: str,
    portion_grams: float,
    values: dict[str, Optional[float]],
    health_score: float,
    timestamp: Optional[datetime] = None,
) -> tuple[Optional[Meal], str]:
    """Save one meal after the user pressed **Save Meal**.

    Two independent guards stop accidental duplicates:

    1. the UI locks the button once a meal has been saved for the current
       analysis (session state), and
    2. this function asks the database whether an identical meal (same food,
       same portion, same calories) was written in the last
       :data:`config.DUPLICATE_GUARD_SECONDS` seconds — which also covers a double
       click racing a rerun.

    Returns ``(meal, message)``. ``meal`` is ``None`` when nothing was written.
    """
    user_id = getattr(profile, "user_id", None)
    if not user_id:
        return None, "The profile has no database id, so the meal cannot be stored."

    calories = float(values.get("calories") or 0.0)
    if calories <= 0:
        return None, "A meal with zero calories is not stored — please check the portion size."

    try:
        duplicate = find_duplicate_meal(
            user_id,
            food_name=food_name,
            portion_grams=float(portion_grams),
            calories=calories,
            within_seconds=DUPLICATE_GUARD_SECONDS,
        )
        if duplicate is not None:
            logger.info("Duplicate save ignored (existing meal_id=%s)", duplicate.meal_id)
            return duplicate, (
                f"This meal was already saved a moment ago as meal #{duplicate.meal_id} — "
                "the duplicate was ignored."
            )

        meal = Meal(
            user_id=int(user_id),
            timestamp=timestamp or datetime.now(timezone.utc),
            food_name=str(food_name)[:120],
            portion_grams=float(portion_grams),
            calories=calories,
            protein=float(values.get("protein") or 0.0),
            fat=float(values.get("fat") or 0.0),
            carbohydrates=float(values.get("carbs") or 0.0),
            fiber=float(values.get("fiber") or 0.0),
            health_score=float(health_score or 0.0),
        )
        saved = save_meal(meal)
        if saved is None:
            return None, "The database rejected the insert (schema problem). Nothing was saved."
        return saved, f"Meal saved as #{saved.meal_id}."
    except Exception as exc:  # noqa: BLE001 - surface the real reason to the user
        logger.exception("Saving the meal failed")
        return None, f"Could not save the meal: {exc.__class__.__name__}: {exc}"


def delete_meal_by_id(meal_id: int) -> tuple[bool, str]:
    """Delete one meal (used by the history page, with its own confirmation)."""
    from database import delete_meal

    try:
        removed = bool(delete_meal(int(meal_id)))
        return removed, ("Meal deleted." if removed else "That meal no longer exists.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Deleting meal %s failed", meal_id)
        return False, f"Could not delete the meal: {exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Range helpers
# --------------------------------------------------------------------------- #
def resolve_range(option: str, custom: Optional[tuple[date, date]] = None) -> tuple[Optional[date], Optional[date]]:
    """Turn a range label into ``(start, end)`` dates (``None`` = unbounded)."""
    today = datetime.now(timezone.utc).date()
    if option == "All time":
        return None, None
    if option == "Custom range":
        if custom and len(custom) == 2:
            start, end = custom
            return (start, end) if start <= end else (end, start)
        return today - timedelta(days=6), today
    days = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Last 90 days": 90,
    }.get(option, 14)
    return today - timedelta(days=days - 1), today


# --------------------------------------------------------------------------- #
# PNIE — the predictive engine (Phase 6)
# --------------------------------------------------------------------------- #
# The engine itself is pure (``pnie/``); these helpers are the only place that
# touches the database for it. Reads are cached by *content*: the cache key is
# the daily-totals payload, so editing, adding or deleting a meal always
# produces a fresh run, while a page rerun with unchanged data is free.
@st.cache_data(show_spinner=False, ttl=900)
def _cached_pnie_run(daily_totals: list[dict[str, Any]], today_iso: str) -> PNIEOutcome:
    """Cached engine run. Keyed by the data itself, so it can never go stale."""
    from datetime import date as _date

    return PNIEEngine(today=_date.fromisoformat(today_iso)).run(daily_totals)


def pnie_daily_totals(user_id: int) -> list[dict[str, Any]]:
    """Every day the user has saved meals on (the PNIE input)."""
    return get_daily_totals(int(user_id)) or []


def run_pnie_for_user(
    user_id: int,
    today: Optional[date] = None,
    daily_totals: Optional[list[dict[str, Any]]] = None,
) -> PNIEOutcome:
    """Run the PNIE for one user over their complete saved history.

    ``daily_totals`` lets a page reuse rows it already fetched instead of
    querying them twice.
    """
    reference = today or datetime.now(timezone.utc).date()
    try:
        totals = daily_totals if daily_totals is not None else pnie_daily_totals(int(user_id))
        return _cached_pnie_run(totals, reference.isoformat())
    except Exception as exc:  # noqa: BLE001 - the page shows the reason instead of crashing
        logger.exception("PNIE run failed for user %s", user_id)
        return PNIEOutcome(
            mode="model_unavailable",
            message=(
                "The prediction engine could not run: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            warnings=(str(exc),),
        )


def clear_pnie_cache() -> None:
    """Drop the cached engine runs (used after bulk data changes)."""
    _cached_pnie_run.clear()


def latest_stored_prediction(user_id: int) -> Optional[Prediction]:
    """Newest stored forecast, or ``None`` (missing table must not break a page)."""
    try:
        return get_latest_prediction(int(user_id))
    except Exception:  # noqa: BLE001
        logger.warning("Could not read predictions", exc_info=False)
        return None


def stored_predictions(user_id: int, limit: int = 8) -> list[Prediction]:
    """Previously stored forecasts, newest first."""
    try:
        return list(list_predictions(int(user_id), limit=limit) or [])
    except Exception:  # noqa: BLE001
        logger.warning("Could not list predictions", exc_info=False)
        return []


def _as_utc(stamp: Optional[datetime]) -> Optional[datetime]:
    """SQLite returns naive datetimes; treat them as UTC (same as Phase 5)."""
    if stamp is None:
        return None
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp


def store_prediction(user_id: int, outcome: PNIEOutcome) -> tuple[Optional[Prediction], str]:
    """Persist one forecast — called **only** from the explicit save button.

    Only a real model forecast is stored: descriptive statistics are never
    written as if they were predictions. A forecast identical (within 0.5 kcal)
    to the one saved in the last :data:`config.PNIE_DUPLICATE_GUARD_SECONDS` is
    reported instead of duplicated.
    """
    if outcome is None or outcome.prediction is None:
        return None, (
            "There is no model forecast to store yet — the engine is still in "
            f"'{outcome.mode if outcome else 'unknown'}' mode and statistics are never saved as "
            "predictions."
        )

    forecast = outcome.prediction
    try:
        latest = get_latest_prediction(int(user_id))
        if latest is not None:
            saved_at = _as_utc(latest.prediction_date)
            age = (datetime.now(timezone.utc) - saved_at).total_seconds() if saved_at else None
            if (
                age is not None
                and age <= PNIE_DUPLICATE_GUARD_SECONDS
                and latest.predicted_next_week_avg_calories is not None
                and abs(float(latest.predicted_next_week_avg_calories) - float(forecast.value)) < 0.5
            ):
                return latest, (
                    f"The same forecast was already saved a moment ago as prediction "
                    f"#{latest.prediction_id} — nothing new was written."
                )

        row = Prediction(
            user_id=int(user_id),
            prediction_date=datetime.now(timezone.utc),
            predicted_next_week_avg_calories=round(float(forecast.value), 1),
            trend_direction=forecast.trend_direction[:20],
            trend_change_kcal=(
                round(float(forecast.trend_change_kcal), 1)
                if forecast.trend_change_kcal is not None
                else None
            ),
            model_name=forecast.model_name[:60],
            model_version=forecast.model_version[:30],
            training_samples=int(forecast.training_samples),
            mae=round(float(forecast.mae), 2) if forecast.mae is not None else None,
            feature_metadata=outcome.metadata(),
        )
        saved = save_prediction(row)
        if saved is None:
            return None, "The database rejected the insert (schema problem). Nothing was saved."
        return saved, f"Forecast saved as prediction #{saved.prediction_id}."
    except Exception as exc:  # noqa: BLE001 - surface the real reason to the user
        logger.exception("Saving the prediction failed")
        return None, f"Could not save the forecast: {exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Gemini AI coach (Phase 7)
# --------------------------------------------------------------------------- #
# The coach costs API quota, so nothing here runs on page load: the page builds
# a snapshot for display, and a call happens only when the user presses a
# button. A second click within COACH_DUPLICATE_GUARD_SECONDS reuses the stored
# answer instead of spending another request.
@dataclass
class CoachOutcome:
    """What one coaching request produced: stored row, live result, or a reuse."""

    row: Optional[Recommendation] = None
    result: Optional[CoachResult] = None
    reused: bool = False
    message: str = ""

    @property
    def text(self) -> str:
        if self.result is not None and self.result.ok:
            return self.result.text
        if self.row is not None:
            return str(self.row.recommendation_text or "")
        return ""

    @property
    def ok(self) -> bool:
        return bool(self.text)


def coach_status() -> CoachStatus:
    """Is the coach configured in this environment? (cheap, no API call)"""
    try:
        return GeminiCoach().status
    except Exception as exc:  # noqa: BLE001 - a broken SDK must not break the page
        logger.exception("Could not read the coach status")
        return CoachStatus(configured=False, sdk_available=False, reason=str(exc))


def coach_meals(user_id: int, limit: int = 120) -> list[Any]:
    """Recent meals used for the snapshot (newest first)."""
    return recent_meals(user_id, limit=limit)


def coach_context(
    user_id: int,
    profile,
    outcome: Optional[Any] = None,
    meals: Optional[list[Any]] = None,
) -> NutritionContext:
    """Build the exact snapshot the model will receive.

    The PNIE outcome supplies the trend summary and the forecast; when it is not
    passed the engine is run here (the result is cached by its input data).
    """
    totals = pnie_daily_totals(int(user_id))
    recent = meals if meals is not None else coach_meals(int(user_id))
    run = outcome if outcome is not None else run_pnie_for_user(int(user_id), daily_totals=totals)
    return build_context(profile, totals, recent, run, max_meals=GEMINI_MAX_MEALS_IN_CONTEXT)


def stored_recommendations(user_id: int, limit: int = 10) -> list[Recommendation]:
    """Previously stored coaching messages, newest first."""
    try:
        return list(list_recommendations(int(user_id), limit=limit) or [])
    except Exception:  # noqa: BLE001
        logger.warning("Could not list recommendations", exc_info=False)
        return []


def coach_is_stale(row: Optional[Recommendation]) -> bool:
    """True when a stored recommendation is old enough to be worth refreshing."""
    if row is None or row.timestamp is None:
        return False
    stamp = _as_utc(row.timestamp)
    return (datetime.now(timezone.utc) - stamp).days >= COACH_STALE_AFTER_DAYS


def request_recommendation(
    user_id: int,
    context: NutritionContext,
    question: str = "",
    coach: Optional[GeminiCoach] = None,
) -> CoachOutcome:
    """Ask the coach once and store the answer — only on an explicit click.

    Nothing is written when the call fails: a failure is reported with its
    specific reason instead of a substitute message, and a request identical to
    one answered in the last :data:`config.COACH_DUPLICATE_GUARD_SECONDS` is
    reused so a double click cannot spend two API calls.
    """
    question = (question or "").strip()
    try:
        newest = get_latest_recommendation(int(user_id))
    except Exception:  # noqa: BLE001 - a missing table must not block the coach
        newest = None

    if newest is not None and str(newest.recommendation_text or "").strip():
        saved_at = _as_utc(newest.timestamp)
        same_question = (newest.prompt or "").strip() == question
        age = (datetime.now(timezone.utc) - saved_at).total_seconds() if saved_at else None
        if same_question and age is not None and age <= COACH_DUPLICATE_GUARD_SECONDS:
            return CoachOutcome(
                row=newest,
                reused=True,
                message=(
                    "This question was answered a moment ago, so the stored answer is shown instead "
                    "of spending another API call."
                ),
            )

    result = (coach or GeminiCoach()).generate(context, question=question)
    if not result.ok:
        return CoachOutcome(result=result, message=result.error_message)

    try:
        row = Recommendation(
            user_id=int(user_id),
            timestamp=datetime.now(timezone.utc),
            recommendation_text=result.text,
            source=(result.model or "gemini-2.5-flash")[:40],
            prompt=question or None,
        )
        saved = save_recommendation(row)
    except Exception as exc:  # noqa: BLE001 - the answer is still shown if only the write failed
        logger.exception("Saving the recommendation failed")
        return CoachOutcome(
            result=result,
            message=(
                "The coach answered, but the reply could not be stored "
                f"({exc.__class__.__name__}). It is shown below but will not appear in your history."
            ),
        )

    if saved is None:
        return CoachOutcome(result=result, message="The database rejected the insert — the answer is shown but not stored.")
    return CoachOutcome(
        row=saved,
        result=result,
        message=f"Recommendation saved as #{saved.recommendation_id} · {result.latency_ms} ms · "
                f"{result.total_tokens or '—'} tokens",
    )


# --------------------------------------------------------------------------- #
# Profile page (Phase 8): overview, persistence and explicit deletes
# --------------------------------------------------------------------------- #
@dataclass
class ProfileOverview:
    """Everything the Profile page shows about what is stored for this user."""

    user_id: Optional[int] = None
    stored: bool = False
    backend: str = ""
    location: str = ""
    meals: int = 0
    logged_days: int = 0
    predictions: int = 0
    recommendations: int = 0
    first_meal_at: Optional[str] = None
    last_meal_at: Optional[str] = None
    error: str = ""
    message: str = ""

    def as_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for the UI (empty rows are simply not shown)."""
        rows: list[tuple[str, str]] = [
            ("Saved profile", f"user_id {self.user_id}" if self.stored else "not stored yet"),
            ("Meals saved", f"{self.meals:,}"),
            ("Days with at least one meal", f"{self.logged_days:,}"),
            ("Stored forecasts", f"{self.predictions:,}"),
            ("Stored AI recommendations", f"{self.recommendations:,}"),
        ]
        if self.first_meal_at:
            rows.append(("First meal", self.first_meal_at))
        if self.last_meal_at:
            rows.append(("Latest meal", self.last_meal_at))
        if self.backend:
            rows.append(("Database backend", f"{self.backend} · {self.location}".strip(" ·")))
        return rows


def _stamp(value: Optional[datetime]) -> Optional[str]:
    """Format a database timestamp for display (UTC, minute precision)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%d %b %Y · %H:%M UTC")


def profile_overview(user_id: Optional[int]) -> ProfileOverview:
    """Summarise what is stored for one user, without ever raising.

    A database problem is reported in ``error`` so the page can show the real
    reason instead of pretending the numbers are zero.
    """
    status = db_status()
    overview = ProfileOverview(
        user_id=user_id,
        stored=bool(user_id),
        backend=status.label,
        location=status.location,
    )
    if not user_id:
        return overview
    if not status.ok:
        overview.stored = False
        overview.error = "The database is not available, so nothing could be read."
        return overview
    try:
        summary = user_data_summary(int(user_id)) or {}
    except (SQLAlchemyError, DatabaseNotConfiguredError) as exc:
        logger.error("profile_overview failed: %s", exc)
        overview.stored = False
        overview.error = f"{exc.__class__.__name__}: {exc}"
        return overview

    if not summary.get("exists"):
        overview.stored = False
        overview.error = "No row with this user_id exists any more."
        return overview

    overview.meals = int(summary.get("meals") or 0)
    overview.logged_days = int(summary.get("logged_days") or 0)
    overview.predictions = int(summary.get("predictions") or 0)
    overview.recommendations = int(summary.get("recommendations") or 0)
    overview.first_meal_at = _stamp(summary.get("first_meal_at"))
    overview.last_meal_at = _stamp(summary.get("last_meal_at"))
    return overview


def persist_profile(
    profile,
    previous_user_id: Optional[int] = None,
    prefer_name_match: bool = True,
) -> tuple[Optional[int], str]:
    """Create or update the stored row for a validated profile.

    Behaviour (kept identical to the Phase 1/2 onboarding flow):

    * ``previous_user_id`` set  → that row is updated in place;
    * otherwise the first user with the same name is updated, so saving twice
      never creates a duplicate account (Phase 2 has no authentication);
    * otherwise a new row is created.

    Returns ``(user_id, message)``. ``user_id`` is ``None`` when nothing was
    written, and ``message`` then carries the real error text — the caller keeps
    the profile in the session so a database problem can never lose it.
    """
    raw = {
        "name": profile.name,
        "age": int(profile.age),
        "gender": profile.gender,
        "height_cm": float(profile.height_cm),
        "weight_kg": float(profile.weight_kg),
        "activity_level": profile.activity_level,
        "diet_preference": profile.diet_preference,
        "fitness_goal": profile.fitness_goal,
    }
    try:
        if previous_user_id:
            row = update_user(int(previous_user_id), **raw)
            if row is not None:
                return int(row.user_id), f"Profile updated in place (user_id={row.user_id})."
        elif prefer_name_match:
            existing = find_user_by_name(profile.name)
            if existing is not None:
                row = update_user(existing.user_id, **raw)
                if row is not None:
                    return int(row.user_id), f"Existing profile updated (user_id={row.user_id})."

        created = save_user(User.from_profile(profile))
        if created is None:
            return None, "Nothing was written — the database reported a schema problem."
        return int(created.user_id), f"Profile saved as user_id={created.user_id}."
    except (SQLAlchemyError, DatabaseNotConfiguredError) as exc:
        logger.error("persist_profile failed: %s", exc)
        return None, f"{exc.__class__.__name__}: {exc}"


def delete_meal_history(user_id: int) -> tuple[int, str]:
    """Delete every meal of this user (the profile and forecasts stay).

    Returns ``(rows_deleted, message)``; the PNIE cache is cleared because the
    training data changed.
    """
    try:
        removed = delete_meals_for_user(int(user_id))
    except (SQLAlchemyError, DatabaseNotConfiguredError) as exc:
        logger.error("delete_meal_history failed: %s", exc)
        return 0, f"{exc.__class__.__name__}: {exc}"
    clear_pnie_cache()
    return removed, f"Deleted {removed:,} meal(s). Your profile is unchanged."


def delete_profile_and_data(user_id: int) -> tuple[dict[str, int], str]:
    """Delete the user row and every meal/forecast/recommendation that belongs to it."""
    try:
        removed = delete_user(int(user_id)) or {}
    except (SQLAlchemyError, DatabaseNotConfiguredError) as exc:
        logger.error("delete_profile_and_data failed: %s", exc)
        return {}, f"{exc.__class__.__name__}: {exc}"
    if not removed:
        return {}, "Nothing was deleted — no row with that user_id exists any more."
    clear_pnie_cache()
    parts = ", ".join(f"{value:,} {table}" for table, value in removed.items())
    return removed, f"Deleted {parts}."
