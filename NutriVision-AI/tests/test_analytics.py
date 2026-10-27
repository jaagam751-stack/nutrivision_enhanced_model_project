"""Phase 5 tests: aggregations, chart builders and the duplicate-save guard.

The analytics module is pure (no database, no Streamlit), so everything here runs
offline. The duplicate guard is tested against the throw-away SQLite database
created by the ``sqlite_db`` fixture in ``tests/test_database.py``, which is
imported here to reuse the same fixtures.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go

from config import TREND_WINDOW_DAYS
from database import repository as repo
from models import Meal
from utils import analytics
from utils.analytics import MealSummary

# Reuse the database fixtures defined for the Phase 2 tests.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class Row:
    """Minimal meal-like object for the pure aggregation helpers."""

    def __init__(self, **kwargs) -> None:
        self.timestamp = kwargs.get("timestamp", datetime.now(timezone.utc))
        self.food_name = kwargs.get("food_name", "Test food")
        self.portion_grams = kwargs.get("portion_grams", 100.0)
        self.calories = kwargs.get("calories", 200.0)
        self.protein = kwargs.get("protein", 8.0)
        self.fat = kwargs.get("fat", 6.0)
        self.carbohydrates = kwargs.get("carbohydrates", 28.0)
        self.fiber = kwargs.get("fiber", 3.0)
        self.health_score = kwargs.get("health_score", 65.0)
        self.meal_id = kwargs.get("meal_id", 1)


def daily_rows(days: int, calories: float = 1800.0, meals_per_day: int = 3) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    return [
        {
            "date": today - timedelta(days=offset),
            "meals": meals_per_day,
            "calories": calories,
            "protein": 70.0,
            "fat": 60.0,
            "carbohydrates": 220.0,
            "fiber": 18.0,
            "avg_health_score": 64.0,
        }
        for offset in range(days)
    ]


# --------------------------------------------------------------------------- #
# summarise_meals
# --------------------------------------------------------------------------- #
def test_summarise_empty_is_zeroed() -> None:
    summary = analytics.summarise_meals([])
    assert summary.is_empty
    assert summary.meals == 0 and summary.calories == 0.0
    assert summary.avg_health_score == 0.0
    assert summary.as_rows()[0] == ("Meals logged", "0")


def test_summarise_totals_and_averages() -> None:
    summary = analytics.summarise_meals(
        [Row(calories=300, protein=10, fiber=4, health_score=70), Row(calories=500, protein=20, fiber=6, health_score=50)]
    )
    assert summary.meals == 2
    assert summary.calories == 800
    assert summary.protein == 30
    assert summary.fiber == 10
    assert summary.avg_health_score == 60.0
    assert summary.avg_calories_per_meal == 400.0


def test_summarise_tolerates_missing_values() -> None:
    class Sparse:
        calories = None
        protein = 5.0
        fat = None
        carbohydrates = 10.0
        fiber = None
        health_score = 0.0

    summary = analytics.summarise_meals([Sparse(), Sparse()])
    assert summary.meals == 2
    assert summary.calories == 0.0
    assert summary.protein == 10.0
    assert summary.carbs == 20.0


# --------------------------------------------------------------------------- #
# Dataframes
# --------------------------------------------------------------------------- #
def test_meals_frame_columns_and_ordering() -> None:
    older = Row(food_name="Dal", timestamp=datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc))
    newer = Row(food_name="Chapati", timestamp=datetime(2026, 10, 5, 20, 0, tzinfo=timezone.utc))
    frame = analytics.meals_frame([older, newer])

    assert list(frame.columns) == analytics.MEAL_TABLE_COLUMNS
    assert frame.iloc[0]["Food"] == "Chapati"          # newest first
    assert frame.iloc[1]["Food"] == "Dal"
    assert "05 Oct 2026" in frame.iloc[0]["When"]


def test_meals_frame_empty_has_columns_only() -> None:
    frame = analytics.meals_frame([])
    assert frame.empty
    assert list(frame.columns) == analytics.MEAL_TABLE_COLUMNS


def test_daily_frame_types_and_sorting() -> None:
    rows = list(reversed(daily_rows(3)))
    frame = analytics.daily_frame(rows)
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame["date"].is_monotonic_increasing
    assert frame["calories"].tolist() == [1800.0, 1800.0, 1800.0]


def test_daily_frame_handles_missing_columns_values() -> None:
    frame = analytics.daily_frame([{"date": date.today(), "meals": 1, "calories": None}])
    assert frame.iloc[0]["calories"] == 0.0
    assert frame.iloc[0]["protein"] == 0.0


def test_fill_missing_days_inserts_zero_rows() -> None:
    today = datetime.now(timezone.utc).date()
    frame = analytics.daily_frame(
        [{"date": today, "meals": 2, "calories": 1500, "protein": 60, "fat": 50,
          "carbohydrates": 180, "fiber": 16, "avg_health_score": 60}]
    )
    filled = analytics.fill_missing_days(frame, today - timedelta(days=4), today)
    assert len(filled) == 5
    assert (filled["meals"] == 0).sum() == 4
    assert filled.iloc[-1]["calories"] == 1500


def test_fill_missing_days_on_empty_input_returns_calendar() -> None:
    today = datetime.now(timezone.utc).date()
    filled = analytics.fill_missing_days(analytics.daily_frame([]), today - timedelta(days=6), today)
    assert len(filled) == 7
    assert float(filled["calories"].sum()) == 0.0


# --------------------------------------------------------------------------- #
# Weekly aggregation
# --------------------------------------------------------------------------- #
def test_weekly_frame_averages_over_logged_days_only() -> None:
    today = datetime.now(timezone.utc).date()
    # Two logged days inside one week, then two empty days.
    rows = []
    for offset, calories in ((0, 2000.0), (1, 1000.0), (2, 0.0), (3, 0.0)):
        rows.append(
            {
                "date": today - timedelta(days=offset),
                "meals": 3 if calories else 0,
                "calories": calories,
                "protein": 50.0 if calories else 0.0,
                "fat": 40.0 if calories else 0.0,
                "carbohydrates": 200.0 if calories else 0.0,
                "fiber": 15.0 if calories else 0.0,
                "avg_health_score": 60.0 if calories else 0.0,
            }
        )
    weekly = analytics.weekly_frame(analytics.daily_frame(rows))
    assert not weekly.empty
    assert weekly["avg_calories"].sum() > 0
    # 2000 + 1000 over two logged days, not over the week length.
    assert weekly["avg_calories"].max() <= 2000.0
    assert weekly["avg_calories"].min() >= 1000.0


def test_weekly_frame_empty() -> None:
    assert analytics.weekly_frame(analytics.daily_frame([])).empty


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_all_figures_return_plotly_figures() -> None:
    today = datetime.now(timezone.utc).date()
    daily = analytics.fill_missing_days(analytics.daily_frame(daily_rows(TREND_WINDOW_DAYS)), today - timedelta(days=13), today)
    weekly = analytics.weekly_frame(daily)
    summary = MealSummary(meals=3, calories=1800, protein=60, fat=50, carbs=200, fiber=15, avg_health_score=62)

    figures = [
        analytics.calorie_trend_figure(daily, target=2000),
        analytics.macro_donut_figure(summary),
        analytics.macro_stack_figure(daily),
        analytics.health_score_trend_figure(daily),
        analytics.weekly_average_figure(weekly, target=2000),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)


def test_empty_periods_return_annotated_placeholders() -> None:
    """Charts must explain the empty state instead of rendering nothing."""
    empty_daily = analytics.daily_frame([])
    empty_weekly = analytics.weekly_frame(empty_daily)
    empty_summary = MealSummary()

    for figure in (
        analytics.calorie_trend_figure(empty_daily),
        analytics.macro_donut_figure(empty_summary),
        analytics.macro_stack_figure(empty_daily),
        analytics.health_score_trend_figure(empty_daily),
        analytics.weekly_average_figure(empty_weekly),
    ):
        assert isinstance(figure, go.Figure)
        assert figure.layout.annotations, "an empty chart must carry an explanatory annotation"
        message = figure.layout.annotations[0].text
        assert isinstance(message, str) and len(message) > 15, "the placeholder must explain itself"


def test_calorie_trend_draws_one_bar_per_day_and_a_target_line() -> None:
    today = datetime.now(timezone.utc).date()
    daily = analytics.fill_missing_days(analytics.daily_frame(daily_rows(7)), today - timedelta(days=6), today)
    figure = analytics.calorie_trend_figure(daily, target=2100)
    assert len(figure.data[0].x) == 7
    assert figure.layout.shapes, "the target line should be drawn"


def test_health_score_chart_is_fixed_to_the_0_100_scale() -> None:
    today = datetime.now(timezone.utc).date()
    daily = analytics.fill_missing_days(analytics.daily_frame(daily_rows(5)), today - timedelta(days=4), today)
    figure = analytics.health_score_trend_figure(daily)
    assert list(figure.layout.yaxis.range) == [0, 100]


def test_weekly_bars_are_coloured_by_distance_from_target() -> None:
    weekly = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2026-09-28", "2026-10-05", "2026-10-12"]),
            "label": ["28 Sep", "05 Oct", "12 Oct"],
            "avg_calories": [2000.0, 1700.0, 900.0],       # on target, mild, far below
            "avg_protein": [70.0, 60.0, 30.0],
            "avg_fiber": [18.0, 15.0, 8.0],
            "days": [7, 7, 7],
        }
    )
    figure = analytics.weekly_average_figure(weekly, target=2000)
    colours = figure.data[0].marker.color
    assert colours[0] != colours[2]                        # on-target vs far-off differ
    assert len(set(colours)) >= 2


# --------------------------------------------------------------------------- #
# Duplicate-save guard (database)
# --------------------------------------------------------------------------- #
def _meal(user_id: int, **overrides) -> Meal:
    values = {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc),
        "food_name": "Samosa (fried)",
        "portion_grams": 60.0,
        "calories": 157.2,
        "protein": 2.7,
        "fat": 8.7,
        "carbohydrates": 18.0,
        "fiber": 1.4,
        "health_score": 34.2,
    }
    values.update(overrides)
    return Meal(**values)


def test_identical_meal_saved_twice_is_detected(user_factory) -> None:  # noqa: F811
    user = user_factory()
    first = repo.save_meal(_meal(user.user_id))
    duplicate = repo.find_duplicate_meal(
        user.user_id, food_name="Samosa (fried)", portion_grams=60.0, calories=157.2
    )
    assert duplicate is not None
    assert duplicate.meal_id == first.meal_id
    assert repo.count_meals(user.user_id) == 1


def test_different_portion_or_food_is_not_a_duplicate(user_factory) -> None:  # noqa: F811
    user = user_factory()
    repo.save_meal(_meal(user.user_id))
    assert repo.find_duplicate_meal(user.user_id, "Samosa (fried)", 120.0, 157.2) is None
    assert repo.find_duplicate_meal(user.user_id, "Dal toor (cooked)", 60.0, 157.2) is None
    assert repo.find_duplicate_meal(user.user_id, "Samosa (fried)", 60.0, 900.0) is None


def test_old_meal_is_not_treated_as_a_duplicate(user_factory) -> None:  # noqa: F811
    user = user_factory()
    repo.save_meal(_meal(user.user_id, timestamp=datetime.now(timezone.utc) - timedelta(minutes=30)))
    assert repo.find_duplicate_meal(user.user_id, "Samosa (fried)", 60.0, 157.2, within_seconds=120) is None


def test_duplicate_guard_is_scoped_to_the_user(user_factory) -> None:  # noqa: F811
    first, second = user_factory(), user_factory()
    repo.save_meal(_meal(first.user_id))
    assert repo.find_duplicate_meal(second.user_id, "Samosa (fried)", 60.0, 157.2) is None


def test_portion_tolerance_accepts_rounding(user_factory) -> None:  # noqa: F811
    user = user_factory()
    repo.save_meal(_meal(user.user_id, portion_grams=60.0))
    # A 0.3 g difference (float rounding from the slider) still counts as the same meal.
    assert repo.find_duplicate_meal(user.user_id, "Samosa (fried)", 60.3, 157.2) is not None
    # ...but 12 g more is a genuinely different portion.
    assert repo.find_duplicate_meal(user.user_id, "Samosa (fried)", 72.0, 157.2) is None


def test_meal_written_through_repository_appears_in_analytics(user_factory) -> None:  # noqa: F811
    """End-to-end: a saved row must flow into the dashboard aggregations."""
    user = user_factory()
    repo.save_meal(_meal(user.user_id, calories=300.0, protein=10.0, health_score=70.0))
    repo.save_meal(_meal(user.user_id, food_name="Dal toor (cooked)", calories=180.0, protein=11.0, health_score=89.0))

    meals = repo.list_meals(user.user_id)
    summary = analytics.summarise_meals(meals)
    assert summary.meals == 2
    assert summary.calories == 480.0
    assert summary.avg_health_score == 79.5

    daily = analytics.daily_frame(repo.get_daily_totals(user.user_id))
    assert len(daily) == 1
    assert daily.iloc[0]["calories"] == 480.0

    frame = analytics.meals_frame(meals)
    assert set(frame["Food"]) == {"Samosa (fried)", "Dal toor (cooked)"}
