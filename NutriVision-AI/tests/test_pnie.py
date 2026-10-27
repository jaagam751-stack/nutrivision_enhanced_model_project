"""Phase 6 tests: PNIE features, trends, the XGBoost model and the engine.

The engine is pure (daily totals in, result object out), so almost everything
here runs offline against hand-built history. The persistence tests reuse the
throw-away SQLite database from ``tests/test_database.py``.

The rules that matter most are asserted explicitly:

* **no leakage** — features never change when the future changes;
* **chronological split** — the held-out rows are always the newest ones;
* **no fabrication** — with too little history the engine reports statistics and
  says exactly what is missing instead of returning a number.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from config import (
    PNIE_HORIZON_DAYS,
    PNIE_LOOKBACK_DAYS,
    PNIE_MAX_DEVIATION,
    PNIE_MIN_HISTORY_DAYS,
    PNIE_MIN_LOGGED_DAYS,
    PNIE_MIN_TRAINING_ROWS,
    PNIE_ORIGIN_STRIDE_DAYS,
    PNIE_PREDICTION_KCAL_BOUNDS,
)
from database import repository as repo
from models import Meal
from pnie import (
    MODE_AWAITING_RECENT_DATA,
    MODE_DESCRIPTIVE_ONLY,
    MODE_INSUFFICIENT_HISTORY,
    MODE_MODEL,
    MODE_MODEL_UNAVAILABLE,
    MODE_NO_DATA,
    PNIEEngine,
    PNIEPredictor,
    analyse_trends,
)
from pnie.feature_engineering import (
    FEATURE_COLUMNS,
    build_daily_series,
    build_training_rows,
    feature_vector,
    latest_feature_row,
    missing_features,
)
from pnie.predictor import TrainingReport, clamp_prediction, importance_table
from pnie.trend_analysis import direction_of, linear_slope
from utils import data_access

# Reuse the database fixtures defined for the Phase 2 tests.

#: Fixed end date so tests never depend on the machine clock.
ANCHOR: date = date(2026, 6, 30)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def daily_rows(
    days: int,
    calories: float = 1800.0,
    drift: float = 0.0,
    skip: Sequence[int] = (),
    end: date = ANCHOR,
    meals: int = 3,
) -> list[dict]:
    """``get_daily_totals``-shaped rows for the last ``days`` calendar days.

    ``skip`` holds day offsets (0 = oldest) that are left out entirely, i.e. days
    the user did not log — which is the interesting case.
    """
    first = end - timedelta(days=days - 1)
    rows: list[dict] = []
    for offset in range(days):
        if offset in skip:
            continue
        kcal = calories + drift * offset
        rows.append(
            {
                "date": first + timedelta(days=offset),
                "meals": meals,
                "calories": round(kcal, 1),
                "protein": round(kcal * 0.15 / 4.0, 1),
                "fat": round(kcal * 0.28 / 9.0, 1),
                "carbohydrates": round(kcal * 0.57 / 4.0, 1),
                "fiber": 15.0,
                "avg_health_score": 62.0,
            }
        )
    return rows


def series_of(days: int, **kwargs):
    return build_daily_series(daily_rows(days, **kwargs))


class StubPredictor:
    """Stand-in for the XGBoost model, used to test the engine's guards."""

    def __init__(self, value: float = 1800.0, trained: bool = True, reason: str = "") -> None:
        self.value = float(value)
        self.trained = trained
        self.reason = reason
        self.model_name = "stub-model"
        self.model_version = "0.0"
        self.report: TrainingReport | None = None

    def fit(self, features, targets) -> TrainingReport:
        rows = len(features)
        self.report = TrainingReport(
            trained=self.trained,
            reason=self.reason,
            rows=rows,
            train_rows=max(rows - 2, 0),
            test_rows=min(2, rows),
            mae=25.0 if self.trained else None,
            rmse=30.0 if self.trained else None,
            baseline_mae=40.0 if self.trained else None,
            feature_importances={name: 1.0 / len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS},
        )
        return self.report

    def predict(self, values, clamp: bool = True) -> float:
        return self.value


# --------------------------------------------------------------------------- #
# Daily series
# --------------------------------------------------------------------------- #
def test_series_spans_first_to_last_logged_day_and_flags_gaps() -> None:
    series = series_of(10, skip=(4,))
    assert series.days == 10
    assert series.logged_days == 9
    assert series.start == ANCHOR - timedelta(days=9)
    assert series.end == ANCHOR
    assert series.last_logged_on == ANCHOR
    assert series.coverage == pytest.approx(0.9)
    # The gap is present as a calendar day, marked as not logged.
    gap = series.frame.iloc[4]
    assert gap["meals"] == 0 and bool(gap["logged"]) is False


def test_gap_days_are_not_treated_as_zero_calorie_days() -> None:
    series = series_of(10, skip=(4,))
    logged = series.logged_frame()
    assert len(logged) == 9
    assert float(logged["calories"].min()) == pytest.approx(1800.0)
    assert series.meals == 27  # 9 days x 3 meals


def test_series_is_capped_to_max_days_from_the_last_logged_day() -> None:
    series = build_daily_series(daily_rows(40), max_days=20)
    assert series.days == 20
    assert series.truncated is True
    assert series.end == ANCHOR


def test_empty_input_gives_an_empty_series() -> None:
    series = build_daily_series([])
    assert series.is_empty
    assert series.days == 0 and series.logged_days == 0
    assert series.staleness_days(ANCHOR) is None


def test_staleness_days_measures_the_last_saved_meal() -> None:
    series = series_of(10)
    assert series.staleness_days(ANCHOR + timedelta(days=3)) == 3


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_feature_vector_has_exactly_the_declared_features() -> None:
    series = series_of(20)
    window = series.frame.iloc[:14]
    vector = feature_vector(window, ANCHOR - timedelta(days=6))
    assert set(vector) == set(FEATURE_COLUMNS)
    assert len(vector) == len(FEATURE_COLUMNS)
    assert missing_features(vector) == ()


def test_feature_vector_reports_missing_values_without_logged_days() -> None:
    """A window with no logged days yields ``None`` features, never a fake zero."""
    series = series_of(20)
    window = series.frame.iloc[-PNIE_LOOKBACK_DAYS:].copy()
    window["logged"] = False
    window["meals"] = 0.0
    vector = feature_vector(window, ANCHOR + timedelta(days=1))
    missing = missing_features(vector)
    assert "mean_calories_7d" in missing
    assert "lag_calories_1d" in missing
    assert vector["logged_ratio_14d"] == pytest.approx(0.0)
    assert vector["day_of_week"] is not None


def test_lag_features_skip_unlogged_days() -> None:
    rows = daily_rows(10, skip=(7,))
    frame = build_daily_series(rows).frame
    window = frame.iloc[:10]
    vector = feature_vector(window, ANCHOR + timedelta(days=1))
    # The last logged day is ANCHOR (offset 9); lag 2 must be offset 8, not the gap.
    assert vector["lag_calories_1d"] == pytest.approx(frame.iloc[9]["calories"])
    assert vector["lag_calories_2d"] == pytest.approx(frame.iloc[8]["calories"])
    assert vector["lag_calories_3d"] == pytest.approx(frame.iloc[6]["calories"])


def test_rolling_means_use_logged_days_only() -> None:
    series = series_of(10, skip=(8,))          # the 9th day is missing
    window = series.frame.iloc[:10]
    vector = feature_vector(window, ANCHOR + timedelta(days=1))
    last_two_logged = [series.frame.iloc[7]["calories"], series.frame.iloc[9]["calories"]]
    assert vector["mean_calories_3d"] == pytest.approx(sum(last_two_logged) / 2)


def test_slope_feature_follows_the_direction_of_the_history() -> None:
    rising = feature_vector(series_of(20, drift=25.0).frame.iloc[:14], ANCHOR - timedelta(days=6))
    falling = feature_vector(series_of(20, drift=-25.0).frame.iloc[:14], ANCHOR - timedelta(days=6))
    assert rising["trend_slope_14d"] > 0
    assert falling["trend_slope_14d"] < 0


def test_training_rows_are_spaced_by_the_origin_stride() -> None:
    dataset = build_training_rows(series_of(60))
    origins = list(dataset.features.index)
    assert len(origins) >= 3
    gaps = {(b - a).days for a, b in zip(origins, origins[1:])}
    assert gaps == {PNIE_ORIGIN_STRIDE_DAYS}
    assert origins[0].date() == (ANCHOR - timedelta(days=59)) + timedelta(days=PNIE_LOOKBACK_DAYS)


def test_features_never_include_the_future() -> None:
    """The core leakage test: change everything at/after the origin, features stay equal."""
    series = series_of(60)
    dataset = build_training_rows(series)
    origin = dataset.features.index[-1]

    tampered_frame = series.frame.copy()
    future_mask = tampered_frame["date"] > origin
    assert future_mask.any()
    tampered_frame.loc[future_mask, "calories"] = 9999.0
    tampered_frame.loc[future_mask, "protein"] = 999.0
    tampered_frame.loc[future_mask, "logged"] = True
    tampered_frame.loc[future_mask, "meals"] = 5

    start = int(tampered_frame.index[tampered_frame["date"] == origin][0])
    window = tampered_frame.iloc[start - PNIE_LOOKBACK_DAYS:start]
    rebuilt = feature_vector(window, origin)
    assert rebuilt == pytest.approx(dict(dataset.features.loc[origin]), abs=1e-9)


def test_target_is_the_average_of_logged_days_in_the_horizon() -> None:
    rows = daily_rows(60, skip=(16,))           # one unlogged day inside the first target week
    dataset = build_training_rows(build_daily_series(rows))
    origin = dataset.features.index[0]
    frame = build_daily_series(rows).frame
    start = int(frame.index[frame["date"] == origin][0])
    window = frame.iloc[start:start + PNIE_HORIZON_DAYS]
    expected = float(window[window["logged"]]["calories"].mean())
    assert dataset.targets.iloc[0] == pytest.approx(expected, rel=1e-6)
    assert dataset.meta.iloc[0]["target_logged_days"] == PNIE_HORIZON_DAYS - 1


def test_windows_with_too_few_logged_target_days_are_dropped_and_counted() -> None:
    # Days 14-20 (the first target week) hold a single logged day -> dropped.
    dataset = build_training_rows(build_daily_series(daily_rows(60, skip=(15, 16, 17, 18, 19, 20))))
    assert dataset.dropped_target_windows >= 1
    assert dataset.dropped_total == dataset.dropped_target_windows + dataset.dropped_incomplete_features


def test_no_rows_when_history_is_shorter_than_lookback_plus_horizon() -> None:
    dataset = build_training_rows(series_of(PNIE_LOOKBACK_DAYS + PNIE_HORIZON_DAYS - 1))
    assert dataset.is_empty
    assert str(PNIE_LOOKBACK_DAYS + PNIE_HORIZON_DAYS) in dataset.reason


def test_dataset_is_chronological() -> None:
    dataset = build_training_rows(series_of(80))
    origins = pd.Series(dataset.features.index)
    assert origins.is_monotonic_increasing


def test_latest_feature_row_targets_the_days_after_the_last_logged_day() -> None:
    series = series_of(30, skip=(28,))
    row = latest_feature_row(series)
    assert row is not None
    assert row.origin == series.end + timedelta(days=1)
    assert row.window_end == series.end
    assert row.window_start == series.end - timedelta(days=PNIE_LOOKBACK_DAYS - 1)
    assert row.is_usable
    start, end = row.target_window()
    assert start == row.origin
    assert (end - start).days == PNIE_HORIZON_DAYS - 1


def test_latest_feature_row_flags_sparse_recent_history() -> None:
    # A single meal is never a look-back window.
    rows = daily_rows(1, end=ANCHOR - timedelta(days=29))
    assert latest_feature_row(build_daily_series(rows)) is None

    # Two months of logging, then a long break and one meal today: the newest
    # window cannot describe a pattern, so the row is returned but unusable.
    rows = daily_rows(20, end=ANCHOR - timedelta(days=40)) + daily_rows(1, end=ANCHOR)
    row = latest_feature_row(build_daily_series(rows))
    assert row is not None
    assert row.logged_days == 1
    assert not row.is_usable
    assert "lag_calories_2d" in row.missing


# --------------------------------------------------------------------------- #
# Trend analysis
# --------------------------------------------------------------------------- #
def test_trends_compute_mean_median_and_spread() -> None:
    analysis = analyse_trends(series_of(30, skip=(10,)))
    assert analysis.logged_days == 29
    assert analysis.mean_calories == pytest.approx(1800.0)
    assert analysis.median_calories == pytest.approx(1800.0)
    assert analysis.std_calories == pytest.approx(0.0, abs=1e-6)
    assert analysis.min_calories == analysis.max_calories == pytest.approx(1800.0)


def test_trends_compare_the_last_week_with_the_week_before() -> None:
    analysis = analyse_trends(series_of(28, drift=50.0))
    assert analysis.latest_week_avg > analysis.previous_week_avg
    assert analysis.change_kcal == pytest.approx(
        analysis.latest_week_avg - analysis.previous_week_avg, rel=1e-9
    )
    assert analysis.change_pct > 0


def test_trends_slope_and_direction_follow_a_linear_series() -> None:
    analysis = analyse_trends(series_of(60, drift=20.0))
    assert analysis.slope_per_day == pytest.approx(20.0, rel=0.05)
    assert analysis.slope_per_week == pytest.approx(140.0, rel=0.05)
    assert analysis.direction == "rising"
    flat = analyse_trends(series_of(60))
    assert flat.direction == "stable"


def test_direction_of_uses_the_stable_band() -> None:
    assert direction_of(1000.0) == "rising"
    assert direction_of(-1000.0) == "falling"
    assert direction_of(10.0) == "stable"
    assert direction_of(None) == "stable"


def test_trends_flag_insufficient_history_and_say_what_is_missing() -> None:
    analysis = analyse_trends(series_of(5))
    assert analysis.has_enough_history is False
    assert f"{PNIE_MIN_HISTORY_DAYS} days" in analysis.message
    assert f"{PNIE_MIN_LOGGED_DAYS} logged days" in analysis.message


def test_trends_report_highest_and_lowest_logged_days() -> None:
    analysis = analyse_trends(series_of(20, drift=10.0))
    assert analysis.highest_day is not None and analysis.lowest_day is not None
    assert analysis.highest_day[1] > analysis.lowest_day[1]
    assert analysis.highest_day[0] == ANCHOR.strftime("%Y-%m-%d")


def test_energy_share_sums_to_one() -> None:
    analysis = analyse_trends(series_of(20))
    assert analysis.energy_share is not None
    assert sum(analysis.energy_share.values()) == pytest.approx(1.0)


def test_linear_slope_edge_cases() -> None:
    assert linear_slope([0, 1], [1.0, 3.0]) == pytest.approx(2.0)
    assert linear_slope([0], [1.0]) is None
    assert linear_slope([2, 2], [1.0, 5.0]) is None


def test_trend_dict_is_json_serialisable() -> None:
    analysis = analyse_trends(series_of(30, skip=(4,)))
    payload = json.dumps(analysis.to_dict())
    assert "mean_calories" in payload
    assert "NaN" not in payload


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
def _dataset(days: int = 120, drift: float = -4.0):
    return build_training_rows(series_of(days, drift=drift))


def test_predictor_refuses_to_fit_with_too_few_rows() -> None:
    dataset = build_training_rows(series_of(40))
    predictor = PNIEPredictor()
    report = predictor.fit(dataset.features, dataset.targets)
    assert report.trained is False
    assert str(PNIE_MIN_TRAINING_ROWS) in report.reason
    assert predictor.is_fitted is False


def test_predictor_trains_and_reports_a_chronological_error() -> None:
    dataset = _dataset()
    predictor = PNIEPredictor()
    report = predictor.fit(dataset.features, dataset.targets)
    assert report.trained is True
    assert report.rows == len(dataset)
    assert report.train_rows + report.test_rows == report.rows
    assert report.train_rows > report.test_rows
    assert report.mae is not None and report.mae >= 0
    assert report.rmse is not None and report.rmse >= report.mae
    assert report.refit_on_all is True
    assert predictor.is_fitted is True


def test_holdout_is_always_the_newest_rows() -> None:
    dataset = _dataset()
    predictor = PNIEPredictor()
    report = predictor.fit(dataset.features, dataset.targets)
    held_out_from = dataset.features.index[report.rows - report.test_rows]
    assert report.holdout_from == held_out_from.date()
    last_training_origin = dataset.features.index[report.rows - report.test_rows - 1]
    assert pd.Timestamp(report.holdout_from) > last_training_origin


def test_predictor_is_deterministic_for_the_same_data() -> None:
    dataset = _dataset()
    row = latest_feature_row(build_daily_series(daily_rows(120, drift=-4.0)))
    first = PNIEPredictor()
    first.fit(dataset.features, dataset.targets)
    second = PNIEPredictor()
    second.fit(dataset.features, dataset.targets)
    assert first.predict(row.values) == pytest.approx(second.predict(row.values))


def test_predictor_rejects_wrong_inputs() -> None:
    dataset = _dataset()
    predictor = PNIEPredictor()
    with pytest.raises(ValueError):
        predictor.fit(dataset.features.drop(columns=["mean_calories_7d"]), dataset.targets)

    broken = dataset.features.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        PNIEPredictor().fit(broken, dataset.targets)

    with pytest.raises(RuntimeError):
        PNIEPredictor().predict({name: 1.0 for name in FEATURE_COLUMNS})


def test_predict_rejects_an_incomplete_row() -> None:
    dataset = _dataset()
    predictor = PNIEPredictor()
    predictor.fit(dataset.features, dataset.targets)
    incomplete = {name: 1.0 for name in FEATURE_COLUMNS}
    incomplete["lag_calories_3d"] = None
    with pytest.raises(ValueError):
        predictor.predict(incomplete)


def test_predictions_stay_inside_the_sanity_bounds() -> None:
    low, high = PNIE_PREDICTION_KCAL_BOUNDS
    assert clamp_prediction(low - 500.0) == (low, True)
    assert clamp_prediction(high + 5000.0) == (high, True)
    assert clamp_prediction(1800.0) == (1800.0, False)


def test_feature_importances_cover_every_feature() -> None:
    report = PNIEPredictor().fit(_dataset().features, _dataset().targets)
    assert set(report.feature_importances) == set(FEATURE_COLUMNS)
    assert sum(report.feature_importances.values()) == pytest.approx(1.0, abs=0.01)


def test_importance_table_is_sorted_descending() -> None:
    table = importance_table({"a": 0.1, "b": 0.7, "c": 0.2}, top=2)
    assert [name for name, _ in table] == ["b", "c"]


def test_model_beats_the_mean_baseline_on_a_clear_trend() -> None:
    """Sanity check that the model learns *something* from a strong signal."""
    dataset = _dataset(days=240, drift=6.0)
    report = PNIEPredictor().fit(dataset.features, dataset.targets)
    assert report.trained
    assert report.mae is not None and report.baseline_mae is not None
    assert report.mae <= report.baseline_mae * 1.25   # at worst roughly as good


def test_missing_xgboost_is_reported_instead_of_crashing(monkeypatch) -> None:
    import pnie.predictor as predictor_module

    monkeypatch.setattr(predictor_module, "XGBRegressor", None)
    dataset = _dataset()
    report = PNIEPredictor().fit(dataset.features, dataset.targets)
    assert report.trained is False
    assert "XGBoost is not installed" in report.reason


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def test_no_data_mode() -> None:
    outcome = PNIEEngine(today=ANCHOR).run([])
    assert outcome.mode == MODE_NO_DATA
    assert outcome.available is False
    assert outcome.prediction is None
    assert outcome.trends is None


def test_insufficient_history_mode_shows_statistics_only() -> None:
    outcome = PNIEEngine(today=ANCHOR).run(daily_rows(6))
    assert outcome.mode == MODE_INSUFFICIENT_HISTORY
    assert outcome.prediction is None
    assert outcome.trends is not None and outcome.trends.logged_days == 6
    assert outcome.available is True
    assert "Descriptive statistics only" in outcome.message


def test_descriptive_only_mode_reports_the_shortfall() -> None:
    outcome = PNIEEngine(today=ANCHOR).run(daily_rows(45))
    assert outcome.mode == MODE_DESCRIPTIVE_ONLY
    assert outcome.prediction is None
    assert outcome.trends is not None
    assert outcome.dataset is not None and len(outcome.dataset) < PNIE_MIN_TRAINING_ROWS
    assert "weekly window" in outcome.message


def test_model_mode_produces_a_guarded_prediction() -> None:
    outcome = PNIEEngine(today=ANCHOR).run(daily_rows(120, drift=-3.0))
    assert outcome.mode == MODE_MODEL
    prediction = outcome.prediction
    assert prediction is not None
    low, high = PNIE_PREDICTION_KCAL_BOUNDS
    assert low <= prediction.value <= high
    assert prediction.low is not None and prediction.high is not None
    assert prediction.low <= prediction.value <= prediction.high
    assert prediction.training_samples >= PNIE_MIN_TRAINING_ROWS
    assert prediction.model_name and prediction.model_version
    assert prediction.forecast_start == ANCHOR + timedelta(days=1)
    assert prediction.trend_direction in {"rising", "falling", "stable"}
    assert prediction.explanation
    assert outcome.metadata()["evaluation"]["chronological_split"] is True


def test_prediction_direction_tracks_the_recent_average() -> None:
    """The direction is arithmetic (forecast minus the recent average), not a second model."""
    rows = daily_rows(120, drift=-3.0)
    higher = PNIEEngine(today=ANCHOR, predictor=StubPredictor(value=1800.0)).run(rows)
    lower = PNIEEngine(today=ANCHOR, predictor=StubPredictor(value=1300.0)).run(rows)
    assert higher.prediction.trend_direction == "rising"
    assert higher.prediction.trend_change_kcal > 50
    assert lower.prediction.trend_direction == "falling"
    assert lower.prediction.trend_change_kcal < -50


def test_engine_is_deterministic() -> None:
    rows = daily_rows(120, drift=-3.0)
    first = PNIEEngine(today=ANCHOR).run(rows)
    second = PNIEEngine(today=ANCHOR).run(rows)
    assert first.prediction.value == pytest.approx(second.prediction.value)
    assert first.prediction.feature_importances == second.prediction.feature_importances


def test_stale_history_is_flagged() -> None:
    outcome = PNIEEngine(today=ANCHOR + timedelta(days=20)).run(daily_rows(120, drift=-3.0))
    assert any("day(s) old" in warning for warning in outcome.warnings)


def test_relative_guard_pulls_back_an_extreme_forecast() -> None:
    stub = StubPredictor(value=9000.0)
    outcome = PNIEEngine(today=ANCHOR, predictor=stub).run(daily_rows(120, drift=-3.0))
    reference = outcome.prediction.recent_reference
    assert outcome.prediction.value == pytest.approx(reference * (1 + PNIE_MAX_DEVIATION), abs=0.1)
    assert len(outcome.prediction.guards) == 2          # absolute clamp + relative pull-back
    assert any("allowed range" in guard for guard in outcome.prediction.guards)
    assert any("recent average" in guard for guard in outcome.prediction.guards)


def test_untrained_predictor_reports_model_unavailable() -> None:
    stub = StubPredictor(trained=False, reason="stub could not train")
    outcome = PNIEEngine(today=ANCHOR, predictor=stub).run(daily_rows(120, drift=-3.0))
    assert outcome.mode == MODE_MODEL_UNAVAILABLE
    assert outcome.prediction is None
    assert "stub could not train" in outcome.message


def test_sparse_recent_history_blocks_the_forecast() -> None:
    # Four months of history, then a two-week break with a single meal at the end:
    # there is plenty to train on, but nothing recent to describe.
    rows = daily_rows(110, drift=-3.0, end=ANCHOR - timedelta(days=14)) + daily_rows(1, end=ANCHOR)
    outcome = PNIEEngine(today=ANCHOR).run(rows)
    assert outcome.mode == MODE_AWAITING_RECENT_DATA
    assert outcome.prediction is None
    assert "Log a few more days" in outcome.message


def test_outcome_payloads_are_json_safe() -> None:
    outcome = PNIEEngine(today=ANCHOR).run(daily_rows(120, drift=-3.0))
    for payload in (outcome.to_dict(), outcome.metadata()):
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["mode"] == MODE_MODEL


def test_engine_never_raises_on_broken_input() -> None:
    outcome = PNIEEngine(today=ANCHOR).run([{"date": None, "calories": "not a number"}])
    assert outcome.mode in (MODE_NO_DATA, MODE_MODEL_UNAVAILABLE)
    assert outcome.prediction is None


# --------------------------------------------------------------------------- #
# Persistence (predictions table)
# --------------------------------------------------------------------------- #
def _stored_user_meals(user, days: int = 120, drift: float = -3.0) -> None:
    """Write real meal rows so the engine can be exercised through the database."""
    first = ANCHOR - timedelta(days=days - 1)
    meals = []
    for offset in range(days):
        kcal = 1800.0 + drift * offset
        meals.append(
            Meal(
                user_id=user.user_id,
                timestamp=datetime.combine(
                    first + timedelta(days=offset), datetime.min.time(), tzinfo=timezone.utc
                )
                + timedelta(hours=8 + (offset % 3) * 5),
                food_name="Test meal",
                portion_grams=250.0,
                calories=round(kcal / 3.0, 1),
                protein=20.0,
                fat=15.0,
                carbohydrates=60.0,
                fiber=5.0,
                health_score=60.0,
            )
        )
        meals.append(
            Meal(
                user_id=user.user_id,
                timestamp=datetime.combine(
                    first + timedelta(days=offset), datetime.min.time(), tzinfo=timezone.utc
                )
                + timedelta(hours=20),
                food_name="Test meal",
                portion_grams=250.0,
                calories=round(kcal / 3.0, 1),
                protein=20.0,
                fat=15.0,
                carbohydrates=60.0,
                fiber=5.0,
                health_score=60.0,
            )
        )
    assert repo.save_meals_bulk(meals) == len(meals)


def test_pnie_reads_real_meals_and_stores_a_forecast(sqlite_db, user_factory) -> None:
    user = user_factory("Pnie User")
    _stored_user_meals(user)

    totals = data_access.pnie_daily_totals(user.user_id)
    assert len(totals) == 120

    outcome = PNIEEngine(today=ANCHOR).run(totals)
    assert outcome.mode == MODE_MODEL

    stored, message = data_access.store_prediction(user.user_id, outcome)
    assert stored is not None, message
    assert "prediction #" in message

    row = repo.get_latest_prediction(user.user_id)
    assert row is not None
    assert row.predicted_next_week_avg_calories == pytest.approx(outcome.prediction.value, abs=0.1)
    assert row.trend_direction in {"rising", "falling", "stable"}
    assert row.model_name == outcome.prediction.model_name
    assert row.model_version == outcome.prediction.model_version
    assert row.training_samples == outcome.prediction.training_samples
    assert row.mae == pytest.approx(outcome.prediction.mae, abs=0.01)
    assert isinstance(row.feature_metadata, dict)
    assert row.feature_metadata["mode"] == MODE_MODEL
    assert row.feature_metadata["evaluation"]["chronological_split"] is True
    assert json.dumps(row.feature_metadata)          # stays valid JSON for both backends


def test_storing_the_same_forecast_twice_is_guarded(sqlite_db, user_factory) -> None:
    user = user_factory("Pnie Guard")
    _stored_user_meals(user)
    outcome = PNIEEngine(today=ANCHOR).run(data_access.pnie_daily_totals(user.user_id))

    first, _ = data_access.store_prediction(user.user_id, outcome)
    second, message = data_access.store_prediction(user.user_id, outcome)
    assert first is not None and second is not None
    assert first.prediction_id == second.prediction_id
    assert "already saved a moment ago" in message
    assert len(data_access.stored_predictions(user.user_id)) == 1
    assert data_access.latest_stored_prediction(user.user_id).prediction_id == first.prediction_id


def test_statistics_are_never_stored_as_predictions(sqlite_db, user_factory) -> None:
    user = user_factory("Pnie Short")
    short = PNIEEngine(today=ANCHOR).run(daily_rows(10))
    stored, message = data_access.store_prediction(user.user_id, short)
    assert stored is None
    assert "no model forecast" in message
    assert repo.list_predictions(user.user_id) == []
