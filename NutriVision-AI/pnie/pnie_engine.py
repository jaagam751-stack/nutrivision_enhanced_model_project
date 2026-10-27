"""PNIE engine — the one place that turns saved meals into a forecast (Phase 6).

Pipeline
--------
``saved meals`` → ``get_daily_totals`` → :func:`pnie.feature_engineering.build_daily_series`
→ :func:`pnie.trend_analysis.analyse_trends` (always) → weekly rows
→ :class:`pnie.predictor.PNIEPredictor` (only when there is enough history)
→ :class:`PNIEOutcome`.

Modes
-----
The engine never invents a number. It reports exactly one of:

``no_data``              nothing has been saved yet;
``insufficient_history`` fewer than ``PNIE_MIN_HISTORY_DAYS`` days or
                         ``PNIE_MIN_LOGGED_DAYS`` logged days — descriptive
                         statistics only;
``descriptive_only``     enough history for statistics, fewer than
                         ``PNIE_MIN_TRAINING_ROWS`` weekly windows — still no
                         forecast, but the exact shortfall is reported;
``awaiting_recent_data`` the last look-back window is too sparsely logged, so a
                         forecast would describe days that were not recorded;
``model_unavailable``    the model could not be trained (e.g. XGBoost missing) —
                         the reason is surfaced, never hidden;
``model``                a real XGBoost forecast, with a chronological hold-out
                         error estimate and every guard that was applied.

Everything in this module is pure: it receives ``get_daily_totals`` rows and
returns a result object. Database reads/writes live in ``utils/data_access.py``
(the same split Phase 5 uses), which keeps the engine unit-testable offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from config import (
    PNIE_HORIZON_DAYS,
    PNIE_LOOKBACK_DAYS,
    PNIE_MAX_DEVIATION,
    PNIE_MAX_SERIES_DAYS,
    PNIE_MIN_HISTORY_DAYS,
    PNIE_MIN_LOGGED_DAYS,
    PNIE_MIN_TARGET_LOGGED_DAYS,
    PNIE_MIN_TRAINING_ROWS,
    PNIE_MODEL_NAME,
    PNIE_MODEL_VERSION,
    PNIE_ORIGIN_STRIDE_DAYS,
    PNIE_STALE_AFTER_DAYS,
    PNIE_STABLE_BAND_KCAL,
    PNIE_TEST_FRACTION,
    PNIE_XGB_PARAMS,
)
from pnie.feature_engineering import (
    FEATURE_COLUMNS,
    DailySeries,
    FeatureDataset,
    FeatureRow,
    build_daily_series,
    build_training_rows,
    latest_feature_row,
)
from pnie.predictor import PNIEPredictor, TrainingReport, clamp_prediction, importance_table
from pnie.trend_analysis import (
    TrendAnalysis,
    analyse_trends,
    direction_label,
    direction_of,
)

logger = logging.getLogger(__name__)

MODE_NO_DATA: str = "no_data"
MODE_INSUFFICIENT_HISTORY: str = "insufficient_history"
MODE_DESCRIPTIVE_ONLY: str = "descriptive_only"
MODE_AWAITING_RECENT_DATA: str = "awaiting_recent_data"
MODE_MODEL_UNAVAILABLE: str = "model_unavailable"
MODE_MODEL: str = "model"

MODE_LABELS: dict[str, str] = {
    MODE_NO_DATA: "No meals logged yet",
    MODE_INSUFFICIENT_HISTORY: "Not enough history yet",
    MODE_DESCRIPTIVE_ONLY: "Trends only — model needs more weeks",
    MODE_AWAITING_RECENT_DATA: "Waiting for recent meals",
    MODE_MODEL_UNAVAILABLE: "Model unavailable",
    MODE_MODEL: "Forecast ready",
}
MODE_TONES: dict[str, str] = {
    MODE_NO_DATA: "info",
    MODE_INSUFFICIENT_HISTORY: "info",
    MODE_DESCRIPTIVE_ONLY: "info",
    MODE_AWAITING_RECENT_DATA: "warn",
    MODE_MODEL_UNAVAILABLE: "warn",
    MODE_MODEL: "good",
}


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class PNIEPrediction:
    """One next-week forecast, with everything needed to judge it."""

    value: float
    low: Optional[float]
    high: Optional[float]
    recent_reference: Optional[float]
    trend_direction: str
    trend_change_kcal: Optional[float]
    forecast_start: date
    forecast_end: date
    mae: Optional[float]
    rmse: Optional[float]
    baseline_mae: Optional[float]
    training_samples: int
    test_samples: int
    model_name: str
    model_version: str
    generated_at: datetime
    feature_importances: dict[str, float] = field(default_factory=dict)
    guards: tuple[str, ...] = ()
    explanation: str = ""

    @property
    def direction_label(self) -> str:
        return direction_label(self.trend_direction)

    @property
    def beats_baseline(self) -> Optional[bool]:
        if self.mae is None or self.baseline_mae in (None, 0):
            return None
        return self.mae < self.baseline_mae

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 1),
            "low": _round(self.low),
            "high": _round(self.high),
            "recent_reference": _round(self.recent_reference),
            "trend_direction": self.trend_direction,
            "trend_change_kcal": _round(self.trend_change_kcal),
            "forecast_start": self.forecast_start.isoformat(),
            "forecast_end": self.forecast_end.isoformat(),
            "mae": _round(self.mae),
            "rmse": _round(self.rmse),
            "baseline_mae": _round(self.baseline_mae),
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat(),
            "top_features": [
                {"feature": name, "share": round(share, 4)}
                for name, share in importance_table(self.feature_importances, top=8)
            ],
            "guards": list(self.guards),
        }


@dataclass(frozen=True, eq=False)
class PNIEOutcome:
    """Everything the UI needs from one engine run."""

    mode: str
    message: str
    available: bool = False                    # was any output produced at all?
    prediction: Optional[PNIEPrediction] = None
    trends: Optional[TrendAnalysis] = None
    training: Optional[TrainingReport] = None
    dataset: Optional[FeatureDataset] = None
    series: Optional[DailySeries] = None
    requirements: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_prediction(self) -> bool:
        return self.prediction is not None

    @property
    def has_model(self) -> bool:
        return self.mode == MODE_MODEL

    @property
    def label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    @property
    def tone(self) -> str:
        return MODE_TONES.get(self.mode, "info")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (used for logs, tests and stored metadata)."""
        return {
            "mode": self.mode,
            "label": self.label,
            "available": self.available,
            "message": self.message,
            "warnings": list(self.warnings),
            "trends": self.trends.to_dict() if self.trends else None,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "training": (
                {
                    "trained": self.training.trained,
                    "rows": self.training.rows,
                    "train_rows": self.training.train_rows,
                    "test_rows": self.training.test_rows,
                    "mae": _round(self.training.mae),
                    "baseline_mae": _round(self.training.baseline_mae),
                    "holdout_from": self.training.holdout_from.isoformat()
                    if self.training.holdout_from
                    else None,
                }
                if self.training
                else None
            ),
            "requirements": self.requirements,
            "generated_at": self.generated_at.isoformat(),
        }

    def metadata(self) -> dict[str, Any]:
        """Compact payload written to ``predictions.feature_metadata``.

        Kept small on purpose: it must stay valid JSON (no NaN, no numpy types)
        for both SQLite and PostgreSQL.
        """
        payload: dict[str, Any] = {
            "mode": self.mode,
            "lookback_days": self.requirements.get("lookback_days", PNIE_LOOKBACK_DAYS),
            "horizon_days": self.requirements.get("horizon_days", PNIE_HORIZON_DAYS),
            "features": list(FEATURE_COLUMNS),
            "generated_at": self.generated_at.isoformat(),
        }
        if self.trends is not None:
            payload["history"] = {
                "span_days": self.trends.span_days,
                "logged_days": self.trends.logged_days,
                "coverage_pct": _round(self.trends.coverage_pct),
                "mean_calories": _round(self.trends.mean_calories),
            }
        if self.dataset is not None:
            payload["dataset"] = {
                "rows": len(self.dataset),
                "dropped_target_windows": self.dataset.dropped_target_windows,
                "dropped_incomplete_features": self.dataset.dropped_incomplete_features,
            }
        if self.training is not None and self.training.trained:
            payload["evaluation"] = {
                "chronological_split": True,
                "train_rows": self.training.train_rows,
                "test_rows": self.training.test_rows,
                "mae": _round(self.training.mae),
                "rmse": _round(self.training.rmse),
                "baseline_mae": _round(self.training.baseline_mae),
                "refit_on_all_rows": self.training.refit_on_all,
            }
        if self.prediction is not None:
            payload["top_features"] = [
                {"feature": name, "share": round(share, 4)}
                for name, share in importance_table(self.prediction.feature_importances, top=8)
            ]
            payload["guards"] = list(self.prediction.guards)
        return payload


def _round(value: Optional[float], decimals: int = 1) -> Optional[float]:
    """Round, mapping NaN/inf to ``None`` so the value stays JSON-safe."""
    if value is None:
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, decimals)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class PNIEEngine:
    """Run the full PNIE pipeline for one user's daily totals."""

    def __init__(
        self,
        lookback: int = PNIE_LOOKBACK_DAYS,
        horizon: int = PNIE_HORIZON_DAYS,
        stride: int = PNIE_ORIGIN_STRIDE_DAYS,
        min_history_days: int = PNIE_MIN_HISTORY_DAYS,
        min_logged_days: int = PNIE_MIN_LOGGED_DAYS,
        min_target_logged_days: int = PNIE_MIN_TARGET_LOGGED_DAYS,
        min_training_rows: int = PNIE_MIN_TRAINING_ROWS,
        test_fraction: float = PNIE_TEST_FRACTION,
        max_series_days: int = PNIE_MAX_SERIES_DAYS,
        max_deviation: float = PNIE_MAX_DEVIATION,
        stable_band: float = PNIE_STABLE_BAND_KCAL,
        stale_after_days: int = PNIE_STALE_AFTER_DAYS,
        params: Optional[Mapping[str, Any]] = None,
        predictor: Optional[PNIEPredictor] = None,
        today: Optional[date] = None,
    ) -> None:
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.min_history_days = int(min_history_days)
        self.min_logged_days = int(min_logged_days)
        self.min_target_logged_days = int(min_target_logged_days)
        self.min_training_rows = int(min_training_rows)
        self.test_fraction = float(test_fraction)
        self.max_series_days = int(max_series_days)
        self.max_deviation = float(max_deviation)
        self.stable_band = float(stable_band)
        self.stale_after_days = int(stale_after_days)
        self.params = dict(params or PNIE_XGB_PARAMS)
        self.predictor = predictor
        self.today = today

    # ------------------------------------------------------------------ #
    def run(self, daily_totals: Sequence[Mapping[str, Any]]) -> PNIEOutcome:
        """Analyse the user's history and produce an outcome (never raises)."""
        try:
            return self._run(daily_totals)
        except Exception as exc:  # noqa: BLE001 - a page must never crash on this
            logger.exception("PNIE run failed")
            return PNIEOutcome(
                mode=MODE_MODEL_UNAVAILABLE,
                message=(
                    "The prediction engine hit an unexpected error and stopped instead of "
                    f"returning a number: {exc.__class__.__name__}: {exc}"
                ),
                warnings=(f"{exc.__class__.__name__}: {exc}",),
            )

    # ------------------------------------------------------------------ #
    def _run(self, daily_totals: Sequence[Mapping[str, Any]]) -> PNIEOutcome:
        series = build_daily_series(daily_totals, max_days=self.max_series_days)
        requirements = self._requirements(series)

        if series.is_empty:
            return PNIEOutcome(
                mode=MODE_NO_DATA,
                message=(
                    "No meals have been saved yet, so there is nothing to analyse. Analyse a food "
                    "photo and press Save Meal — the engine needs saved history before it can say "
                    "anything about next week."
                ),
                available=False,
                requirements=requirements,
            )

        trends = analyse_trends(
            series,
            min_history_days=self.min_history_days,
            min_logged_days=self.min_logged_days,
            window_days=self.horizon,
            stable_band=self.stable_band,
        )
        warnings = self._warnings(series)

        # --- 1. enough history for statistics? --------------------------- #
        if not trends.has_enough_history:
            return PNIEOutcome(
                mode=MODE_INSUFFICIENT_HISTORY,
                message=trends.message,
                available=True,
                trends=trends,
                series=series,
                requirements=requirements,
                warnings=warnings,
            )

        # --- 2. enough weekly windows to train? -------------------------- #
        dataset = build_training_rows(
            series,
            lookback=self.lookback,
            horizon=self.horizon,
            stride=self.stride,
            min_target_logged_days=self.min_target_logged_days,
        )
        row = latest_feature_row(series, lookback=self.lookback)

        if len(dataset) < self.min_training_rows:
            missing_rows = self.min_training_rows - len(dataset)
            requirements = {**requirements, "training_rows": {
                "have": len(dataset), "need": self.min_training_rows}}
            return PNIEOutcome(
                mode=MODE_DESCRIPTIVE_ONLY,
                message=(
                    f"{dataset.describe()} The XGBoost model is trained from "
                    f"{self.min_training_rows} weekly window(s) — {missing_rows} more are needed "
                    f"(about {missing_rows * self.stride} more days of logging). Until then the "
                    "statistics above are the honest answer; no forecast is invented."
                ),
                available=True,
                trends=trends,
                dataset=dataset,
                series=series,
                requirements=requirements,
                warnings=warnings,
            )

        # --- 3. can the newest row be described? ------------------------- #
        if row is None or not row.is_usable:
            missing = ", ".join(row.missing) if row is not None else "the look-back window"
            requirements = {**requirements, "recent_missing_features": list(row.missing) if row else []}
            return PNIEOutcome(
                mode=MODE_AWAITING_RECENT_DATA,
                message=(
                    f"The last {self.lookback} days do not contain enough logged meals to describe "
                    f"your current pattern ({missing}). Log a few more days — a forecast built from "
                    "a window without data would describe days you never recorded."
                ),
                available=True,
                trends=trends,
                dataset=dataset,
                series=series,
                requirements=requirements,
                warnings=warnings,
            )

        # --- 4. train the single model ----------------------------------- #
        predictor = self.predictor or PNIEPredictor(
            params=self.params,
            test_fraction=self.test_fraction,
            min_rows=self.min_training_rows,
        )
        report = predictor.fit(dataset.features, dataset.targets)
        if not report.trained:
            return PNIEOutcome(
                mode=MODE_MODEL_UNAVAILABLE,
                message=report.reason,
                available=True,
                trends=trends,
                dataset=dataset,
                series=series,
                training=report,
                requirements=requirements,
                warnings=warnings + (report.reason,),
            )

        raw_value = predictor.predict(row.values, clamp=False)
        value, guards = self._apply_guards(raw_value, trends)

        mae = report.mae
        low = max(0.0, value - mae) if mae is not None else None
        high = (value + mae) if mae is not None else None

        reference = trends.latest_week_avg if trends.latest_week_avg is not None else trends.mean_calories
        change = (value - reference) if reference is not None else None
        forecast_start, forecast_end = row.target_window(self.horizon)

        prediction = PNIEPrediction(
            value=round(value, 1),
            low=_round(low),
            high=_round(high),
            recent_reference=_round(reference),
            trend_direction=direction_of(change, self.stable_band),
            trend_change_kcal=_round(change),
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            mae=_round(mae),
            rmse=_round(report.rmse),
            baseline_mae=_round(report.baseline_mae),
            training_samples=report.rows,
            test_samples=report.test_rows,
            model_name=PNIE_MODEL_NAME,
            model_version=PNIE_MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            feature_importances=dict(report.feature_importances),
            guards=tuple(guards),
            explanation=self._explain(row, report, reference, change),
        )

        if guards:
            warnings = warnings + tuple(guards)

        return PNIEOutcome(
            mode=MODE_MODEL,
            message=(
                f"Next week ({forecast_start:%d %b} – {forecast_end:%d %b}) is forecast at "
                f"{prediction.value:,.0f} kcal/day on average."
            ),
            available=True,
            prediction=prediction,
            trends=trends,
            dataset=dataset,
            series=series,
            training=report,
            requirements=requirements,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def _apply_guards(self, raw_value: float, trends: TrendAnalysis) -> tuple[float, list[str]]:
        """Apply the sanity guards and describe every clamp that was applied."""
        guards: list[str] = []
        value, clamped = clamp_prediction(raw_value)
        if clamped:
            guards.append(
                f"The raw model output ({raw_value:,.0f} kcal/day) was outside the allowed range "
                "and was clamped to a sane value."
            )

        reference = trends.latest_week_avg if trends.latest_week_avg is not None else trends.mean_calories
        if reference and reference > 0:
            limit = reference * self.max_deviation
            if value > reference + limit:
                value = reference + limit
                guards.append(
                    f"The forecast was more than {self.max_deviation:.0%} above your recent average "
                    "and was pulled back to that limit."
                )
            elif value < reference - limit:
                value = max(0.0, reference - limit)
                guards.append(
                    f"The forecast was more than {self.max_deviation:.0%} below your recent average "
                    "and was pulled back to that limit."
                )
        return value, guards

    # ------------------------------------------------------------------ #
    def _requirements(self, series: DailySeries) -> dict[str, Any]:
        """What the engine has and what it still needs — shown in the UI."""
        return {
            "lookback_days": self.lookback,
            "horizon_days": self.horizon,
            "stride_days": self.stride,
            "history_days": {"have": series.days, "need": self.min_history_days},
            "logged_days": {"have": series.logged_days, "need": self.min_logged_days},
            "training_rows": {"have": None, "need": self.min_training_rows},
            "min_target_logged_days": self.min_target_logged_days,
            "last_logged_on": series.last_logged_on.isoformat() if series.last_logged_on else None,
            "staleness_days": series.staleness_days(self.today),
            "series_truncated": series.truncated,
        }

    def _warnings(self, series: DailySeries) -> tuple[str, ...]:
        warnings: list[str] = []
        staleness = series.staleness_days(self.today)
        if staleness is not None and staleness > self.stale_after_days:
            warnings.append(
                f"The most recent saved meal is {staleness} day(s) old, so this forecast describes "
                "a pattern that may already have changed. Log new meals to refresh it."
            )
        if series.truncated:
            warnings.append(
                f"Only the most recent {self.max_series_days} days of history are used, so older "
                "meals are excluded from training."
            )
        return tuple(warnings)

    @staticmethod
    def _explain(
        row: FeatureRow,
        report: TrainingReport,
        reference: Optional[float],
        change: Optional[float],
    ) -> str:
        """Plain-language description of how the number was produced."""
        pieces = [
            f"One XGBoost regressor trained on {report.train_rows} weekly window(s) of your saved "
            f"meals and scored on the {report.test_rows} newest ones (chronological split, MAE "
            f"{report.mae:,.0f} kcal/day)." if report.mae is not None
            else f"One XGBoost regressor trained on {report.train_rows} weekly window(s) of your saved meals.",
            f"The model then read the {row.window_start:%d %b} – {row.window_end:%d %b} window "
            f"({row.logged_days} logged day(s)) and predicted the average daily calories for the "
            "following week.",
        ]
        if reference is not None and change is not None:
            pieces.append(
                f"That is {abs(change):,.0f} kcal {'above' if change >= 0 else 'below'} your recent "
                f"average of {reference:,.0f} kcal/day."
            )
        pieces.append("It is an estimate from your own history, not a guarantee or medical advice.")
        return " ".join(pieces)


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
def run_pnie(daily_totals: Sequence[Mapping[str, Any]], **kwargs: Any) -> PNIEOutcome:
    """One-shot helper: ``run_pnie(get_daily_totals(user_id))``."""
    return PNIEEngine(**kwargs).run(daily_totals)


def forecast_window(today: Optional[date] = None, horizon: int = PNIE_HORIZON_DAYS) -> tuple[date, date]:
    """Fallback window shown when no prediction exists yet."""
    start = (today or datetime.now(timezone.utc).date()) + timedelta(days=1)
    return start, start + timedelta(days=horizon - 1)
