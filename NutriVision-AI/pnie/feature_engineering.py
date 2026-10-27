"""PNIE feature engineering (Phase 6).

The Predictive Nutrition Intelligence Engine learns from **one row per week** of
history. This module builds those rows, and it is the only place where the
model's inputs are defined — training and live prediction call the same
:func:`feature_vector`, so the two can never drift apart.

The rules that keep the model honest
------------------------------------
* **Calendar-day grid.** Meals are bucketed per day by
  ``repository.get_daily_totals``; here the days between the first and last
  logged meal are re-indexed onto a continuous calendar grid. Days without meals
  are marked ``logged = False`` and are *never* counted as "0 kcal days": they
  are missing data, the same convention the Phase 5 charts use.
* **No leakage.** A row's features are computed from the ``lookback`` days
  **strictly before** its origin date; its target is the average daily calories
  over the ``horizon`` days **starting at** the origin. Nothing that the target
  window contains can reach the features (unit-tested by perturbing the future).
* **Overlap is explicit.** Rows are spaced ``stride`` days apart (one week by
  default) and a window is discarded when its target week has fewer than
  ``min_target_logged_days`` logged days, so a "next week average" always
  describes days the user actually recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config import (
    PNIE_HORIZON_DAYS,
    PNIE_LOOKBACK_DAYS,
    PNIE_MAX_SERIES_DAYS,
    PNIE_MIN_TARGET_LOGGED_DAYS,
    PNIE_ORIGIN_STRIDE_DAYS,
)
from pnie.trend_analysis import linear_slope

logger = logging.getLogger(__name__)

#: Nutrition columns carried per day (the keys ``get_daily_totals`` returns).
NUTRITION_COLUMNS: tuple[str, ...] = (
    "meals",
    "calories",
    "protein",
    "fat",
    "carbohydrates",
    "fiber",
    "avg_health_score",
)

#: The exact model input vector, in order. Every name is computed from history
#: only; the list is shared by training and prediction so the column order can
#: never silently change between the two.
FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_calories_3d",
    "mean_calories_7d",
    "mean_calories_14d",
    "std_calories_7d",
    "trend_slope_14d",
    "momentum_7d_vs_14d",
    "lag_calories_1d",
    "lag_calories_2d",
    "lag_calories_3d",
    "mean_protein_7d",
    "mean_fat_7d",
    "mean_carbohydrates_7d",
    "mean_fiber_7d",
    "mean_health_score_7d",
    "energy_share_fat_7d",
    "energy_share_carbohydrate_7d",
    "meals_per_logged_day_7d",
    "logged_ratio_7d",
    "logged_ratio_14d",
    "day_of_week",
)

#: Human-readable meanings, shown on the Predictions page next to the importances.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "mean_calories_3d": "Average kcal on logged days in the last 3 days before the prediction",
    "mean_calories_7d": "Average kcal on logged days in the last 7 days",
    "mean_calories_14d": "Average kcal on logged days in the last 14 days",
    "std_calories_7d": "Day-to-day spread of kcal in the last 7 days",
    "trend_slope_14d": "Direction of the last 14 days (kcal per day)",
    "momentum_7d_vs_14d": "Last week's average minus the two-week average",
    "lag_calories_1d": "Kcal on the most recent logged day",
    "lag_calories_2d": "Kcal on the second most recent logged day",
    "lag_calories_3d": "Kcal on the third most recent logged day",
    "mean_protein_7d": "Average protein per logged day, last 7 days (g)",
    "mean_fat_7d": "Average fat per logged day, last 7 days (g)",
    "mean_carbohydrates_7d": "Average carbohydrate per logged day, last 7 days (g)",
    "mean_fiber_7d": "Average fibre per logged day, last 7 days (g)",
    "mean_health_score_7d": "Average meal health score, last 7 days",
    "energy_share_fat_7d": "Share of macro energy from fat, last 7 days",
    "energy_share_carbohydrate_7d": "Share of macro energy from carbohydrate, last 7 days",
    "meals_per_logged_day_7d": "Average number of meals logged per recorded day",
    "logged_ratio_7d": "Share of the last 7 calendar days that have meals",
    "logged_ratio_14d": "Share of the last 14 calendar days that have meals",
    "day_of_week": "Weekday the forecast window starts on (0 = Monday)",
}


# --------------------------------------------------------------------------- #
# The daily series
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class DailySeries:
    """One row per calendar day, gaps included and flagged."""

    frame: pd.DataFrame
    start: Optional[date] = None
    end: Optional[date] = None
    last_logged_on: Optional[date] = None
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return self.frame.empty

    @property
    def days(self) -> int:
        """Number of calendar days the series spans."""
        return int(len(self.frame))

    @property
    def logged_days(self) -> int:
        """Days carrying at least one saved meal."""
        return int(self.frame["logged"].sum()) if not self.frame.empty else 0

    @property
    def meals(self) -> int:
        """Total meals in the series."""
        return int(self.frame["meals"].sum()) if not self.frame.empty else 0

    @property
    def coverage(self) -> float:
        """Fraction of the covered days that carry meals (0-1)."""
        return (self.logged_days / self.days) if self.days else 0.0

    def staleness_days(self, today: Optional[date] = None) -> Optional[int]:
        """How many days ago the last meal was logged (``None`` if never)."""
        if self.last_logged_on is None:
            return None
        reference = today or datetime.now(timezone.utc).date()
        return (reference - self.last_logged_on).days

    def logged_frame(self) -> pd.DataFrame:
        """Only the days with meals — that is what statistics are computed on."""
        return self.frame[self.frame["logged"]].copy() if not self.frame.empty else self.frame


def _empty_series() -> DailySeries:
    return DailySeries(
        frame=pd.DataFrame(
            {"date": pd.Series(dtype="datetime64[ns]"), **{name: pd.Series(dtype="float64") for name in NUTRITION_COLUMNS}, "logged": pd.Series(dtype="bool")}
        )
    )


def build_daily_series(
    daily_totals: Sequence[Mapping[str, Any]],
    max_days: int = PNIE_MAX_SERIES_DAYS,
    end: Optional[date] = None,
) -> DailySeries:
    """Turn ``get_daily_totals`` rows into a continuous calendar-day series.

    ``max_days`` keeps the most recent ``max_days`` calendar days (counted back
    from the last logged day, so a long break in logging does not silently drop
    history). ``truncated`` records that older days were cut.
    """
    if not daily_totals:
        return _empty_series()

    frame = pd.DataFrame.from_records(list(daily_totals))
    if "date" not in frame.columns:
        return _empty_series()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if frame.empty:
        return _empty_series()

    # Missing columns are tolerated (same defensiveness as the Phase 5 charts).
    for column in NUTRITION_COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)

    first_logged = frame["date"].min().date()
    last_logged = frame["date"].max().date()
    series_end = end or last_logged
    if series_end < last_logged:      # never cut off recorded meals
        series_end = last_logged
    series_start = max(first_logged, series_end - timedelta(days=max(1, int(max_days)) - 1))
    truncated = series_start > first_logged

    calendar = pd.DataFrame({"date": pd.date_range(start=series_start, end=series_end, freq="D")})
    merged = calendar.merge(frame, on="date", how="left")
    for column in NUTRITION_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["logged"] = merged["meals"] > 0

    return DailySeries(
        frame=merged.reset_index(drop=True),
        start=series_start,
        end=series_end,
        last_logged_on=last_logged,
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def _mean_over_logged(window: pd.DataFrame, column: str) -> Optional[float]:
    """Mean of ``column`` over the *logged* days of the window (``None`` if none)."""
    logged = window[window["logged"]]
    if logged.empty:
        return None
    value = pd.to_numeric(logged[column], errors="coerce").dropna()
    return float(value.mean()) if not value.empty else None


def _std_over_logged(window: pd.DataFrame, column: str) -> Optional[float]:
    """Spread of ``column`` over logged days.

    A single logged day has zero spread (not "unknown"), so it returns ``0.0``;
    ``None`` means the window holds no logged days at all.
    """
    logged = window[window["logged"]]
    values = pd.to_numeric(logged[column], errors="coerce").dropna()
    if values.empty:
        return None
    if len(values) == 1:
        return 0.0
    return float(values.std(ddof=1))


def _lag_over_logged(window: pd.DataFrame, column: str, lag: int) -> Optional[float]:
    """Value of ``column`` on the ``lag``-th most recent *logged* day.

    Gaps are skipped: with two meals logged in a 14-day window, lag 2 is the
    older of those two meals, not an empty day.
    """
    logged = window[window["logged"]]
    values = pd.to_numeric(logged[column], errors="coerce").dropna()
    if len(values) < lag or lag < 1:
        return None
    return float(values.iloc[-lag])


def _slope_over_logged(window: pd.DataFrame, column: str) -> Optional[float]:
    """Least-squares slope (units per day) of ``column`` over logged days."""
    logged = window[window["logged"]]
    if len(logged) < 2:
        return None
    values = pd.to_numeric(logged[column], errors="coerce")
    if values.isna().any():
        return None
    positions = (logged["date"] - logged["date"].iloc[0]).dt.days.tolist()
    return linear_slope(positions, values.tolist())


def _energy_share(window: pd.DataFrame, macro: str, kcal_per_gram: float) -> Optional[float]:
    """Share of macro energy contributed by one macronutrient (0-1)."""
    protein = _mean_over_logged(window, "protein")
    fat = _mean_over_logged(window, "fat")
    carbs = _mean_over_logged(window, "carbohydrates")
    if protein is None or fat is None or carbs is None:
        return None
    total = protein * 4.0 + fat * 9.0 + carbs * 4.0
    if total <= 0:
        return None
    grams = {"protein": protein, "fat": fat, "carbohydrates": carbs}[macro]
    return float(grams * kcal_per_gram / total)


def feature_vector(window: pd.DataFrame, origin_date: date | datetime | pd.Timestamp) -> dict[str, Optional[float]]:
    """Build one model input row from the ``window`` of days before an origin.

    ``window`` must contain **only** days strictly before ``origin_date`` — this
    function is the single definition of the model's inputs, used for both
    training rows and the live forecast. Values are ``None`` when the window
    does not hold enough logged days; the caller decides whether to drop the row
    (training) or to refuse to predict (live).
    """
    origin = pd.Timestamp(origin_date)
    mean_3 = _mean_over_logged(window.tail(3), "calories")
    mean_7 = _mean_over_logged(window.tail(7), "calories")
    mean_14 = _mean_over_logged(window.tail(14), "calories")
    window_7 = window.tail(7)

    momentum = (
        mean_7 - mean_14 if (mean_7 is not None and mean_14 is not None) else None
    )

    return {
        "mean_calories_3d": mean_3,
        "mean_calories_7d": mean_7,
        "mean_calories_14d": mean_14,
        "std_calories_7d": _std_over_logged(window_7, "calories"),
        "trend_slope_14d": _slope_over_logged(window.tail(14), "calories"),
        "momentum_7d_vs_14d": momentum,
        "lag_calories_1d": _lag_over_logged(window, "calories", 1),
        "lag_calories_2d": _lag_over_logged(window, "calories", 2),
        "lag_calories_3d": _lag_over_logged(window, "calories", 3),
        "mean_protein_7d": _mean_over_logged(window_7, "protein"),
        "mean_fat_7d": _mean_over_logged(window_7, "fat"),
        "mean_carbohydrates_7d": _mean_over_logged(window_7, "carbohydrates"),
        "mean_fiber_7d": _mean_over_logged(window_7, "fiber"),
        "mean_health_score_7d": _mean_over_logged(window_7, "avg_health_score"),
        "energy_share_fat_7d": _energy_share(window_7, "fat", 9.0),
        "energy_share_carbohydrate_7d": _energy_share(window_7, "carbohydrates", 4.0),
        "meals_per_logged_day_7d": (
            float(window_7[window_7["logged"]]["meals"].mean())
            if not window_7[window_7["logged"]].empty
            else None
        ),
        "logged_ratio_7d": float(window_7["logged"].mean()) if not window_7.empty else None,
        "logged_ratio_14d": float(window["logged"].mean()) if not window.empty else None,
        "day_of_week": float(origin.dayofweek),
    }


def missing_features(vector: Mapping[str, Optional[float]]) -> tuple[str, ...]:
    """Names of the features that could not be computed (``None`` or NaN)."""
    missing = []
    for name, value in vector.items():
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            missing.append(name)
    return tuple(missing)


# --------------------------------------------------------------------------- #
# Training rows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class FeatureDataset:
    """Supervised rows: one row per weekly origin, chronologically ordered."""

    features: pd.DataFrame
    targets: pd.Series
    meta: pd.DataFrame
    dropped_target_windows: int = 0
    dropped_incomplete_features: int = 0
    reason: str = ""

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_COLUMNS)

    def __len__(self) -> int:
        return int(len(self.features))

    @property
    def is_empty(self) -> bool:
        return self.features.empty

    @property
    def dropped_total(self) -> int:
        return self.dropped_target_windows + self.dropped_incomplete_features

    def describe(self) -> str:
        """One-line explanation of what was kept and what was thrown away."""
        if self.is_empty:
            return self.reason or "No usable weekly windows yet."
        return (
            f"{len(self)} weekly window(s) usable; "
            f"{self.dropped_target_windows} dropped for too few logged days in the target week, "
            f"{self.dropped_incomplete_features} dropped for insufficient look-back history."
        )


@dataclass(frozen=True, eq=False)
class FeatureRow:
    """The live (latest) feature row — what the model will be asked about."""

    values: dict[str, Optional[float]]
    origin: date
    window_start: date
    window_end: date
    logged_days: int
    missing: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return not self.missing

    def target_window(self, horizon: int = PNIE_HORIZON_DAYS) -> tuple[date, date]:
        """Forecast window implied by this row: ``horizon`` days from the origin."""
        return self.origin, self.origin + timedelta(days=horizon - 1)


def build_training_rows(
    series: DailySeries,
    lookback: int = PNIE_LOOKBACK_DAYS,
    horizon: int = PNIE_HORIZON_DAYS,
    stride: int = PNIE_ORIGIN_STRIDE_DAYS,
    min_target_logged_days: int = PNIE_MIN_TARGET_LOGGED_DAYS,
) -> FeatureDataset:
    """Build weekly supervised rows from a daily series.

    For every origin at index ``i`` (steps of ``stride`` days):

    * features come from days ``[i - lookback, i)``  — the past,
    * the target is the average kcal over **logged** days in ``[i, i + horizon)``.

    Windows whose target week has fewer than ``min_target_logged_days`` logged
    days are dropped: a forecast target built from one stray meal would teach the
    model noise. Rows missing features are dropped too, and both counts are kept
    so the UI can explain the numbers instead of hiding them.
    """
    empty = FeatureDataset(
        features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
        targets=pd.Series(dtype="float64"),
        meta=pd.DataFrame(columns=["origin", "target_start", "target_end", "target_logged_days", "target_calories"]),
        reason="Not enough history yet.",
    )
    if series.is_empty or series.days < lookback + horizon:
        return replace(
            empty,
            reason=(
                f"Need at least {lookback + horizon} calendar days of history "
                f"({lookback} look-back + {horizon} forecast); found {series.days}."
            ),
        )

    frame = series.frame.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    dropped_target = 0
    dropped_features = 0

    for start in range(lookback, len(frame) - horizon + 1, max(1, stride)):
        window = frame.iloc[start - lookback:start]
        target_window = frame.iloc[start:start + horizon]
        logged_target_days = int(target_window["logged"].sum())
        if logged_target_days < min_target_logged_days:
            dropped_target += 1
            continue

        origin = frame.at[start, "date"]
        vector = feature_vector(window, origin)
        if missing_features(vector):
            dropped_features += 1
            continue

        target_values = pd.to_numeric(
            target_window[target_window["logged"]]["calories"], errors="coerce"
        ).dropna()
        if target_values.empty:
            dropped_target += 1
            continue

        rows.append(dict(vector))
        metas.append(
            {
                "origin": origin,
                "target_start": target_window.at[target_window.index[0], "date"],
                "target_end": target_window.at[target_window.index[-1], "date"],
                "target_logged_days": logged_target_days,
                "target_calories": round(float(target_values.mean()), 1),
            }
        )

    if not rows:
        return replace(
            empty,
            dropped_target_windows=dropped_target,
            dropped_incomplete_features=dropped_features,
            reason=(
                f"No usable weekly window yet: {dropped_target} window(s) skipped for too few "
                f"logged days in the target week, {dropped_features} for insufficient look-back "
                "history."
            ),
        )

    meta_frame = pd.DataFrame.from_records(metas)
    features = pd.DataFrame.from_records(rows, columns=list(FEATURE_COLUMNS))
    features["origin"] = meta_frame["origin"].values
    features = features.set_index("origin")
    targets = pd.Series(
        meta_frame["target_calories"].astype(float).values,
        index=meta_frame["origin"].values,
        name="target_calories",
    )
    logger.info(
        "PNIE dataset built | rows=%s dropped_target=%s dropped_features=%s",
        len(features), dropped_target, dropped_features,
    )
    return FeatureDataset(
        features=features,
        targets=targets,
        meta=meta_frame,
        dropped_target_windows=dropped_target,
        dropped_incomplete_features=dropped_features,
    )


def latest_feature_row(series: DailySeries, lookback: int = PNIE_LOOKBACK_DAYS) -> Optional[FeatureRow]:
    """Features for "the next ``horizon`` days", taken from the newest history.

    The origin is the day **after** the last calendar day of the series, so the
    forecast window never overlaps recorded meals. Returns ``None`` only when
    there is not even one full look-back window available; a row with missing
    values is returned with :attr:`FeatureRow.missing` filled in so the UI can
    say precisely what is missing.
    """
    if series.is_empty or series.days < lookback:
        return None

    frame = series.frame.reset_index(drop=True)
    window = frame.iloc[-lookback:]
    last_day = pd.Timestamp(frame.at[len(frame) - 1, "date"])
    origin = (last_day + timedelta(days=1)).date()
    vector = feature_vector(window, origin)

    return FeatureRow(
        values=vector,
        origin=origin,
        window_start=pd.Timestamp(window.at[window.index[0], "date"]).date(),
        window_end=last_day.date(),
        logged_days=int(window["logged"].sum()),
        missing=missing_features(vector),
    )
