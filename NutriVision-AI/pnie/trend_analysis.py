"""Descriptive trend analysis for the PNIE (Phase 6).

This module contains **no machine learning**. It turns a
:class:`pnie.feature_engineering.DailySeries` into honest descriptive statistics:
averages, spread, week-over-week change, a least-squares slope and the macro
energy split.

Why it exists
-------------
A forecast is only defensible when there is enough history to learn from. Below
that threshold the engine reports these statistics instead of a made-up number,
which is exactly what the project brief requires ("if history is insufficient,
show descriptive statistics instead of fabricating a prediction").

Everything here is pure Python/NumPy/Pandas: no database, no Streamlit, so it is
unit-testable on hand-built series.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from config import (
    PNIE_HORIZON_DAYS,
    PNIE_MIN_HISTORY_DAYS,
    PNIE_MIN_LOGGED_DAYS,
    PNIE_STABLE_BAND_KCAL,
)

logger = logging.getLogger(__name__)

#: Energy factors (kcal per gram) used for the macro energy split.
KCAL_PER_GRAM: dict[str, float] = {"protein": 4.0, "carbohydrates": 4.0, "fat": 9.0}

DIRECTION_RISING: str = "rising"
DIRECTION_FALLING: str = "falling"
DIRECTION_STABLE: str = "stable"

_DIRECTION_LABEL: dict[str, str] = {
    DIRECTION_RISING: "trending up",
    DIRECTION_FALLING: "trending down",
    DIRECTION_STABLE: "holding steady",
}


# --------------------------------------------------------------------------- #
# Small maths helpers (shared with the feature engineering)
# --------------------------------------------------------------------------- #
def linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    """Least-squares slope of ``y`` against ``x`` (``None`` when undefined).

    Used for "kcal per day" trends. At least two points are required, and a
    perfectly vertical fit (all x equal) returns ``None`` instead of ``inf``.
    """
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    if x.size != y.size or x.size < 2:
        return None

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator <= 0 or math.isnan(denominator):
        return None

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    return None if math.isnan(slope) or math.isinf(slope) else slope


def direction_of(change_kcal: Optional[float], stable_band: float = PNIE_STABLE_BAND_KCAL) -> str:
    """Classify a change in kcal as rising / falling / stable.

    The band exists because day-to-day nutrition data is noisy: a 5 kcal
    difference is not a trend, and calling it one would be misleading.
    """
    if change_kcal is None:
        return DIRECTION_STABLE
    if change_kcal > stable_band:
        return DIRECTION_RISING
    if change_kcal < -stable_band:
        return DIRECTION_FALLING
    return DIRECTION_STABLE


def direction_label(direction: str) -> str:
    """Human-readable form of a direction code (for pills and captions)."""
    return _DIRECTION_LABEL.get(direction, direction)


def _mean(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class TrendAnalysis:
    """Descriptive statistics for one user's daily calorie history."""

    span_days: int = 0
    logged_days: int = 0
    meals: int = 0
    coverage_pct: float = 0.0

    mean_calories: Optional[float] = None
    median_calories: Optional[float] = None
    std_calories: Optional[float] = None
    min_calories: Optional[float] = None
    max_calories: Optional[float] = None

    latest_week_avg: Optional[float] = None
    previous_week_avg: Optional[float] = None
    change_kcal: Optional[float] = None
    change_pct: Optional[float] = None

    slope_per_day: Optional[float] = None
    slope_per_week: Optional[float] = None
    direction: str = DIRECTION_STABLE

    mean_protein: Optional[float] = None
    mean_fat: Optional[float] = None
    mean_carbohydrates: Optional[float] = None
    mean_fiber: Optional[float] = None
    mean_health_score: Optional[float] = None
    energy_share: Optional[dict[str, float]] = None

    highest_day: Optional[tuple[str, float]] = None
    lowest_day: Optional[tuple[str, float]] = None

    has_enough_history: bool = False
    min_history_days: int = PNIE_MIN_HISTORY_DAYS
    min_logged_days: int = PNIE_MIN_LOGGED_DAYS
    message: str = ""

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_rows(self) -> list[tuple[str, str]]:
        """Label/value rows for the UI's ``ui.stat_rows`` component."""
        rows: list[tuple[str, str]] = [
            ("Days covered", f"{self.span_days} ({self.logged_days} days with meals)"),
            ("Log consistency", f"{self.coverage_pct:.0f}% of days logged"),
        ]
        if self.mean_calories is not None:
            rows.append(("Average per logged day", f"{self.mean_calories:,.0f} kcal"))
        if self.std_calories is not None:
            rows.append(
                ("Typical spread",
                 f"± {self.std_calories:,.0f} kcal between days"),
            )
        if self.min_calories is not None and self.max_calories is not None:
            rows.append(
                ("Lowest / highest day",
                 f"{self.min_calories:,.0f} / {self.max_calories:,.0f} kcal"),
            )
        if self.latest_week_avg is not None:
            rows.append(("Last 7 days", f"{self.latest_week_avg:,.0f} kcal/day"))
        if self.previous_week_avg is not None:
            rows.append(("Previous 7 days", f"{self.previous_week_avg:,.0f} kcal/day"))
        if self.change_kcal is not None:
            rows.append(
                ("Week-over-week change",
                 f"{self.change_kcal:+,.0f} kcal ({self.change_pct:+.1f}%)"
                 if self.change_pct is not None
                 else f"{self.change_kcal:+,.0f} kcal"),
            )
        if self.slope_per_week is not None:
            rows.append(("Direction", f"{direction_label(self.direction)} "
                                      f"({self.slope_per_week:+,.0f} kcal/week)"))
        if self.mean_protein is not None:
            rows.append(("Average protein", f"{self.mean_protein:,.1f} g/day"))
        if self.mean_fiber is not None:
            rows.append(("Average fibre", f"{self.mean_fiber:,.1f} g/day"))
        if self.mean_health_score is not None:
            rows.append(("Average health score", f"{self.mean_health_score:,.1f} / 100"))
        if self.energy_share:
            share = self.energy_share
            rows.append(
                ("Energy from macros",
                 f"{share['protein'] * 100:.0f}% protein · {share['fat'] * 100:.0f}% fat · "
                 f"{share['carbohydrates'] * 100:.0f}% carbs"),
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot (no NaN: missing values stay ``None``)."""
        return {
            "span_days": self.span_days,
            "logged_days": self.logged_days,
            "meals": self.meals,
            "coverage_pct": round(self.coverage_pct, 1),
            "mean_calories": _round(self.mean_calories),
            "median_calories": _round(self.median_calories),
            "std_calories": _round(self.std_calories),
            "min_calories": _round(self.min_calories),
            "max_calories": _round(self.max_calories),
            "latest_week_avg": _round(self.latest_week_avg),
            "previous_week_avg": _round(self.previous_week_avg),
            "change_kcal": _round(self.change_kcal),
            "change_pct": _round(self.change_pct, 2),
            "slope_per_day": _round(self.slope_per_day, 2),
            "slope_per_week": _round(self.slope_per_week, 1),
            "direction": self.direction,
            "mean_protein": _round(self.mean_protein),
            "mean_fat": _round(self.mean_fat),
            "mean_carbohydrates": _round(self.mean_carbohydrates),
            "mean_fiber": _round(self.mean_fiber),
            "mean_health_score": _round(self.mean_health_score),
            "energy_share": (
                {key: round(value, 4) for key, value in self.energy_share.items()}
                if self.energy_share
                else None
            ),
            "highest_day": list(self.highest_day) if self.highest_day else None,
            "lowest_day": list(self.lowest_day) if self.lowest_day else None,
            "has_enough_history": self.has_enough_history,
            "min_history_days": self.min_history_days,
            "min_logged_days": self.min_logged_days,
            "message": self.message,
        }


def _round(value: Optional[float], decimals: int = 1) -> Optional[float]:
    """Round while keeping ``None`` (and never emitting NaN into JSON)."""
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, decimals)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse_trends(
    series,
    min_history_days: int = PNIE_MIN_HISTORY_DAYS,
    min_logged_days: int = PNIE_MIN_LOGGED_DAYS,
    window_days: int = PNIE_HORIZON_DAYS,
    stable_band: float = PNIE_STABLE_BAND_KCAL,
) -> TrendAnalysis:
    """Describe a daily series without predicting anything.

    ``window_days`` is the length of the "last" and "previous" blocks compared
    week-over-week (7 by default, matching the forecast horizon).
    """
    frame = getattr(series, "frame", None)
    if frame is None or frame.empty:
        return TrendAnalysis(
            min_history_days=min_history_days,
            min_logged_days=min_logged_days,
            message="No meals have been saved yet, so there is nothing to describe.",
        )

    span_days = int(len(frame))
    # Only days that actually carry meals are statistics; the rest are gaps.
    logged = frame[frame["logged"]].copy()
    logged_days = int(len(logged))
    meals = int(frame["meals"].sum())

    if logged_days == 0:
        return TrendAnalysis(
            span_days=span_days,
            min_history_days=min_history_days,
            min_logged_days=min_logged_days,
            message="No meals have been saved yet, so there is nothing to describe.",
        )

    calories = pd.to_numeric(logged["calories"], errors="coerce")
    day_index = (logged["date"] - frame["date"].iloc[0]).dt.days
    slope_per_day = linear_slope(day_index.tolist(), calories.tolist())

    # "Last week" and "the week before" are compared over *logged* days so a
    # partially recorded week is not mixed up with missing data.
    previous_block = logged.iloc[:-window_days] if len(logged) > window_days else logged.iloc[0:0]
    latest = _window_mean(logged, window_days)
    previous = _window_mean(previous_block, window_days)
    change = latest - previous if (latest is not None and previous is not None) else None
    change_pct = (
        change / previous * 100.0
        if (change is not None and previous not in (None, 0))
        else None
    )

    protein = _mean(logged["protein"])
    fat = _mean(logged["fat"])
    carbs = _mean(logged["carbohydrates"])

    highest = logged.loc[calories.idxmax()] if not calories.empty else None
    lowest = logged.loc[calories.idxmin()] if not calories.empty else None

    slope_per_week = slope_per_day * 7.0 if slope_per_day is not None else None
    has_enough = span_days >= min_history_days and logged_days >= min_logged_days
    analysis = TrendAnalysis(
        span_days=span_days,
        logged_days=logged_days,
        meals=meals,
        coverage_pct=(logged_days / span_days * 100.0) if span_days else 0.0,
        mean_calories=_mean(calories),
        median_calories=(float(calories.median()) if not calories.empty else None),
        std_calories=(float(calories.std(ddof=1)) if len(calories) > 1 else None),
        min_calories=(float(calories.min()) if not calories.empty else None),
        max_calories=(float(calories.max()) if not calories.empty else None),
        latest_week_avg=latest,
        previous_week_avg=previous,
        change_kcal=change,
        change_pct=change_pct,
        slope_per_day=slope_per_day,
        slope_per_week=slope_per_week,
        direction=direction_of(slope_per_week, stable_band),
        mean_protein=protein,
        mean_fat=fat,
        mean_carbohydrates=carbs,
        mean_fiber=_mean(logged["fiber"]),
        mean_health_score=_mean(logged["avg_health_score"]),
        energy_share=_energy_share(protein, fat, carbs),
        highest_day=(_iso(highest["date"]), float(highest["calories"])) if highest is not None else None,
        lowest_day=(_iso(lowest["date"]), float(lowest["calories"])) if lowest is not None else None,
        has_enough_history=has_enough,
        min_history_days=min_history_days,
        min_logged_days=min_logged_days,
    )
    return _with_message(analysis)


def _window_mean(logged: pd.DataFrame, window_days: int) -> Optional[float]:
    """Mean calories over the last ``window_days`` *logged* days of the frame."""
    if window_days <= 0 or logged.empty:
        return None
    block = logged.tail(window_days)
    return _mean(block["calories"]) if not block.empty else None


def _energy_share(
    protein: Optional[float], fat: Optional[float], carbs: Optional[float]
) -> Optional[dict[str, float]]:
    """Share of *macro* energy coming from protein, fat and carbohydrate.

    Fibre and other energy are excluded on purpose: the shares describe the
    protein/fat/carbohydrate mix, which is what the plot shows.
    """
    if protein is None or fat is None or carbs is None:
        return None
    parts = {
        "protein": protein * KCAL_PER_GRAM["protein"],
        "fat": fat * KCAL_PER_GRAM["fat"],
        "carbohydrates": carbs * KCAL_PER_GRAM["carbohydrates"],
    }
    total = sum(parts.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in parts.items()}


def _iso(value: Any) -> str:
    """``date``/``Timestamp`` -> ``YYYY-MM-DD`` string."""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _with_message(analysis: TrendAnalysis) -> TrendAnalysis:
    """Attach a plain-language summary of what the numbers mean."""
    if not analysis.has_enough_history:
        missing_days = max(0, analysis.min_history_days - analysis.span_days)
        missing_logs = max(0, analysis.min_logged_days - analysis.logged_days)
        needs = []
        if missing_days:
            needs.append(f"{missing_days} more day(s) of history")
        if missing_logs:
            needs.append(f"{missing_logs} more logged day(s)")
        message = (
            f"History so far: {analysis.span_days} day(s) with {analysis.logged_days} logged. "
            f"Descriptive statistics only — the engine needs "
            f"{analysis.min_history_days} days and {analysis.min_logged_days} logged days before "
            "it will make a forecast. "
            + ("Keep logging: " + " and ".join(needs) + "." if needs else "")
        ).strip()
    else:
        message = (
            f"Based on {analysis.logged_days} logged day(s) across {analysis.span_days} days "
            f"({analysis.coverage_pct:.0f}% coverage). Average intake on logged days is "
            f"{analysis.mean_calories:,.0f} kcal."
            if analysis.mean_calories is not None
            else f"Based on {analysis.logged_days} logged day(s) across {analysis.span_days} days."
        )

    return replace(analysis, message=message)
