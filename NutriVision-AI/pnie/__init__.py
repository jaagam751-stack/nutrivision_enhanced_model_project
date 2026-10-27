"""PNIE — Predictive Nutrition Intelligence Engine (Phase 6).

The engine answers exactly one question with machine learning: *what will my
average daily calories be next week?* It uses a single XGBoost regressor per
user; every other number the Predictions page shows comes from transparent
arithmetic over the same saved meals.

Public API
----------
* :func:`run_pnie` — one call: daily totals in, :class:`PNIEOutcome` out.
* :class:`PNIEEngine` — the orchestrator (history checks → features → model →
  guarded forecast).
* :class:`PNIEOutcome` / :class:`PNIEPrediction` — what the UI renders.
* :class:`PNIEPredictor` — the XGBoost regressor on its own (chronological
  hold-out evaluation, refit on all rows for the live forecast).
* feature engineering helpers (:func:`build_daily_series`,
  :func:`build_training_rows`, :func:`latest_feature_row`) and the descriptive
  :func:`analyse_trends` / :class:`TrendAnalysis`.
"""

from pnie.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    DailySeries,
    FeatureDataset,
    FeatureRow,
    build_daily_series,
    build_training_rows,
    latest_feature_row,
    missing_features,
)
from pnie.pnie_engine import (
    MODE_AWAITING_RECENT_DATA,
    MODE_DESCRIPTIVE_ONLY,
    MODE_INSUFFICIENT_HISTORY,
    MODE_LABELS,
    MODE_MODEL,
    MODE_MODEL_UNAVAILABLE,
    MODE_NO_DATA,
    MODE_TONES,
    PNIEEngine,
    PNIEOutcome,
    PNIEPrediction,
    forecast_window,
    run_pnie,
)
from pnie.predictor import PNIEPredictor, TrainingReport, clamp_prediction, importance_table
from pnie.trend_analysis import (
    DIRECTION_FALLING,
    DIRECTION_RISING,
    DIRECTION_STABLE,
    TrendAnalysis,
    analyse_trends,
    direction_label,
    direction_of,
    linear_slope,
)

__all__ = [
    # feature engineering
    "FEATURE_COLUMNS",
    "FEATURE_DESCRIPTIONS",
    "DailySeries",
    "FeatureDataset",
    "FeatureRow",
    "build_daily_series",
    "build_training_rows",
    "latest_feature_row",
    "missing_features",
    # trends
    "TrendAnalysis",
    "analyse_trends",
    "linear_slope",
    "direction_of",
    "direction_label",
    "DIRECTION_RISING",
    "DIRECTION_FALLING",
    "DIRECTION_STABLE",
    # model
    "PNIEPredictor",
    "TrainingReport",
    "clamp_prediction",
    "importance_table",
    # engine
    "PNIEEngine",
    "PNIEOutcome",
    "PNIEPrediction",
    "run_pnie",
    "forecast_window",
    "MODE_NO_DATA",
    "MODE_INSUFFICIENT_HISTORY",
    "MODE_DESCRIPTIVE_ONLY",
    "MODE_AWAITING_RECENT_DATA",
    "MODE_MODEL_UNAVAILABLE",
    "MODE_MODEL",
    "MODE_LABELS",
    "MODE_TONES",
]
