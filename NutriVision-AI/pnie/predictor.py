"""The single PNIE regression model (Phase 6).

One XGBoost regressor, one target: **the average daily calories of the next
week**. It is trained per user on the weekly windows produced by
:mod:`pnie.feature_engineering`.

Honest evaluation
-----------------
Rows are chronological, so the hold-out is chronological too: the model is
trained on the oldest rows and scored on the newest ones
(:data:`config.PNIE_TEST_FRACTION`). A random split would let the model train on
the future and then "predict" it, which would make the reported error a fantasy.

The error that is shown to the user (MAE) comes from that hold-out evaluation.
The model that is then used for the live forecast is refit on *all* rows, which
is standard practice for a small dataset — both facts are stated on the
Predictions page and in the README rather than being hidden.

The model is deliberately tiny (depth-3 trees, few hundred rounds): with ~10-40
weekly rows anything larger would memorise noise. Training takes milliseconds, so
no model file is committed and nothing is persisted except the resulting
prediction row.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config import (
    PNIE_MIN_TEST_ROWS,
    PNIE_MIN_TRAINING_ROWS,
    PNIE_MODEL_NAME,
    PNIE_MODEL_VERSION,
    PNIE_PREDICTION_KCAL_BOUNDS,
    PNIE_RANDOM_STATE,
    PNIE_TEST_FRACTION,
    PNIE_XGB_PARAMS,
)
from pnie.feature_engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the dependency is missing
    from xgboost import XGBRegressor
except Exception as exc:  # noqa: BLE001 - any import problem is reported, not hidden
    XGBRegressor = None  # type: ignore[assignment]
    _XGBOOST_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    _XGBOOST_IMPORT_ERROR = ""


@dataclass(frozen=True, eq=False)
class TrainingReport:
    """What happened during training — shown to the user, not just logged."""

    trained: bool = False
    reason: str = ""
    rows: int = 0
    train_rows: int = 0
    test_rows: int = 0
    mae: Optional[float] = None
    rmse: Optional[float] = None
    baseline_mae: Optional[float] = None
    train_mae: Optional[float] = None
    holdout_from: Optional[date] = None
    refit_on_all: bool = True
    feature_importances: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def beats_baseline(self) -> Optional[bool]:
        """Is the model better than "next week = the average week so far"?"""
        if self.mae is None or self.baseline_mae in (None, 0):
            return None
        return self.mae < self.baseline_mae

    def describe(self) -> str:
        if not self.trained:
            return self.reason or "The model was not trained."
        return (
            f"Trained on {self.train_rows} weekly window(s) and evaluated on the "
            f"{self.test_rows} newest ones (chronological split): MAE "
            f"{self.mae:,.0f} kcal/day. Baseline (always predicting the mean) MAE "
            f"{self.baseline_mae:,.0f} kcal/day."
            if self.mae is not None and self.baseline_mae is not None
            else f"Trained on {self.train_rows} weekly window(s)."
        )


def clamp_prediction(value: float) -> tuple[float, bool]:
    """Keep a prediction inside the absolute sane bounds from ``config``.

    Returns ``(value, clamped)``. The bounds exist so a pathological model can
    never store something like "8 kcal/day" or "18000 kcal/day" in the database.
    """
    low, high = PNIE_PREDICTION_KCAL_BOUNDS
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return float((low + high) / 2.0), True
    if value < low:
        return float(low), True
    if value > high:
        return float(high), True
    return float(value), False


class PNIEPredictor:
    """Train and query the single XGBoost model.

    Usage::

        predictor = PNIEPredictor()
        report = predictor.fit(dataset.features, dataset.targets)
        if report.trained:
            kcal = predictor.predict(row.values)
    """

    def __init__(
        self,
        params: Optional[Mapping[str, Any]] = None,
        random_state: int = PNIE_RANDOM_STATE,
        test_fraction: float = PNIE_TEST_FRACTION,
        min_rows: int = PNIE_MIN_TRAINING_ROWS,
        min_test_rows: int = PNIE_MIN_TEST_ROWS,
    ) -> None:
        self.params: dict[str, Any] = {**PNIE_XGB_PARAMS, **(dict(params) if params else {})}
        self.params["random_state"] = random_state
        self.test_fraction = float(test_fraction)
        self.min_rows = int(min_rows)
        self.min_test_rows = int(min_test_rows)
        self.model_name = PNIE_MODEL_NAME
        self.model_version = PNIE_MODEL_VERSION
        self._model = None
        self.report: TrainingReport = TrainingReport(trained=False, reason="Not trained yet.")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def feature_importances(self) -> dict[str, float]:
        """Model gain per feature, normalised to sum to 1 (empty when unfitted)."""
        return dict(self.report.feature_importances)

    def _make_model(self):
        if XGBRegressor is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "XGBoost is not available in this environment "
                f"({_XGBOOST_IMPORT_ERROR}). Install it with: pip install -r requirements.txt"
            )
        return XGBRegressor(**self.params)

    def fit(self, features: pd.DataFrame, targets: Sequence[float]) -> TrainingReport:
        """Fit the model with a chronological train/test split.

        Returns a :class:`TrainingReport` instead of raising for the ordinary
        "not enough data" case, so the engine can fall back to descriptive
        statistics. A missing XGBoost installation is reported the same way.
        """
        features = pd.DataFrame(features)
        targets = pd.Series(list(targets), dtype="float64").reset_index(drop=True)
        # The weekly rows are indexed by origin date; keep that index so the
        # reported hold-out boundary can name the first evaluated week.
        original_index = features.index
        features = features.reset_index(drop=True)

        if XGBRegressor is None:  # pragma: no cover - dependency guard
            self.report = TrainingReport(
                trained=False,
                rows=len(features),
                reason=(
                    "XGBoost is not installed in this environment, so the predictive model "
                    f"cannot run ({_XGBOOST_IMPORT_ERROR}). Install requirements.txt and retry."
                ),
            )
            return self.report

        if len(features) != len(targets):
            raise ValueError("features and targets must have the same number of rows")

        rows = len(features)
        missing_columns = [name for name in FEATURE_COLUMNS if name not in features.columns]
        if missing_columns:
            raise ValueError(f"missing feature column(s): {', '.join(missing_columns)}")
        features = features[list(FEATURE_COLUMNS)]   # fixed order, always

        if features.isna().any().any() or targets.isna().any():
            raise ValueError("training data contains missing values")

        if rows < self.min_rows:
            self.report = TrainingReport(
                trained=False,
                rows=rows,
                reason=(
                    f"Only {rows} weekly window(s) of history: at least {self.min_rows} are needed "
                    "before a model is trained."
                ),
            )
            return self.report

        # --- chronological split: oldest rows train, newest rows score ------- #
        test_rows = int(round(rows * self.test_fraction))
        test_rows = max(self.min_test_rows, test_rows)
        test_rows = min(test_rows, max(0, rows - self.min_rows + 1), max(0, rows - 3))
        test_rows = max(0, test_rows)
        if test_rows == 0:  # pragma: no cover - only with min_rows <= 3
            test_rows = 1

        train_features = features.iloc[: rows - test_rows]
        train_targets = targets.iloc[: rows - test_rows]
        test_features = features.iloc[rows - test_rows:]
        test_targets = targets.iloc[rows - test_rows:]

        evaluation_model = self._make_model()
        evaluation_model.fit(train_features, train_targets)
        predictions = np.asarray(evaluation_model.predict(test_features), dtype=float)

        mae = float(np.mean(np.abs(predictions - test_targets.to_numpy())))
        rmse = float(np.sqrt(np.mean((predictions - test_targets.to_numpy()) ** 2)))
        baseline_mae = float(np.mean(np.abs(train_targets.mean() - test_targets.to_numpy())))
        train_mae = float(
            np.mean(np.abs(np.asarray(evaluation_model.predict(train_features), dtype=float)
                           - train_targets.to_numpy()))
        )
        importances = _normalise_importances(
            getattr(evaluation_model, "feature_importances_", None), FEATURE_COLUMNS
        )

        # Refit on every row for the live forecast, keeping the hold-out MAE as
        # the reported error estimate (documented in the module docstring).
        final_model = self._make_model()
        final_model.fit(features, targets)
        self._model = final_model

        holdout_from = None
        if isinstance(original_index, pd.DatetimeIndex) and len(original_index) == rows:
            holdout_from = pd.Timestamp(original_index[rows - test_rows]).date()

        self.report = TrainingReport(
            trained=True,
            rows=rows,
            train_rows=len(train_features),
            test_rows=len(test_features),
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            baseline_mae=round(baseline_mae, 2),
            train_mae=round(train_mae, 2),
            holdout_from=holdout_from,
            refit_on_all=True,
            feature_importances=importances,
            params={key: value for key, value in self.params.items() if key != "random_state"},
        )
        logger.info("PNIE model trained | %s", self.report.describe())
        return self.report

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self, values: Mapping[str, Optional[float]] | pd.Series, clamp: bool = True
    ) -> float:
        """Predict next week's average daily calories for one feature row.

        ``clamp=True`` (the default) applies the absolute sanity bounds from
        :data:`config.PNIE_PREDICTION_KCAL_BOUNDS`. The engine passes
        ``clamp=False`` so it can report *which* guard fired — the absolute
        bound and the relative deviation limit are applied and disclosed
        separately.
        """
        if not self.is_fitted:
            raise RuntimeError("The PNIE model has not been trained yet — call fit() first.")

        row = dict(values)
        missing = [
            name for name in FEATURE_COLUMNS
            if row.get(name) is None
            or (isinstance(row.get(name), float) and math.isnan(float(row[name])))
        ]
        if missing:
            raise ValueError(f"cannot predict: missing feature(s) {', '.join(missing)}")

        frame = pd.DataFrame([[float(row[name]) for name in FEATURE_COLUMNS]], columns=list(FEATURE_COLUMNS))
        raw = float(np.asarray(self._model.predict(frame), dtype=float)[0])
        if not clamp:
            return raw
        value, was_clamped = clamp_prediction(raw)
        if was_clamped:
            logger.warning("PNIE prediction clamped: raw=%.1f -> %.1f", raw, value)
        return value

    def predict_many(self, features: pd.DataFrame) -> np.ndarray:
        """Vectorised prediction (used by the tests and for batch sanity checks)."""
        if not self.is_fitted:
            raise RuntimeError("The PNIE model has not been trained yet — call fit() first.")
        frame = pd.DataFrame(features)[list(FEATURE_COLUMNS)]
        raw = np.asarray(self._model.predict(frame), dtype=float)
        return np.array([clamp_prediction(float(value))[0] for value in raw])


def _normalise_importances(raw: Any, names: Sequence[str]) -> dict[str, float]:
    """Turn XGBoost's importance array into a ``{feature: share}`` mapping."""
    if raw is None:
        return {}
    values = np.asarray(raw, dtype=float)
    if values.size != len(names):
        return {}
    total = float(values.sum())
    if total <= 0:
        return {name: 0.0 for name in names}
    return {name: round(float(value) / total, 4) for name, value in zip(names, values)}


def importance_table(importances: Mapping[str, float], top: Optional[int] = None) -> list[tuple[str, float]]:
    """Importances sorted from most to least important, optionally truncated."""
    rows = sorted((importances or {}).items(), key=lambda item: item[1], reverse=True)
    return rows[:top] if top else rows
