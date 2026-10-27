"""Prediction model — the ``predictions`` table (Phase 2).

Rows are written by the Predictive Nutrition Intelligence Engine (PNIE, Phase 6)
after the XGBoost model has been trained on the user's *past* meals. Model
metadata is stored alongside the value so a result can always be traced back to
the data and settings that produced it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models.user import User


class Prediction(Base):
    """One PNIE forecast for one user."""

    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    prediction_date: Mapped[datetime] = mapped_column(nullable=False, default=utcnow, index=True)

    # --- the forecast ----------------------------------------------------- #
    # Target of the single XGBoost model: average daily calories for the next
    # 7 days. `trend_direction` is a plain description of that number compared
    # with the recent average — it is not a second model.
    predicted_next_week_avg_calories: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trend_direction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    trend_change_kcal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- model metadata --------------------------------------------------- #
    model_name: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feature_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # ``lazy="selectin"`` loads the parent user in the same round trip, so
    # ``meal.user`` stays usable after the repository session has closed
    # ``lazy="selectin"`` loads the parent user in the same round trip, so
    # ``obj.user`` stays usable after the repository session has closed.
    # The collections on User stay lazy to avoid loading every child row.
    user: Mapped["User"] = relationship(back_populates="predictions", lazy="selectin")

    __table_args__ = (
        CheckConstraint(
            "predicted_next_week_avg_calories IS NULL OR predicted_next_week_avg_calories >= 0",
            name="ck_predictions_calories_non_negative",
        ),
        CheckConstraint(
            "mae IS NULL OR mae >= 0",
            name="ck_predictions_mae_non_negative",
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary representation (JSON friendly)."""
        return super().to_dict()
