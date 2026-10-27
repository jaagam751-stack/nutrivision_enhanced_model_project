"""Meal model — the ``meals`` table (Phase 2).

One row per saved meal. Rows are only ever written after the user explicitly
clicks **Save Meal** (Phase 5); nothing in the app inserts meals automatically.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models.user import User


class Meal(Base):
    """A single saved meal with its estimated nutrition values."""

    __tablename__ = "meals"

    meal_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False, default=utcnow, index=True)

    food_name: Mapped[str] = mapped_column(String(120), nullable=False)
    portion_grams: Mapped[float] = mapped_column(Float, nullable=False)

    # Nutrition values for the selected portion (not per 100 g).
    calories: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    protein: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fat: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    carbohydrates: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fiber: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    health_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ``lazy="selectin"`` loads the parent user in the same round trip, so
    # ``meal.user`` stays usable after the repository session has closed
    # ``lazy="selectin"`` loads the parent user in the same round trip, so
    # ``obj.user`` stays usable after the repository session has closed.
    # The collections on User stay lazy to avoid loading every child row.
    user: Mapped["User"] = relationship(back_populates="meals", lazy="selectin")

    __table_args__ = (
        CheckConstraint("portion_grams > 0", name="ck_meals_portion_positive"),
        CheckConstraint("calories >= 0", name="ck_meals_calories_non_negative"),
        CheckConstraint("protein >= 0", name="ck_meals_protein_non_negative"),
        CheckConstraint("fat >= 0", name="ck_meals_fat_non_negative"),
        CheckConstraint("carbohydrates >= 0", name="ck_meals_carbs_non_negative"),
        CheckConstraint("fiber >= 0", name="ck_meals_fiber_non_negative"),
        CheckConstraint(
            "health_score >= 0 AND health_score <= 100", name="ck_meals_health_score_range"
        ),
        # Daily aggregation (dashboard + PNIE) filters by user and date range.
        Index("ix_meals_user_timestamp", "user_id", "timestamp"),
    )

    @property
    def calories_per_100g(self) -> float:
        """Energy density, useful for the dashboard tooltips."""
        return round(self.calories * 100.0 / self.portion_grams, 1) if self.portion_grams else 0.0
