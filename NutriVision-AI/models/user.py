"""User model — the ``users`` table (Phase 2).

Mirrors the Streamlit onboarding profile from Phase 1. ``UserProfile`` (in
``utils/helpers.py``) is imported lazily inside methods so that this module —
and therefore the CLI in ``database/init_db.py`` — never requires Streamlit.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, CheckConstraint, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from utils.helpers import UserProfile

    from models.meal import Meal
    from models.prediction import Prediction
    from models.recommendation import Recommendation


class User(Base):
    """A NutriVision AI user profile."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    activity_level: Mapped[str] = mapped_column(String(80), nullable=False)
    diet_preference: Mapped[str] = mapped_column(String(40), nullable=False)
    fitness_goal: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)

    # True only for rows created by `python database/init_db.py --sample-data`.
    # Marking demo records in the schema (instead of guessing from the name)
    # keeps sample data strictly separated from real user records.
    is_sample: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships ------------------------------------------------------- #
    meals: Mapped[List["Meal"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Meal.timestamp",
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Prediction.prediction_date",
    )
    recommendations: Mapped[List["Recommendation"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Recommendation.timestamp",
    )

    __table_args__ = (
        CheckConstraint("age > 0 AND age < 130", name="ck_users_age_range"),
        CheckConstraint("height_cm > 0", name="ck_users_height_positive"),
        CheckConstraint("weight_kg > 0", name="ck_users_weight_positive"),
    )

    # Profile bridging ----------------------------------------------------- #
    @classmethod
    def from_profile(cls, profile: "UserProfile") -> "User":
        """Build a new ``User`` row from a validated :class:`UserProfile`."""
        return cls(
            name=profile.name,
            age=int(profile.age),
            gender=profile.gender,
            height_cm=float(profile.height_cm),
            weight_kg=float(profile.weight_kg),
            activity_level=profile.activity_level,
            diet_preference=profile.diet_preference,
            fitness_goal=profile.fitness_goal,
            is_sample=False,
        )

    def update_from_profile(self, profile: "UserProfile") -> None:
        """Copy editable fields from a profile onto this row (in place)."""
        self.name = profile.name
        self.age = int(profile.age)
        self.gender = profile.gender
        self.height_cm = float(profile.height_cm)
        self.weight_kg = float(profile.weight_kg)
        self.activity_level = profile.activity_level
        self.diet_preference = profile.diet_preference
        self.fitness_goal = profile.fitness_goal

    def to_profile(self) -> "UserProfile":
        """Convert this row back into a :class:`UserProfile` (Phase 1 object)."""
        from utils.helpers import UserProfile  # lazy: avoids a Streamlit import at module load

        return UserProfile(
            name=self.name,
            age=self.age,
            gender=self.gender,
            height_cm=self.height_cm,
            weight_kg=self.weight_kg,
            activity_level=self.activity_level,
            diet_preference=self.diet_preference,
            fitness_goal=self.fitness_goal,
            created_at=self.created_at.isoformat(timespec="seconds") if self.created_at else "",
            user_id=self.user_id,
        )
