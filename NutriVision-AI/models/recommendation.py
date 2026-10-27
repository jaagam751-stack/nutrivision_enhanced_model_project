"""Recommendation model — the ``recommendations`` table (Phase 2).

Stores the personalised nutrition guidance produced by the Gemini 2.5 Flash
coach (Phase 7). ``source`` records what generated the text so AI output is
never confused with the rule-based nutrition engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models.user import User


class Recommendation(Base):
    """A stored AI coaching message for one user."""

    __tablename__ = "recommendations"

    recommendation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False, default=utcnow, index=True)
    recommendation_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Which engine produced the text: "gemini-2.5-flash" for the AI coach.
    # Kept explicit so AI output is always traceable and never presented as
    # clinical advice.
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="gemini-2.5-flash")
    # The question the user asked, for context when reviewing history.
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ``lazy="selectin"`` loads the parent user in the same round trip, so
    # ``meal.user`` stays usable after the repository session has closed
    # ``lazy="selectin"`` loads the parent user in the same round trip, so
    # ``obj.user`` stays usable after the repository session has closed.
    # The collections on User stay lazy to avoid loading every child row.
    user: Mapped["User"] = relationship(back_populates="recommendations", lazy="selectin")

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary representation (JSON friendly)."""
        return super().to_dict()
