"""SQLAlchemy models package (Phase 2).

Importing this package registers every table with ``Base.metadata``, which is
what ``Base.metadata.create_all()`` in ``database/init_db.py`` relies on.
"""

from models.base import Base, utcnow
from models.meal import Meal
from models.prediction import Prediction
from models.recommendation import Recommendation
from models.user import User

__all__ = [
    "Base",
    "utcnow",
    "User",
    "Meal",
    "Prediction",
    "Recommendation",
    "ALL_MODELS",
]

#: Convenience list used by tests, ``init_db`` and the Streamlit status panel.
ALL_MODELS = (User, Meal, Prediction, Recommendation)
