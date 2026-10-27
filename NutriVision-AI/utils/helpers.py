"""Shared helpers: the user profile model, health maths, validation and
Streamlit session-state utilities.

Keeping this logic out of ``app.py`` means every future page (Dashboard,
Upload Meal, Profile, ...) reuses exactly the same rules, and the pure
functions can be unit tested without running Streamlit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from config import (
    ACTIVITY_LEVELS,
    AGE_RANGE,
    BMI_BANDS,
    BMI_TONE,
    DIET_PREFERENCES,
    FITNESS_GOALS,
    FIBER_G_PER_1000_KCAL,
    GENDERS,
    GOAL_CALORIE_FACTOR,
    GOAL_PROTEIN_PER_KG,
    HEIGHT_RANGE_CM,
    MIN_SAFE_CALORIES,
    NAME_MAX_LENGTH,
    WEIGHT_RANGE_KG,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Session-state keys (declared once so pages never fight over string literals)
# --------------------------------------------------------------------------- #
SESSION_PROFILE: str = "nv_profile"
SESSION_EDITING: str = "nv_editing_profile"
SESSION_FLASH: str = "nv_flash_message"
SESSION_CONFIRM_RESET: str = "nv_confirm_reset"
SESSION_FORM_EPOCH: str = "nv_form_epoch"

NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")


# --------------------------------------------------------------------------- #
# Pure health calculations (no Streamlit, easy to unit test)
# --------------------------------------------------------------------------- #
def bmi_of(weight_kg: float, height_cm: float) -> float:
    """Body Mass Index = kg / m²."""
    height_m = height_cm / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    return round(weight_kg / (height_m ** 2), 1)


def bmi_category_of(bmi: float) -> str:
    """Return the Asian-Indian BMI band label for a BMI value."""
    for low, high, label in BMI_BANDS:
        if low <= bmi < high:
            return label
    return "Obese"


def bmr_of(age: int, gender: str, height_cm: float, weight_kg: float) -> float:
    """Basal Metabolic Rate using the Mifflin-St Jeor equation (kcal/day).

    For ``Other`` the male and female constants are averaged, because no single
    published constant exists for that option. This is documented rather than
    hidden.
    """
    base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    if gender == "Male":
        return round(base + 5, 0)
    if gender == "Female":
        return round(base - 161, 0)
    return round(base + (5 - 161) / 2, 0)


def tdee_of(bmr: float, activity_level: str) -> float:
    """Total Daily Energy Expenditure = BMR × activity multiplier."""
    multiplier = ACTIVITY_LEVELS.get(activity_level, {}).get("multiplier", 1.2)
    return round(bmr * float(multiplier), 0)


def calorie_target_of(tdee: float, fitness_goal: str, gender: str) -> float:
    """Goal-adjusted daily calorie target, floored at a minimum safe intake."""
    factor = GOAL_CALORIE_FACTOR.get(fitness_goal, 1.0)
    floor = MIN_SAFE_CALORIES.get(gender, 1300.0)
    return round(max(tdee * factor, floor), 0)


# --------------------------------------------------------------------------- #
# Profile model
# --------------------------------------------------------------------------- #
@dataclass
class UserProfile:
    """A validated user profile.

    In Phase 1 the object lives in Streamlit session state. From Phase 2 it is
    persisted to the Neon PostgreSQL ``users`` table (``user_id`` is filled
    after insertion).
    """

    name: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    activity_level: str
    diet_preference: str
    fitness_goal: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    user_id: Optional[int] = None

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary (JSON/DB friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Rebuild a profile from a dictionary, ignoring unknown keys."""
        allowed = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    # -- derived health values --------------------------------------------- #
    @property
    def bmi(self) -> float:
        return bmi_of(self.weight_kg, self.height_cm)

    @property
    def bmi_category(self) -> str:
        return bmi_category_of(self.bmi)

    @property
    def bmi_tone(self) -> str:
        return BMI_TONE.get(self.bmi_category, "default")

    @property
    def activity_multiplier(self) -> float:
        return float(ACTIVITY_LEVELS.get(self.activity_level, {}).get("multiplier", 1.2))

    @property
    def bmr(self) -> float:
        """Basal Metabolic Rate (kcal/day)."""
        return bmr_of(self.age, self.gender, self.height_cm, self.weight_kg)

    @property
    def maintenance_calories(self) -> float:
        """Maintenance calories = TDEE (kcal/day)."""
        return tdee_of(self.bmr, self.activity_level)

    @property
    def daily_calorie_target(self) -> float:
        """Goal-adjusted daily calorie target (kcal/day)."""
        return calorie_target_of(self.maintenance_calories, self.fitness_goal, self.gender)

    @property
    def protein_target_g(self) -> float:
        """Daily protein target in grams."""
        return round(self.weight_kg * GOAL_PROTEIN_PER_KG.get(self.fitness_goal, 1.2), 0)

    @property
    def fiber_target_g(self) -> float:
        """Daily fibre target in grams (14 g per 1000 kcal)."""
        return round(self.daily_calorie_target * FIBER_G_PER_1000_KCAL / 1000.0, 0)

    @property
    def goal_calorie_delta(self) -> float:
        """Difference between target and maintenance calories (negative = deficit)."""
        return round(self.daily_calorie_target - self.maintenance_calories, 0)

    def summary_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs used to render the profile card."""
        return [
            ("Age", f"{self.age} years"),
            ("Gender", self.gender),
            ("Height", f"{self.height_cm:g} cm"),
            ("Weight", f"{self.weight_kg:g} kg"),
            ("Activity level", self.activity_level),
            ("Diet preference", self.diet_preference),
            ("Fitness goal", self.fitness_goal),
            ("Profile created", self.created_at.replace("T", " · ")),
        ]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _as_number(value: Any) -> Optional[float]:
    """Best-effort conversion of a widget value to ``float``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def validate_profile_inputs(raw: Dict[str, Any]) -> Dict[str, str]:
    """Validate raw form values.

    Returns a mapping of ``field name -> human readable error``. An empty
    dictionary means the input is valid.
    """
    errors: Dict[str, str] = {}

    name = str(raw.get("name", "")).strip()
    if not name:
        errors["name"] = "Please enter your name."
    elif len(name) > NAME_MAX_LENGTH:
        errors["name"] = f"Name must be {NAME_MAX_LENGTH} characters or fewer."
    elif not NAME_PATTERN.match(name):
        errors["name"] = "Name may only contain letters, spaces, apostrophes, hyphens and dots."

    age = _as_number(raw.get("age"))
    if age is None:
        errors["age"] = "Please enter your age."
    elif not (AGE_RANGE[0] <= age <= AGE_RANGE[1]):
        errors["age"] = f"Age must be between {AGE_RANGE[0]} and {AGE_RANGE[1]} years."

    if raw.get("gender") not in GENDERS:
        errors["gender"] = "Please select a gender."

    height = _as_number(raw.get("height_cm"))
    if height is None:
        errors["height_cm"] = "Please enter your height in centimetres."
    elif not (HEIGHT_RANGE_CM[0] <= height <= HEIGHT_RANGE_CM[1]):
        errors["height_cm"] = (
            f"Height must be between {HEIGHT_RANGE_CM[0]:g} cm and {HEIGHT_RANGE_CM[1]:g} cm."
        )

    weight = _as_number(raw.get("weight_kg"))
    if weight is None:
        errors["weight_kg"] = "Please enter your weight in kilograms."
    elif not (WEIGHT_RANGE_KG[0] <= weight <= WEIGHT_RANGE_KG[1]):
        errors["weight_kg"] = (
            f"Weight must be between {WEIGHT_RANGE_KG[0]:g} kg and {WEIGHT_RANGE_KG[1]:g} kg."
        )

    if raw.get("activity_level") not in ACTIVITY_LEVELS:
        errors["activity_level"] = "Please select your activity level."

    if raw.get("diet_preference") not in DIET_PREFERENCES:
        errors["diet_preference"] = "Please select a diet preference."

    if raw.get("fitness_goal") not in FITNESS_GOALS:
        errors["fitness_goal"] = "Please select a fitness goal."

    return errors


def build_profile(raw: Dict[str, Any]) -> UserProfile:
    """Validate raw input and return a :class:`UserProfile`.

    Raises:
        ValueError: if the input fails validation (message lists every problem).
    """
    errors = validate_profile_inputs(raw)
    if errors:
        raise ValueError("; ".join(errors.values()))

    profile = UserProfile(
        name=str(raw["name"]).strip(),
        age=int(_as_number(raw["age"]) or 0),
        gender=str(raw["gender"]),
        height_cm=float(_as_number(raw["height_cm"]) or 0.0),
        weight_kg=float(_as_number(raw["weight_kg"]) or 0.0),
        activity_level=str(raw["activity_level"]),
        diet_preference=str(raw["diet_preference"]),
        fitness_goal=str(raw["fitness_goal"]),
    )
    logger.info(
        "Profile built | age=%s gender=%s goal=%s bmi=%.1f target_kcal=%.0f",
        profile.age,
        profile.gender,
        profile.fitness_goal,
        profile.bmi,
        profile.daily_calorie_target,
    )
    return profile


# --------------------------------------------------------------------------- #
# Session-state utilities
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Create every session-state key the app expects (idempotent)."""
    defaults: Dict[str, Any] = {
        SESSION_PROFILE: None,
        SESSION_EDITING: False,
        SESSION_FLASH: None,
        SESSION_CONFIRM_RESET: False,
        SESSION_FORM_EPOCH: 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_profile() -> Optional[UserProfile]:
    """Return the stored profile, or ``None`` if onboarding is not complete."""
    stored = st.session_state.get(SESSION_PROFILE)
    if stored is None:
        return None
    if isinstance(stored, UserProfile):
        return stored
    if isinstance(stored, dict):  # e.g. restored after a hot reload
        return UserProfile.from_dict(stored)
    return None


def save_profile(profile: UserProfile) -> None:
    """Store a profile in session state and exit edit mode."""
    st.session_state[SESSION_PROFILE] = profile
    st.session_state[SESSION_EDITING] = False
    logger.info("Profile stored in session state for '%s'", profile.name)


def form_epoch() -> int:
    """Return the current form generation counter.

    Widget keys are suffixed with this number, so bumping it hands the user a
    genuinely blank form instead of one pre-filled with earlier input.
    """
    return int(st.session_state.get(SESSION_FORM_EPOCH, 0))


def bump_form_epoch() -> None:
    """Invalidate the current form widgets (used on reset)."""
    st.session_state[SESSION_FORM_EPOCH] = form_epoch() + 1


def clear_profile() -> None:
    """Remove the profile and every profile-related flag from session state."""
    st.session_state[SESSION_PROFILE] = None
    st.session_state[SESSION_EDITING] = False
    st.session_state[SESSION_CONFIRM_RESET] = False
    bump_form_epoch()
    logger.info("Profile cleared from session state")


def reset_cached_resources() -> None:
    """Clear Streamlit resource caches after the database configuration changes.

    Wrapped in ``try/except`` so that importing this module outside a Streamlit
    runtime (tests, ``init_db.py``) stays safe.
    """
    try:
        st.cache_resource.clear()
    except Exception:  # pragma: no cover - only when no caches exist yet
        pass


def is_editing_profile() -> bool:
    return bool(st.session_state.get(SESSION_EDITING, False))


def set_editing_profile(value: bool) -> None:
    st.session_state[SESSION_EDITING] = value


def flash(message: str, level: str = "success") -> None:
    """Queue a one-shot message to be shown on the next script run."""
    st.session_state[SESSION_FLASH] = {"message": message, "level": level}


def render_pending_flash() -> None:
    """Display and clear a queued flash message."""
    payload = st.session_state.get(SESSION_FLASH)
    if not payload:
        return
    st.session_state[SESSION_FLASH] = None
    renderer = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }.get(payload.get("level", "success"), st.success)
    renderer(payload.get("message", ""))


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_number(value: Optional[float], decimals: int = 0) -> str:
    """Format a number for display, tolerating ``None``."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def friendly_timestamp(iso_value: Optional[str]) -> str:
    """Convert an ISO timestamp into ``14 Oct 2026 · 09:31`` for display."""
    if not iso_value:
        return "—"
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return str(iso_value)
    return parsed.strftime("%d %b %Y · %H:%M")
