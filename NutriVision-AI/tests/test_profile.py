"""Tests for NutriVision AI Phase 1: profile validation and health calculations.

Run from the project root:

    pytest -v

Only Streamlit-free logic is tested here, which is exactly why the maths and
validation live in ``utils/helpers.py`` and not inside ``app.py``.
"""

from __future__ import annotations

import pytest

from config import GOAL_CALORIE_FACTOR, MIN_SAFE_CALORIES
from utils.helpers import (
    UserProfile,
    bmi_category_of,
    bmi_of,
    bmr_of,
    build_profile,
    calorie_target_of,
    tdee_of,
    validate_profile_inputs,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def valid_input() -> dict:
    return {
        "name": "Aarav Sharma",
        "age": 28,
        "gender": "Male",
        "height_cm": 175.0,
        "weight_kg": 72.0,
        "activity_level": "Moderately active (moderate exercise 3-5 days/week)",
        "diet_preference": "Vegetarian",
        "fitness_goal": "Muscle Gain",
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_valid_input_has_no_errors(valid_input: dict) -> None:
    assert validate_profile_inputs(valid_input) == {}


def test_empty_input_reports_every_problem() -> None:
    errors = validate_profile_inputs({})
    assert set(errors) == {
        "name",
        "age",
        "gender",
        "height_cm",
        "weight_kg",
        "activity_level",
        "diet_preference",
        "fitness_goal",
    }


@pytest.mark.parametrize(
    ("field", "value", "expected_fragment"),
    [
        ("name", "", "enter your name"),
        ("name", "A1", "only contain letters"),
        ("name", "x" * 61, "characters or fewer"),
        ("age", 5, "Age must be between"),
        ("age", 120, "Age must be between"),
        ("age", "abc", "enter your age"),
        ("height_cm", 40, "Height must be between"),
        ("weight_kg", 500, "Weight must be between"),
        ("gender", "Alien", "select a gender"),
        ("activity_level", "Sometimes", "select your activity level"),
        ("diet_preference", "Keto-ish", "select a diet preference"),
        ("fitness_goal", "Get Ripped", "select a fitness goal"),
    ],
)
def test_invalid_values_are_rejected(valid_input: dict, field: str, value: object, expected_fragment: str) -> None:
    payload = dict(valid_input)
    payload[field] = value
    errors = validate_profile_inputs(payload)
    assert field in errors
    assert expected_fragment.lower() in errors[field].lower()


def test_build_profile_strips_name_and_casts_types(valid_input: dict) -> None:
    payload = dict(valid_input)
    payload["name"] = "  Aarav Sharma  "
    payload["height_cm"] = "175"
    payload["weight_kg"] = "72.5"
    profile = build_profile(payload)
    assert profile.name == "Aarav Sharma"
    assert isinstance(profile.height_cm, float)
    assert isinstance(profile.age, int)
    assert profile.weight_kg == 72.5


def test_build_profile_raises_on_invalid_input(valid_input: dict) -> None:
    payload = dict(valid_input)
    payload["age"] = 4
    with pytest.raises(ValueError):
        build_profile(payload)


# --------------------------------------------------------------------------- #
# BMI
# --------------------------------------------------------------------------- #
def test_bmi_known_value() -> None:
    # 70 kg at 175 cm -> 22.9 BMI
    assert bmi_of(70.0, 175.0) == pytest.approx(22.9, abs=0.05)


def test_bmi_zero_height_raises() -> None:
    with pytest.raises(ValueError):
        bmi_of(70.0, 0)


@pytest.mark.parametrize(
    ("bmi", "expected"),
    [
        (17.0, "Underweight"),
        (18.5, "Normal"),
        (22.9, "Normal"),
        (23.0, "Overweight"),
        (24.9, "Overweight"),
        (25.0, "Obese"),
        (31.0, "Obese"),
    ],
)
def test_bmi_category_bands(bmi: float, expected: str) -> None:
    assert bmi_category_of(bmi) == expected


# --------------------------------------------------------------------------- #
# Energy maths
# --------------------------------------------------------------------------- #
def test_bmr_male_mifflin_st_jeor() -> None:
    # 10*72 + 6.25*175 - 5*28 + 5 = 1678.75 -> 1679
    assert bmr_of(28, "Male", 175.0, 72.0) == pytest.approx(1679, abs=1)


def test_bmr_female_mifflin_st_jeor() -> None:
    # 10*64.5 + 6.25*163 - 5*28 - 161 = 1362.75 -> 1363
    assert bmr_of(28, "Female", 163.0, 64.5) == pytest.approx(1363, abs=1)


def test_tdee_uses_activity_multiplier() -> None:
    assert tdee_of(1600, "Sedentary (little or no exercise)") == pytest.approx(1920, abs=1)
    assert tdee_of(
        1600, "Very active (hard exercise 6-7 days/week)"
    ) == pytest.approx(1600 * 1.725, abs=1)


def test_tdee_unknown_activity_falls_back_to_sedentary() -> None:
    assert tdee_of(1600, "Does not exist") == pytest.approx(1600 * 1.2, abs=1)


@pytest.mark.parametrize("goal", ["Weight Loss", "Muscle Gain", "Maintain Weight"])
def test_calorie_target_applies_goal_factor(goal: str) -> None:
    tdee = 2400.0
    assert calorie_target_of(tdee, goal, "Male") == pytest.approx(
        round(tdee * GOAL_CALORIE_FACTOR[goal], 0), abs=1
    )


def test_calorie_target_never_goes_below_safety_floor() -> None:
    # A very low TDEE combined with a deficit must still respect the floor.
    target = calorie_target_of(1300.0, "Weight Loss", "Female")
    assert target >= MIN_SAFE_CALORIES["Female"]


# --------------------------------------------------------------------------- #
# Profile object
# --------------------------------------------------------------------------- #
def test_profile_derived_values_are_consistent(valid_input: dict) -> None:
    profile = build_profile(valid_input)
    assert profile.bmr == bmr_of(profile.age, profile.gender, profile.height_cm, profile.weight_kg)
    assert profile.maintenance_calories == tdee_of(profile.bmr, profile.activity_level)
    assert profile.daily_calorie_target == calorie_target_of(
        profile.maintenance_calories, profile.fitness_goal, profile.gender
    )
    assert profile.bmi == bmi_of(profile.weight_kg, profile.height_cm)
    assert profile.bmi_category in {"Underweight", "Normal", "Overweight", "Obese"}


def test_profile_roundtrip_dict(valid_input: dict) -> None:
    profile = build_profile(valid_input)
    restored = UserProfile.from_dict(profile.to_dict())
    assert restored == profile


def test_summary_rows_cover_every_field(valid_input: dict) -> None:
    rows = build_profile(valid_input).summary_rows()
    assert len(rows) == 8
    assert all(isinstance(label, str) and isinstance(value, str) for label, value in rows)
