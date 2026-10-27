"""Phase 4 tests: dataset integrity, matching rules, portion scaling, daily context.

The dataset is validated here as data (no negative values, energy balance, unique
ids), and the matching rules are tested against the documented order:
exact → alias → model label → token → fuzzy → no match.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from config import NUTRITION_CSV, PORTION_GRAMS_RANGE
from utils.helpers import build_profile
from nutrition.nutrition_engine import (
    CONFIDENT_SCORE,
    FUZZY_THRESHOLD,
    NutritionDataError,
    NutritionEngine,
    normalise,
    singularise,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine() -> NutritionEngine:
    return NutritionEngine()


@pytest.fixture(scope="module")
def sample_profile():
    return build_profile(
        {
            "name": "Riya Verma",
            "age": 31,
            "gender": "Female",
            "height_cm": 160.0,
            "weight_kg": 61.0,
            "activity_level": "Lightly active (light exercise 1-3 days/week)",
            "diet_preference": "Vegetarian",
            "fitness_goal": "Weight Loss",
        }
    )


# --------------------------------------------------------------------------- #
# Dataset integrity
# --------------------------------------------------------------------------- #
def test_csv_exists_and_has_expected_columns() -> None:
    assert NUTRITION_CSV.is_file(), "the nutrition dataset must ship with the project"
    with NUTRITION_CSV.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in NutritionEngine.REQUIRED_COLUMNS:
        assert column in header, f"missing column {column}"


def test_dataset_loads_with_a_useful_number_of_foods(engine: NutritionEngine) -> None:
    assert len(engine.foods) >= 100, "the dataset must cover a useful range of Indian foods"
    assert len(engine.categories) >= 8


def test_core_indian_foods_are_present(engine: NutritionEngine) -> None:
    expected = [
        "rice", "dal", "chapati", "paneer", "idli", "dosa", "poha", "upma", "khichdi",
        "samosa", "rajma", "chole", "curd", "gulab jamun", "banana", "mango", "tea",
    ]
    missing = [term for term in expected if not engine.search(term, limit=1)]
    assert not missing, f"dataset is missing common foods: {missing}"


def test_no_duplicate_ids_or_names(engine: NutritionEngine) -> None:
    ids = [item.food_id for item in engine.foods]
    names = [item.name for item in engine.foods]
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))


def test_every_row_uses_valid_numeric_values(engine: NutritionEngine) -> None:
    for item in engine.foods:
        for key in ("calories", "protein", "fat", "carbs", "fiber", "serving_grams"):
            value = getattr(item, key)
            assert value is not None, f"{item.name}: {key} must be numeric"
            assert value >= 0, f"{item.name}: {key} must not be negative"
        assert item.serving_grams > 0, f"{item.name}: serving weight must be positive"
        assert item.calories > 0, f"{item.name}: calories per 100 g must be positive"
        if item.sugar is not None:
            assert 0 <= item.sugar <= item.carbs + 0.05, f"{item.name}: sugar cannot exceed carbs"
        assert item.fiber <= item.carbs + 0.05, f"{item.name}: fibre cannot exceed carbs"


def test_energy_balance_is_plausible(engine: NutritionEngine) -> None:
    """Fibre-corrected Atwater energy must be within 15 % of the stated calories.

    4 kcal/g for protein and available carbohydrate, 2 kcal/g for fibre and
    9 kcal/g for fat (the standard correction: Atwater over-estimates low-energy
    plant foods when fibre is counted at 4 kcal/g).
    """
    offenders = []
    for item in engine.foods:
        estimated = 4 * (item.carbs - item.fiber) + 2 * item.fiber + 4 * item.protein + 9 * item.fat
        error = abs(estimated - item.calories) / item.calories
        if error > 0.15:
            offenders.append(f"{item.name}: csv {item.calories:.0f} kcal vs calculated {estimated:.0f}")
    assert not offenders, "energy-balance outliers:\n" + "\n".join(offenders)


def test_diet_types_are_from_the_allowed_set(engine: NutritionEngine) -> None:
    allowed = {"vegan", "vegetarian", "eggetarian", "non_vegetarian"}
    assert {item.diet_type for item in engine.foods} <= allowed


def test_broken_csv_raises_a_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "food_nutrition.csv"
    broken.write_text("food_id,food_name\n1,Not a real row\n")
    with pytest.raises(NutritionDataError) as excinfo:
        NutritionEngine(broken)
    assert "missing required column" in str(excinfo.value)

    with pytest.raises(NutritionDataError):
        NutritionEngine(tmp_path / "does_not_exist.csv")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Chicken Curry", "chicken curry"),
        ("Chicken_curry", "chicken curry"),
        ("Chapati (whole wheat no ghee)", "chapati"),
        ("Dal, toor!", "dal toor"),
        ("  TEA   WITH   MILK  ", "tea milk"),
        ("", ""),
    ],
)
def test_normalise(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_singularise_handles_plurals() -> None:
    assert singularise("chicken wings") == "chicken wing"
    assert singularise("dumplings rolls") == "dumpling roll"
    assert singularise("glass") == "glass"      # 'ss' ending is preserved


# --------------------------------------------------------------------------- #
# Matching rules
# --------------------------------------------------------------------------- #
def test_rule_1_exact_name_match(engine: NutritionEngine) -> None:
    result = engine.match("Dal toor (cooked)")
    assert result.found and result.method == "exact"
    assert result.score == 100.0
    assert result.needs_confirmation is False


def test_rule_2_alias_match(engine: NutritionEngine) -> None:
    result = engine.match("phulka")
    assert result.found and result.method == "alias"
    assert "Chapati" in result.food.name
    assert result.needs_confirmation is False


def test_rule_3_model_label_match(engine: NutritionEngine) -> None:
    """Food-101 class names must land on a curated dataset row."""
    for label, expected_fragment in (
        ("samosa", "Samosa"),
        ("chicken_curry", "Chicken curry"),
        ("pizza", "Pizza"),
        ("french_fries", "French fries"),
        ("ice_cream", "Ice cream"),
    ):
        result = engine.match(label)
        assert result.found, f"{label} did not match anything"
        assert result.method in {"exact", "alias", "model"}
        assert expected_fragment.lower() in result.food.name.lower()
        assert result.needs_confirmation is False


def test_exact_alias_needs_no_confirmation(engine: NutritionEngine) -> None:
    result = engine.match("banana")
    assert result.found
    assert result.needs_confirmation is False


def test_rule_4_token_match_for_partial_name(engine: NutritionEngine) -> None:
    result = engine.match("paneer masala")
    assert result.found
    assert result.method in {"token", "fuzzy"}
    assert result.needs_confirmation is True, "partial names must be confirmed by the user"


def test_rule_5_fuzzy_match_is_flagged(engine: NutritionEngine) -> None:
    result = engine.match("chapathi rolls")     # close to real names, not exact
    assert result.found
    assert result.needs_confirmation is True
    assert result.score < CONFIDENT_SCORE
    assert result.score >= FUZZY_THRESHOLD * 100.0


def test_unmatched_label_returns_no_food_and_suggestions(engine: NutritionEngine) -> None:
    result = engine.match("unobtainium soufflé")
    assert result.food is None
    assert result.method == "none"
    assert result.needs_confirmation is True
    assert result.alternatives, "the UI needs suggestions even when nothing matches"
    assert "will not guess" in result.message


def test_empty_label_is_handled(engine: NutritionEngine) -> None:
    result = engine.match("")
    assert result.food is None and result.method == "none"


def test_match_never_invents_nutrition(engine: NutritionEngine) -> None:
    """An unmatched label must not return a random food or fabricated values."""
    result = engine.match("zzzz qqqq")
    assert result.food is None
    assert result.alternatives  # suggestions only — the caller must ask the user
    for alternative, _score in result.alternatives:
        assert alternative.food_id is not None


# --------------------------------------------------------------------------- #
# Search & listing
# --------------------------------------------------------------------------- #
def test_search_ranks_relevant_foods_first(engine: NutritionEngine) -> None:
    results = engine.search("paneer", limit=5)
    assert results
    assert all("Paneer" in item.name or "paneer" in " ".join(item.aliases).lower() for item in results)


def test_list_foods_filters_by_category_and_diet(engine: NutritionEngine) -> None:
    vegetarian = engine.list_foods(diet_preference="Vegetarian")
    assert vegetarian and all(item.diet_type in {"vegan", "vegetarian"} for item in vegetarian)

    vegan = engine.list_foods(diet_preference="Vegan")
    assert vegan and all(item.diet_type == "vegan" for item in vegan)

    cereals = engine.list_foods(category="Cereals & grains")
    assert cereals and all(item.category == "Cereals & grains" for item in cereals)


def test_diet_compatibility_reports_conflicts(engine: NutritionEngine) -> None:
    chicken = engine.match("grilled chicken").food
    assert chicken is not None
    compatible, explanation = chicken.matches_diet("Vegetarian")
    assert compatible is False and "non vegetarian" in explanation

    dal = engine.match("dal").food
    assert dal is not None
    assert dal.matches_diet("Vegan")[0] is True
    assert dal.matches_diet("Jain")[0] is True


# --------------------------------------------------------------------------- #
# Portion scaling
# --------------------------------------------------------------------------- #
def test_scaling_is_linear_from_per_100g(engine: NutritionEngine) -> None:
    food = engine.match("Dal toor (cooked)").food
    for portion, expected_calories in ((100, food.calories), (50, food.calories / 2), (300, food.calories * 3)):
        values = food.scaled(portion)
        assert values["calories"] == pytest.approx(expected_calories, abs=0.1)
        assert values["protein"] == pytest.approx(food.protein * portion / 100, abs=0.1)


def test_calculate_returns_breakdown_with_macro_split(engine: NutritionEngine, sample_profile) -> None:
    food = engine.match("Chapati (whole wheat no ghee)").food
    breakdown = engine.calculate(food, 80.0)

    assert breakdown.portion_grams == 80.0
    assert breakdown.calories == pytest.approx(food.calories * 0.8, abs=0.1)
    assert sum(breakdown.macro_energy_pct.values()) == pytest.approx(100.0, abs=0.5)
    assert "80 g" in breakdown.portion_label()

    context = breakdown.daily_context(sample_profile)
    labels = [row[0] for row in context]
    assert labels == ["Energy", "Protein", "Fibre"]
    for _label, value, target, percent in context:
        assert value >= 0 and target > 0
        assert percent == pytest.approx(value / target * 100, abs=0.2)


def test_calculate_rejects_out_of_range_portions(engine: NutritionEngine) -> None:
    food = engine.match("Steamed white rice").food
    low, high = PORTION_GRAMS_RANGE
    for bad in (0, -5, high + 1):
        with pytest.raises(ValueError):
            engine.calculate(food, bad)
    assert engine.calculate(food, low).portion_grams == low


def test_missing_nutrients_stay_missing_after_scaling(engine: NutritionEngine) -> None:
    """A food without sodium data must not gain invented sodium when scaled."""
    almonds = engine.match("almonds").food
    assert almonds is not None and almonds.sodium_mg is None
    assert almonds.scaled(30)["sugar"] is not None        # sugar is known for almonds
    sugar_unknown = next((item for item in engine.foods if item.sugar is None), None)
    if sugar_unknown is not None:
        assert sugar_unknown.scaled(150)["sugar"] is None


def test_portion_presets_are_sensible(engine: NutritionEngine) -> None:
    food = engine.match("Samosa (fried)").food
    presets = engine.portion_presets(food)
    assert presets
    for label, grams in presets:
        assert isinstance(label, str) and label
        assert PORTION_GRAMS_RANGE[0] <= grams <= PORTION_GRAMS_RANGE[1]


def test_get_returns_none_for_unknown_id(engine: NutritionEngine) -> None:
    assert engine.get(999_999) is None
    assert engine.get(engine.foods[0].food_id) is engine.foods[0]
