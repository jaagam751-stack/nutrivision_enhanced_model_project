"""Phase 4 tests: the rule-based Nutrition Health Score.

These tests pin down the behaviour of the rubric rather than exact numbers for a
specific food (values change when the dataset is corrected). They check bounds,
monotonicity, the documented exemptions and the handling of missing data.
"""

from __future__ import annotations

import pytest

from nutrition import get_nutrition_engine
from nutrition.health_score import (
    ADJUSTMENT_CAP_NEGATIVE,
    ADJUSTMENT_CAP_POSITIVE,
    BANDS,
    BASE_WITH_SODIUM,
    TAG_ADJUSTMENTS,
    band_for,
    score_food,
    score_item,
    score_table,
)

# A neutral "reference" food: average macros, some fibre, moderate sodium.
REFERENCE = dict(
    calories=200.0,
    protein=10.0,
    fat=7.0,
    carbs=25.0,
    fiber=3.0,
    sugar=3.0,
    sodium_mg=250.0,
    tags=(),
)


def score(**overrides):
    payload = dict(REFERENCE)
    payload.update(overrides)
    return score_food(**payload)


@pytest.fixture(scope="module")
def engine():
    return get_nutrition_engine()


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #
def test_score_is_bounded_and_explainable() -> None:
    result = score()
    assert 0.0 <= result.total <= 100.0
    assert result.band in {band for _t, band, _tone, _h in BANDS}
    assert result.components, "every score must expose its components"
    for component in result.components:
        assert component.max_points > 0
        assert 0.0 <= component.points <= component.max_points
        assert component.explanation, "each rule must explain itself"


def test_components_are_exactly_the_documented_rubric() -> None:
    labels = [component.label for component in score().components]
    assert labels == ["Protein density", "Fibre density", "Sugar load", "Fat balance", "Sodium"]


def test_missing_calories_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_food(calories=None, protein=1, fat=1, carbs=1, fiber=0, sugar=None, sodium_mg=None)
    with pytest.raises(ValueError):
        score_food(calories=0, protein=1, fat=1, carbs=1, fiber=0, sugar=None, sodium_mg=None)


def test_scoring_is_deterministic() -> None:
    assert score().total == score().total
    assert score_item.__module__  # importable


# --------------------------------------------------------------------------- #
# Component behaviour
# --------------------------------------------------------------------------- #
def test_protein_density_increases_the_score() -> None:
    low = score(protein=2.0, fat=6.0, carbs=30.0)
    high = score(protein=20.0, fat=6.0, carbs=10.0)
    assert high.total > low.total
    assert any(c.label == "Protein density" and c.points > 15 for c in high.components)


def test_fibre_density_increases_the_score() -> None:
    low = score(fiber=0.3)
    high = score(fiber=8.0)
    assert high.total > low.total


def test_added_sugar_is_penalised() -> None:
    low = score(sugar=2.0)
    high = score(sugar=40.0, calories=400.0, carbs=60.0)
    assert high.total < low.total
    sugar_component = next(c for c in high.components if c.label == "Sugar load")
    assert sugar_component.points == pytest.approx(0.0, abs=0.1)


def test_natural_sugar_exemption_is_applied_and_explained() -> None:
    """Whole fruit keeps its sugar points because the data cannot split sugar types."""
    fruit = score(calories=52, protein=0.3, fat=0.2, carbs=13.8, fiber=2.4, sugar=10.4,
                  sodium_mg=1, tags=("raw", "fruit", "natural_sugar"))
    added = score(calories=52, protein=0.3, fat=0.2, carbs=13.8, fiber=2.4, sugar=10.4,
                  sodium_mg=1, tags=())
    sugar_component = next(c for c in fruit.components if c.label == "Sugar load")
    assert sugar_component.points == pytest.approx(16.0)
    assert "naturally occurring sugar" in sugar_component.explanation
    assert fruit.total > added.total


def test_high_fat_is_penalised_but_zero_fat_is_not_rewarded() -> None:
    fat_bomb = score(calories=900, protein=0, fat=100, carbs=0, fiber=0, sugar=0, sodium_mg=0)
    balanced = score(calories=200, protein=10, fat=7, carbs=25, fiber=3, sugar=3, sodium_mg=250)
    assert fat_bomb.total < balanced.total
    fat_component = next(c for c in fat_bomb.components if c.label == "Fat balance")
    assert fat_component.points == pytest.approx(0.0, abs=0.1)

    # A zero-fat, high-carb food sits on the low plateau, not at full marks.
    zero_fat = score(calories=380, protein=4, fat=0, carbs=90, fiber=1, sugar=2, sodium_mg=5)
    zero_fat_component = next(c for c in zero_fat.components if c.label == "Fat balance")
    assert zero_fat_component.points < 8.0


def test_low_fat_guard_for_low_energy_foods() -> None:
    """A spoonful of curd should not be punished for its fat energy share."""
    curd = score(calories=60, protein=3.1, fat=3.3, carbs=4.7, fiber=0, sugar=4.7,
                 sodium_mg=40, tags=("natural_sugar",))
    fat_component = next(c for c in curd.components if c.label == "Fat balance")
    assert fat_component.points == pytest.approx(16.0)
    assert "small absolute amount" in fat_component.explanation


def test_sodium_penalty_is_monotonic() -> None:
    clean = score(sodium_mg=10)
    salty = score(sodium_mg=1500)
    assert clean.total > salty.total
    salty_component = next(c for c in salty.components if c.label == "Sodium")
    assert salty_component.points == 0.0


# --------------------------------------------------------------------------- #
# Missing data handling
# --------------------------------------------------------------------------- #
def test_missing_sodium_is_rescaled_and_flagged() -> None:
    with_sodium = score(sodium_mg=120.0)
    without = score(sodium_mg=None)
    assert without.sodium_known is False
    assert without.rescaled is True
    assert not any(c.label == "Sodium" for c in without.components)
    assert without.notes and "Sodium is not available" in without.notes[0]
    # The rescaling keeps the two comparable (same food, only sodium differs).
    assert abs(with_sodium.total - without.total) < 8.0


def test_missing_sugar_drops_the_component() -> None:
    without = score(sugar=None)
    assert without.sugar_known is False
    assert not any(c.label == "Sugar load" for c in without.components)
    assert any("Sugar is not available" in note for note in without.notes)


def test_base_maximum_matches_the_rubric() -> None:
    assert BASE_WITH_SODIUM == pytest.approx(sum(c.max_points for c in score().components))


# --------------------------------------------------------------------------- #
# Food-type adjustments
# --------------------------------------------------------------------------- #
def test_fried_food_is_penalised() -> None:
    plain = score()
    fried = score(tags=("fried",))
    assert fried.total < plain.total
    assert any(a.label.startswith("Food type") for a in fried.adjustments)


def test_positive_adjustments_are_capped() -> None:
    boosted = score(tags=("raw", "fruit", "legume", "leafy", "high_fibre", "high_protein"))
    total_adjustment = sum(a.points for a in boosted.adjustments)
    assert total_adjustment <= ADJUSTMENT_CAP_POSITIVE + 1e-6


def test_negative_adjustments_are_capped() -> None:
    junk = score(tags=("fried", "sweet", "processed", "high_sodium", "high_sugar", "refined"))
    total_adjustment = sum(a.points for a in junk.adjustments)
    assert total_adjustment >= ADJUSTMENT_CAP_NEGATIVE - 1e-6


def test_every_declared_tag_has_an_adjustment_definition() -> None:
    assert set(TAG_ADJUSTMENTS) == {
        "raw", "fruit", "vegetable", "legume", "leafy", "high_fibre", "high_protein",
        "wholegrain", "fried", "sweet", "processed", "high_sodium", "high_sugar", "refined",
    }


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("total", "expected"),
    [(100.0, "Excellent"), (78.0, "Excellent"), (77.9, "Good"), (62.0, "Good"),
     (61.9, "Fair"), (45.0, "Fair"), (44.9, "Poor"), (32.0, "Poor"), (31.9, "Very poor"),
     (0.0, "Very poor")],
)
def test_band_boundaries(total: float, expected: str) -> None:
    assert band_for(total)[0] == expected


def test_bands_are_ordered_and_complete() -> None:
    thresholds = [threshold for threshold, *_ in BANDS]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[-1] == 0.0


# --------------------------------------------------------------------------- #
# Against the real dataset
# --------------------------------------------------------------------------- #
def test_real_foods_score_sensibly(engine) -> None:
    """Whole foods must outrank fried snacks and sweets."""
    dal = score_item(engine.match("dal").food)
    chapati = score_item(engine.match("chapati").food)
    sprouts = score_item(engine.match("sprouts").food)
    samosa = score_item(engine.match("samosa").food)
    jamun = score_item(engine.match("gulab jamun").food)
    cola = score_item(engine.match("cola").food)
    ghee = score_item(engine.match("ghee").food)

    assert dal.total > samosa.total
    assert chapati.total > samosa.total
    assert sprouts.total > jamun.total
    assert samosa.total > cola.total
    assert ghee.total < chapati.total
    assert jamun.band in {"Poor", "Very poor"}
    assert cola.band == "Very poor"


def test_every_dataset_food_can_be_scored(engine) -> None:
    totals = [score_item(item).total for item in engine.foods]
    assert len(totals) == len(engine.foods)
    assert all(0.0 <= value <= 100.0 for value in totals)
    # A useful spread: the dataset must not collapse to a single band.
    assert len({score_item(item).band for item in engine.foods}) >= 4


def test_score_table_helper(engine) -> None:
    rows = [
        {
            "calories": item.calories,
            "protein": item.protein,
            "fat": item.fat,
            "carbs": item.carbs,
            "fiber": item.fiber,
            "sugar": item.sugar,
            "sodium_mg": item.sodium_mg,
            "tags": item.tags,
        }
        for item in engine.foods[:5]
    ]
    totals = score_table(rows)
    assert len(totals) == 5
    assert totals == [score_item(item).total for item in engine.foods[:5]]
