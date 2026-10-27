"""Nutrition package (Phase 4).

No machine learning lives here: the nutrition engine is a documented CSV lookup
plus arithmetic, and the health score is a transparent point-based rubric.

Public surface used by the pages::

    from nutrition import get_nutrition_engine, score_item

    engine = get_nutrition_engine()
    match = engine.match("samosa")                     # MatchResult
    breakdown = engine.calculate(match.food, 120.0)    # NutritionBreakdown
    health = score_item(match.food)                    # HealthScore
"""

from nutrition.health_score import (
    BANDS,
    TAG_ADJUSTMENTS,
    HealthScore,
    ScoreComponent,
    band_for,
    score_food,
    score_item,
    score_table,
)
from nutrition.nutrition_engine import (
    CONFIDENT_SCORE,
    FUZZY_THRESHOLD,
    NUTRIENT_LABELS,
    NUTRIENT_UNITS,
    MatchResult,
    FoodItem,
    NutritionBreakdown,
    NutritionDataError,
    NutritionEngine,
    get_nutrition_engine,
    normalise,
)

__all__ = [
    "BANDS",
    "CONFIDENT_SCORE",
    "FUZZY_THRESHOLD",
    "NUTRIENT_LABELS",
    "NUTRIENT_UNITS",
    "TAG_ADJUSTMENTS",
    "FoodItem",
    "HealthScore",
    "MatchResult",
    "NutritionBreakdown",
    "NutritionDataError",
    "NutritionEngine",
    "ScoreComponent",
    "band_for",
    "get_nutrition_engine",
    "normalise",
    "score_food",
    "score_item",
    "score_table",
]
