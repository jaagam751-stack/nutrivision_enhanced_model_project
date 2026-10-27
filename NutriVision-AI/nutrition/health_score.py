"""Rule-based Nutrition Health Score (Phase 4).

The score is **0–100 and fully transparent**: every point is produced by a
documented rule, and the app shows the breakdown (rule → points → explanation)
next to the number. There is deliberately **no machine-learning model** here —
a score nobody can explain is worse than no score.

The score describes the *composition of 100 g of the food*, not how much you ate
(portion size is handled by the nutrition engine and the daily-target context).

Rubric (base 100 points)
------------------------
========================  =====  =================================================
Component                 Max    Rule
========================  =====  =================================================
Protein density           22     Protein's share of energy (``4·P / kcal``): linear
                                 0 → 22 points from 0 % to 25 % of energy.
Fibre density             22     Fibre per 100 kcal: 0 → 22 points from 0 to 2.5 g
                                 per 100 kcal (2.5 g is the WHO-style "high fibre"
                                 density target).
Sugar load                16     Sugar's share of energy (``4·sugar / kcal``):
                                 16 points at ≤ 5 % of energy, falling linearly
                                 to 0 points at ≥ 30 %. **Exemption:** foods tagged
                                 ``natural_sugar`` (whole fruit, plain dairy, coconut
                                 water) keep full marks, because this dataset cannot
                                 separate naturally occurring from added sugar.
Fat balance               16     Piecewise linear "not too little, not too much":
                                 4 points at 0 % fat energy, rising to 16 points
                                 in the 15–30 % band, falling to 0 at ≥ 60 %.
                                 **Guards:** a food with exactly 0 g fat scores the
                                 low plateau (4 pts), while a low-energy food
                                 (≤ 200 kcal) containing 0–5 g fat per 100 g keeps
                                 full marks — the taper targets genuinely high-fat
                                 foods, not a spoonful of curd.
Sodium                    12     12 points at ≤ 120 mg per 100 g, falling
                                 linearly to 0 at ≥ 800 mg per 100 g.
========================  =====  =================================================

Integrity adjustments (capped at +8 and −12, applied after the base score)
-------------------------------------------------------------------------
``raw`` +6 · ``fruit`` +6 · ``vegetable`` +5 · ``legume`` +4 · ``leafy`` +4 ·
``high_fibre`` +3 · ``high_protein`` +3 · ``wholegrain`` +2 · ``fried`` −8 ·
``sweet`` −5 · ``processed`` −4 · ``high_sodium`` −6 · ``high_sugar`` −4 ·
``refined`` −2

Final score = clamp(base + adjustments, 0, 100), rounded to one decimal.

Missing data
------------
Sodium is unavailable for some foods. Rather than guessing it, the sodium
component is dropped (base becomes 88) and the **final score is rescaled by
100/88** so foods are still comparable; ``sodium_known`` is ``False`` and the UI
says so. Sugar is treated the same way when unknown (the component is dropped
and a note is added).

Bands
-----
78–100 Excellent · 62–77.9 Good · 45–61.9 Fair · 32–44.9 Poor · 0–31.9 Very poor

Honest limitations
------------------
* Sugar in this dataset is **total sugar** — the tables do not separate naturally
  occurring (fruit, milk) from added sugar, so fruit is scored slightly harshly.
* Fat quality (saturated/trans) is not in the data, so "fat balance" only looks at
  quantity, not type.
* Whole-grain vs refined distinction is partial, and cooking style (oil, salt,
  portion) is not captured beyond the tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Rubric constants (kept here so the rubric is auditable in one place)
# --------------------------------------------------------------------------- #
PROTEIN_MAX: float = 22.0
PROTEIN_ENERGY_FULL: float = 25.0        # % of energy for full marks

FIBER_MAX: float = 22.0
FIBER_PER_100_KCAL_FULL: float = 2.5     # g per 100 kcal for full marks

SUGAR_MAX: float = 16.0
SUGAR_ENERGY_BEST: float = 5.0           # % of energy at/below which full marks
SUGAR_ENERGY_WORST: float = 30.0         # % of energy at/above which zero marks

FAT_MAX: float = 16.0
FAT_ENERGY_BAND: tuple[float, float] = (15.0, 30.0)
FAT_ENERGY_MIN_POINTS: float = 4.0
FAT_ENERGY_ZERO: float = 60.0            # % of energy at/above which zero marks
FAT_LOW_ABSOLUTE_G: float = 5.0          # g/100 g below which the taper is skipped
FAT_LOW_ENERGY_MAX_KCAL: float = 200.0   # ...but only for low-energy-density foods

SODIUM_MAX: float = 12.0
SODIUM_BEST_MG: float = 120.0            # per 100 g
SODIUM_WORST_MG: float = 800.0           # per 100 g

# Base maxima: 22 + 22 + 16 + 16 + 12 = 88 points for a fully known food.
BASE_WITH_SODIUM: float = PROTEIN_MAX + FIBER_MAX + SUGAR_MAX + FAT_MAX + SODIUM_MAX
BASE_WITHOUT_SODIUM: float = PROTEIN_MAX + FIBER_MAX + SUGAR_MAX + FAT_MAX             # 76

TAG_ADJUSTMENTS: dict[str, float] = {
    "raw": 6.0,
    "fruit": 6.0,
    "vegetable": 5.0,
    "legume": 4.0,
    "leafy": 4.0,
    "high_fibre": 3.0,
    "high_protein": 3.0,
    "wholegrain": 2.0,
    "fried": -8.0,
    "sweet": -5.0,
    "processed": -4.0,
    "high_sodium": -6.0,
    "high_sugar": -4.0,
    "refined": -2.0,
}
ADJUSTMENT_CAP_POSITIVE: float = 8.0
ADJUSTMENT_CAP_NEGATIVE: float = -12.0

BANDS: tuple[tuple[float, str, str, str], ...] = (
    (78.0, "Excellent", "good", "A nutrient-dense choice you can lean on regularly."),
    (62.0, "Good", "good", "A solid everyday choice; watch the portion if it is energy dense."),
    (45.0, "Fair", "warn", "Fine occasionally — pair it with fibre and protein to balance the meal."),
    (32.0, "Poor", "bad", "Enjoy sparingly; it contributes little nutrient value per calorie."),
    (0.0, "Very poor", "bad", "Occasional treat territory — not an everyday food."),
)


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreComponent:
    """One line of the rubric: points earned, maximum, and the reasoning."""

    label: str
    points: float
    max_points: float
    explanation: str

    @property
    def fraction(self) -> float:
        return round(self.points / self.max_points, 3) if self.max_points else 0.0


@dataclass(frozen=True)
class HealthScore:
    """The full, explainable result."""

    total: float
    band: str
    tone: str
    headline: str
    components: tuple[ScoreComponent, ...]
    adjustments: tuple[ScoreComponent, ...]
    sodium_known: bool
    sugar_known: bool
    rescaled: bool
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[tuple[str, str, str]]:
        """``(component, points, why)`` rows for the UI table."""
        rows = []
        for component in self.components:
            rows.append(
                (
                    component.label,
                    f"{component.points:.1f} / {component.max_points:.0f}",
                    component.explanation,
                )
            )
        for adjustment in self.adjustments:
            sign = "+" if adjustment.points >= 0 else ""
            rows.append(
                (
                    adjustment.label,
                    f"{sign}{adjustment.points:.0f}",
                    adjustment.explanation,
                )
            )
        return rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _piecewise_linear(value: float, best: float, worst: float, max_points: float) -> float:
    """Linear score between ``best`` (full marks) and ``worst`` (zero marks)."""
    if abs(worst - best) < 1e-9:
        return max_points
    ratio = 1.0 - (value - best) / (worst - best)
    return _clamp(ratio, 0.0, 1.0) * max_points


def _fat_balance_points(fat_energy_pct: float) -> float:
    """Full marks in the 15–30 % energy band; tapering outside it (documented)."""
    low, high = FAT_ENERGY_BAND
    if low <= fat_energy_pct <= high:
        return FAT_MAX
    if fat_energy_pct < low:
        # Very low fat is nutritionally neither good nor bad: 4 → 16 points.
        ratio = fat_energy_pct / low if low else 1.0
        return _clamp(FAT_ENERGY_MIN_POINTS + (FAT_MAX - FAT_ENERGY_MIN_POINTS) * ratio, 0.0, FAT_MAX)
    return _clamp(
        FAT_MAX * (FAT_ENERGY_ZERO - fat_energy_pct) / (FAT_ENERGY_ZERO - high), 0.0, FAT_MAX
    )


def band_for(total: float) -> tuple[str, str, str]:
    """Return ``(band, tone, headline)`` for a total score."""
    for threshold, band, tone, headline in BANDS:
        if total >= threshold:
            return band, tone, headline
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def score_food(
    *,
    calories: Optional[float],
    protein: Optional[float],
    fat: Optional[float],
    carbs: Optional[float],
    fiber: Optional[float],
    sugar: Optional[float],
    sodium_mg: Optional[float],
    tags: Iterable[str] = (),
) -> HealthScore:
    """Compute the transparent 0–100 Nutrition Health Score for **100 g** of food.

    All nutrient arguments are per 100 g and may be ``None`` when unknown.
    Raises ``ValueError`` when calories are missing or non-positive, because every
    density rule divides by energy.
    """
    if calories is None or calories <= 0:
        raise ValueError("Health score needs a positive calorie value per 100 g.")

    kcal = float(calories)
    protein = float(protein or 0.0)
    fat = float(fat or 0.0)
    carbs = float(carbs or 0.0)
    fiber = float(fiber or 0.0)

    components: list[ScoreComponent] = []

    # 1. Protein density ---------------------------------------------------- #
    protein_energy = 4.0 * protein / kcal * 100.0
    protein_points = _clamp(protein_energy / PROTEIN_ENERGY_FULL, 0.0, 1.0) * PROTEIN_MAX
    components.append(
        ScoreComponent(
            "Protein density",
            round(protein_points, 1),
            PROTEIN_MAX,
            f"{protein:.1f} g protein = {protein_energy:.0f}% of energy "
            f"(full marks at {PROTEIN_ENERGY_FULL:.0f}%).",
        )
    )

    # 2. Fibre density ------------------------------------------------------ #
    fiber_per_100kcal = fiber / kcal * 100.0
    fiber_points = _clamp(fiber_per_100kcal / FIBER_PER_100_KCAL_FULL, 0.0, 1.0) * FIBER_MAX
    components.append(
        ScoreComponent(
            "Fibre density",
            round(fiber_points, 1),
            FIBER_MAX,
            f"{fiber:.1f} g fibre per 100 g = {fiber_per_100kcal:.2f} g per 100 kcal "
            f"(full marks at {FIBER_PER_100_KCAL_FULL:.1f}).",
        )
    )

    # 3. Sugar load --------------------------------------------------------- #
    sugar_known = sugar is not None
    natural_sugar = "natural_sugar" in {str(tag).strip().lower() for tag in tags}
    if sugar_known:
        sugar_energy = 4.0 * float(sugar) / kcal * 100.0
        if natural_sugar:
            # Documented exemption: the dataset cannot separate intrinsic from
            # added sugar, so whole fruit / plain dairy / coconut water keep the
            # sugar points and the rule is stated in the breakdown.
            sugar_points = SUGAR_MAX
            explanation = (
                f"{float(sugar):.1f} g sugar = {sugar_energy:.0f}% of energy, but this food is "
                "tagged as containing naturally occurring sugar (whole fruit or plain dairy), "
                "so the added-sugar penalty does not apply."
            )
        else:
            sugar_points = _clamp(
                (SUGAR_ENERGY_WORST - sugar_energy) / (SUGAR_ENERGY_WORST - SUGAR_ENERGY_BEST), 0.0, 1.0
            ) * SUGAR_MAX
            explanation = (
                f"{float(sugar):.1f} g sugar = {sugar_energy:.0f}% of energy "
                f"(full marks at ≤ {SUGAR_ENERGY_BEST:.0f}%, zero at ≥ {SUGAR_ENERGY_WORST:.0f}%)."
            )
        components.append(ScoreComponent("Sugar load", round(sugar_points, 1), SUGAR_MAX, explanation))

    # 4. Fat balance -------------------------------------------------------- #
    fat_energy = 9.0 * fat / kcal * 100.0
    low_fat_guard = 0 < fat <= FAT_LOW_ABSOLUTE_G and kcal <= FAT_LOW_ENERGY_MAX_KCAL
    fat_points = FAT_MAX if low_fat_guard else _fat_balance_points(fat_energy)
    fat_explanation = (
        f"{fat:.1f} g fat per 100 g is a small absolute amount in a low-energy food, so the "
        "fat taper is not applied."
        if low_fat_guard
        else f"{fat:.1f} g fat = {fat_energy:.0f}% of energy (best between "
        f"{FAT_ENERGY_BAND[0]:.0f}% and {FAT_ENERGY_BAND[1]:.0f}%)."
    )
    components.append(ScoreComponent("Fat balance", round(fat_points, 1), FAT_MAX, fat_explanation))

    # 5. Sodium ------------------------------------------------------------- #
    sodium_known = sodium_mg is not None
    if sodium_known:
        sodium = float(sodium_mg)
        sodium_points = _piecewise_linear(sodium, SODIUM_BEST_MG, SODIUM_WORST_MG, SODIUM_MAX)
        components.append(
            ScoreComponent(
                "Sodium",
                round(sodium_points, 1),
                SODIUM_MAX,
                f"{sodium:.0f} mg sodium per 100 g (full marks at ≤ {SODIUM_BEST_MG:.0f} mg, "
                f"zero at ≥ {SODIUM_WORST_MG:.0f} mg).",
            )
        )

    base_points = sum(component.points for component in components)
    base_max = sum(component.max_points for component in components)
    if base_max <= 0:  # pragma: no cover - defensive
        raise ValueError("No scoring components could be evaluated.")

    # Integrity adjustments ------------------------------------------------- #
    tag_list = [str(tag).strip().lower() for tag in tags]
    adjustments: list[ScoreComponent] = []
    raw_adjustment = 0.0
    for tag, delta in TAG_ADJUSTMENTS.items():
        if tag in tag_list:
            raw_adjustment += delta
            adjustments.append(
                ScoreComponent(
                    f"Food type: {tag.replace('_', ' ')}",
                    delta,
                    abs(delta),
                    f"Tag '{tag}' applies a {delta:+.0f} point adjustment.",
                )
            )
    capped_adjustment = _clamp(raw_adjustment, ADJUSTMENT_CAP_NEGATIVE, ADJUSTMENT_CAP_POSITIVE)
    if abs(capped_adjustment - raw_adjustment) > 1e-9:
        adjustments.append(
            ScoreComponent(
                "Adjustment cap",
                round(capped_adjustment - raw_adjustment, 1),
                0.0,
                f"Food-type adjustments were capped to {capped_adjustment:+.0f} points.",
            )
        )

    # Rescale when a component is missing so foods stay comparable ---------- #
    rescaled = False
    if base_max < BASE_WITH_SODIUM:
        scale = BASE_WITH_SODIUM / base_max
        base_points = base_points * scale
        rescaled = True

    total = _clamp(base_points + capped_adjustment, 0.0, 100.0)
    band, tone, headline = band_for(total)

    notes: list[str] = []
    if not sodium_known:
        notes.append(
            "Sodium is not available for this food, so that component was dropped and the "
            "macronutrient score was rescaled to keep scores comparable."
        )
    if not sugar_known:
        notes.append("Sugar is not available for this food, so the sugar component was omitted.")
    if {"sweet", "high_sugar"} & set(tag_list):
        notes.append(
            "Sugar here is total sugar; the dataset does not separate naturally occurring "
            "from added sugar, so sweet foods are penalised on both sugar and food-type."
        )

    return HealthScore(
        total=round(total, 1),
        band=band,
        tone=tone,
        headline=headline,
        components=tuple(components),
        adjustments=tuple(adjustments),
        sodium_known=sodium_known,
        sugar_known=sugar_known,
        rescaled=rescaled,
        notes=tuple(notes),
    )


def score_item(food: Any) -> HealthScore:
    """Convenience wrapper for a :class:`nutrition.nutrition_engine.FoodItem`."""
    return score_food(
        calories=getattr(food, "calories", None),
        protein=getattr(food, "protein", None),
        fat=getattr(food, "fat", None),
        carbs=getattr(food, "carbs", None),
        fiber=getattr(food, "fiber", None),
        sugar=getattr(food, "sugar", None),
        sodium_mg=getattr(food, "sodium_mg", None),
        tags=getattr(food, "tags", ()),
    )


def score_table(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Score many per-100 g records (used by tests and batch checks)."""
    totals = []
    for row in rows:
        totals.append(
            score_food(
                calories=row.get("calories"),
                protein=row.get("protein"),
                fat=row.get("fat"),
                carbs=row.get("carbs"),
                fiber=row.get("fiber"),
                sugar=row.get("sugar"),
                sodium_mg=row.get("sodium_mg"),
                tags=row.get("tags", ()),
            ).total
        )
    return totals
