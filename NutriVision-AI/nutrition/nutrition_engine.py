"""Nutrition engine (Phase 4): CSV lookup, portion scaling, daily context.

This module contains **no machine learning**. It is a deterministic lookup over a
curated CSV of common Indian foods, plus arithmetic. That is deliberate: nutrition
values must be reproducible and explainable, unlike the recognition model.

Dataset
-------
``nutrition/food_nutrition.csv`` — 134 foods, values **per 100 g** of the edible
portion, sourced and rounded from published Indian and international food
composition tables (IFCT 2017 / USDA FoodData Central style values). Every row
carries a ``note`` explaining what the values assume. Values are *typical
averages*: home cooking varies enormously with oil, salt and portion, so the app
labels everything an estimate.

Per-100 g columns: calories_kcal, protein_g, fat_g, carbs_g, fiber_g, sugar_g
(blank where unknown), sodium_mg (blank where unknown).

Food name matching — the documented rules
----------------------------------------
The recognition model returns Food-101 class names (``samosa``,
``chicken_curry``, ``bruschetta``…). The CSV uses human names ("Samosa (fried)").
Matching therefore runs through these rules, in order, and every rule reports
its method + score so the UI can be explicit about how the food was found:

======  ==========  ==========================================================
Order   Method      Rule
======  ==========  ==========================================================
1       ``exact``   Normalised names identical (lower-case, punctuation and
                    bracketed text removed, whitespace collapsed).
2       ``alias``   Query equals one of the row's ``aliases`` (pipe-separated).
3       ``model``   Query equals one of the row's ``model_labels`` — this is the
                    curated Food-101 → CSV mapping, kept inside the CSV itself.
4       ``token``   All query tokens appear in the name/aliases and at least one
                    token is 4+ characters (guards against "egg" matching
                    everything); score 80.
5       ``fuzzy``   ``difflib.SequenceMatcher`` ratio ≥ 0.72 over name and
                    aliases; score = ratio × 100, capped at 79.
======  ==========  ==========================================================

Anything below :data:`CONFIDENT_SCORE` (90) — and anything matched by ``fuzzy``
or ``token`` — is returned with ``needs_confirmation=True`` so the interface asks
the user to verify the food instead of silently trusting the guess. If nothing
scores at all, ``food`` is ``None`` and the caller must let the user choose from
the full list; the engine never invents a value for an unmatched food.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from config import NUTRITION_CSV, PORTION_GRAMS_RANGE

logger = logging.getLogger(__name__)

#: Nutrients exposed by the engine, in display order.
NUTRIENT_KEYS: tuple[str, ...] = ("calories", "protein", "fat", "carbs", "fiber", "sugar")

NUTRIENT_LABELS: dict[str, str] = {
    "calories": "Energy",
    "protein": "Protein",
    "fat": "Fat",
    "carbs": "Carbohydrates",
    "fiber": "Fibre",
    "sugar": "Sugar",
}

NUTRIENT_UNITS: dict[str, str] = {
    "calories": "kcal",
    "protein": "g",
    "fat": "g",
    "carbs": "g",
    "fiber": "g",
    "sugar": "g",
}

#: Matching thresholds (documented above).
FUZZY_THRESHOLD: float = 0.72
CONFIDENT_SCORE: float = 90.0
TOKEN_SCORE: float = 80.0

#: Words ignored when normalising a food query.
STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "with", "and", "or", "some", "plain", "fresh", "dish", "food"}
)

_BRACKETS = re.compile(r"\([^)]*\)")
_NON_ALPHA = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


class NutritionDataError(RuntimeError):
    """Raised when the nutrition CSV is missing or malformed."""


# --------------------------------------------------------------------------- #
# Normalisation helper (public: the UI and tests use it too)
# --------------------------------------------------------------------------- #
def normalise(text: str) -> str:
    """Normalise a food string for comparison.

    Lower-cases, drops bracketed qualifiers and punctuation, collapses spaces and
    removes trailing plural "s" from each token (``wings`` → ``wing``).
    """
    value = str(text or "").lower().replace("_", " ")
    value = _BRACKETS.sub(" ", value)
    value = _NON_ALPHA.sub(" ", value)
    value = _WHITESPACE.sub(" ", value).strip()
    tokens = [token for token in value.split(" ") if token and token not in STOPWORDS]
    return " ".join(tokens)


def singularise(text: str) -> str:
    """Drop a trailing plural ``s`` from every token (>3 characters)."""
    tokens = []
    for token in text.split():
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise NutritionDataError(f"Column '{field_name}' must be numeric, got {value!r}") from exc


def _to_optional_float(value: Any) -> Optional[float]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _split_multi(value: Any) -> tuple[str, ...]:
    """Split a pipe-separated cell into a tuple of trimmed strings."""
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoodItem:
    """One row of the nutrition dataset (values per 100 g)."""

    food_id: int
    name: str
    aliases: tuple[str, ...]
    model_labels: tuple[str, ...]
    category: str
    diet_type: str
    serving_label: str
    serving_grams: float
    calories: float
    protein: float
    fat: float
    carbs: float
    fiber: float
    sugar: Optional[float]
    sodium_mg: Optional[float]
    tags: tuple[str, ...]
    note: str

    # -- helpers ----------------------------------------------------------- #
    def per_100g(self) -> dict[str, Optional[float]]:
        """All nutrient values per 100 g."""
        return {
            "calories": self.calories,
            "protein": self.protein,
            "fat": self.fat,
            "carbs": self.carbs,
            "fiber": self.fiber,
            "sugar": self.sugar,
        }

    def value(self, nutrient: str) -> Optional[float]:
        """Value of one nutrient per 100 g (``None`` when unknown)."""
        return self.per_100g().get(nutrient)

    def scaled(self, grams: float) -> dict[str, Optional[float]]:
        """Nutrient values for an arbitrary portion, scaled linearly per 100 g.

        Scaling is linear by design and documented in the UI: the model of this
        phase has no density or cooking-loss correction.
        """
        if grams <= 0:
            raise ValueError("Portion size must be greater than zero grams.")
        factor = grams / 100.0
        return {
            key: (None if value is None else round(value * factor, 1))
            for key, value in self.per_100g().items()
        }

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    @property
    def serving_equivalents(self) -> float:
        """How many standard servings a 100 g portion equals."""
        return round(100.0 / self.serving_grams, 2) if self.serving_grams else 0.0

    def energy_density(self) -> str:
        """Short descriptive label for calorie density per 100 g."""
        if self.calories < 60:
            return "very low"
        if self.calories < 150:
            return "low"
        if self.calories < 300:
            return "moderate"
        if self.calories < 500:
            return "high"
        return "very high"

    def matches_diet(self, diet_preference: str) -> tuple[bool, str]:
        """Check the food against a profile diet preference.

        Returns ``(is_compatible, explanation)``. Jain also excludes foods tagged
        ``jain_unsuitable`` (onion/garlic/root-vegetable bases).
        """
        order = ["vegan", "vegetarian", "eggetarian", "non_vegetarian"]
        allowed = {
            "Vegan": ("vegan",),
            "Vegetarian": ("vegan", "vegetarian"),
            "Eggetarian": ("vegan", "vegetarian", "eggetarian"),
            "Non-Vegetarian": tuple(order),
            "Jain": ("vegan", "vegetarian"),
        }.get(diet_preference, tuple(order))

        if self.diet_type not in allowed:
            return False, (
                f"This food is classified as {self.diet_type.replace('_', ' ')} but your profile "
                f"preference is {diet_preference}."
            )
        if diet_preference == "Jain" and self.has_tag("jain_unsuitable"):
            return False, "Jain preference excludes foods with onion, garlic or root vegetables."
        return True, ""


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching a recognised label to a dataset row."""

    food: Optional[FoodItem]
    method: str                 # exact | alias | model | token | fuzzy | none
    score: float
    needs_confirmation: bool
    alternatives: tuple[tuple[FoodItem, float], ...] = ()
    message: str = ""

    @property
    def found(self) -> bool:
        return self.food is not None

    @property
    def method_label(self) -> str:
        return {
            "exact": "Exact name match",
            "alias": "Known alternative name",
            "model": "Mapped from the model's Food-101 label",
            "token": "Partial word match",
            "fuzzy": "Approximate (fuzzy) match",
            "manual": "Chosen by you",
            "none": "No match found",
        }.get(self.method, self.method)


@dataclass(frozen=True)
class NutritionBreakdown:
    """Nutrition for one portion, plus context against daily targets."""

    food: FoodItem
    portion_grams: float
    values: dict[str, Optional[float]]           # per portion
    per_100g: dict[str, Optional[float]]
    macro_energy_pct: dict[str, float] = field(default_factory=dict)
    serving_equivalents: float = 0.0

    @property
    def calories(self) -> Optional[float]:
        return self.values.get("calories")

    def portion_label(self) -> str:
        """Human description of the portion, e.g. ``"150 g (≈ 1.0 × 1 katori)"``."""
        equivalents = self.portion_grams / self.food.serving_grams if self.food.serving_grams else 0.0
        if self.food.serving_label and equivalents:
            return f"{self.portion_grams:g} g (≈ {equivalents:.1f} × {self.food.serving_label})"
        return f"{self.portion_grams:g} g"

    def daily_context(self, profile) -> list[tuple[str, float, float, float]]:
        """Compare this portion with the profile's daily targets.

        Returns rows of ``(label, portion_value, daily_target, percent)``.
        Targets come from the Phase 1 profile (calorie, protein and fibre goals);
        fat and carbohydrate targets are not defined in this project, so they are
        reported as percent of energy instead of a fake target.
        """
        rows: list[tuple[str, float, float, float]] = []
        calorie_target = float(getattr(profile, "daily_calorie_target", 0) or 0)
        protein_target = float(getattr(profile, "protein_target_g", 0) or 0)
        fiber_target = float(getattr(profile, "fiber_target_g", 0) or 0)

        checks = (
            ("Energy", "calories", calorie_target, "kcal"),
            ("Protein", "protein", protein_target, "g"),
            ("Fibre", "fiber", fiber_target, "g"),
        )
        for label, key, target, _unit in checks:
            value = self.values.get(key)
            if value is None or target <= 0:
                continue
            rows.append((label, float(value), target, round(float(value) / target * 100.0, 1)))
        return rows


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class NutritionEngine:
    """Lookup and calculation over the Indian food nutrition CSV."""

    REQUIRED_COLUMNS = (
        "food_id",
        "food_name",
        "aliases",
        "model_labels",
        "category",
        "diet_type",
        "serving_label",
        "serving_grams",
        "calories_kcal",
        "protein_g",
        "fat_g",
        "carbs_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
        "tags",
        "note",
    )

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else Path(NUTRITION_CSV)
        self._foods: tuple[FoodItem, ...] = ()
        self._by_id: dict[int, FoodItem] = {}
        self._by_normalised_name: dict[str, FoodItem] = {}
        self._by_alias: dict[str, FoodItem] = {}
        self._by_model_label: dict[str, FoodItem] = {}
        self._load()

    # -- loading ----------------------------------------------------------- #
    def _load(self) -> None:
        if not self.csv_path.is_file():
            raise NutritionDataError(
                f"Nutrition dataset not found at {self.csv_path}. "
                "It should ship with the project in the nutrition/ folder."
            )

        foods: list[FoodItem] = []
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in self.REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise NutritionDataError(
                    f"{self.csv_path.name} is missing required column(s): {', '.join(missing)}"
                )

            for line_number, row in enumerate(reader, start=2):
                try:
                    item = FoodItem(
                        food_id=int(row["food_id"]),
                        name=row["food_name"].strip(),
                        aliases=_split_multi(row["aliases"]),
                        model_labels=_split_multi(row["model_labels"]),
                        category=row["category"].strip(),
                        diet_type=row["diet_type"].strip().lower(),
                        serving_label=row["serving_label"].strip(),
                        serving_grams=_to_float(row["serving_grams"], "serving_grams"),
                        calories=_to_float(row["calories_kcal"], "calories_kcal"),
                        protein=_to_float(row["protein_g"], "protein_g"),
                        fat=_to_float(row["fat_g"], "fat_g"),
                        carbs=_to_float(row["carbs_g"], "carbs_g"),
                        fiber=_to_float(row["fiber_g"], "fiber_g"),
                        sugar=_to_optional_float(row["sugar_g"]),
                        sodium_mg=_to_optional_float(row["sodium_mg"]),
                        tags=_split_multi(row["tags"]),
                        note=(row["note"] or "").strip(),
                    )
                except (NutritionDataError, ValueError, KeyError) as exc:
                    raise NutritionDataError(f"Invalid row {line_number} in {self.csv_path.name}: {exc}") from exc
                foods.append(item)

        if not foods:
            raise NutritionDataError(f"{self.csv_path.name} contains no food rows.")

        self._foods = tuple(foods)
        self._by_id = {item.food_id: item for item in foods}
        self._by_normalised_name = {}
        self._by_alias = {}
        self._by_model_label = {}
        for item in foods:
            key = normalise(item.name)
            if key:
                self._by_normalised_name.setdefault(key, item)
            for alias in item.aliases:
                alias_key = normalise(alias)
                if alias_key:
                    self._by_alias.setdefault(alias_key, item)
            for label in item.model_labels:
                label_key = normalise(label)
                if label_key:
                    self._by_model_label.setdefault(label_key, item)

        logger.info(
            "Nutrition dataset loaded | %d foods | %d aliases | %d model-label mappings",
            len(self._foods),
            len(self._by_alias),
            len(self._by_model_label),
        )

    # -- access ------------------------------------------------------------ #
    @property
    def foods(self) -> tuple[FoodItem, ...]:
        return self._foods

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({item.category for item in self._foods}))

    def get(self, food_id: int) -> Optional[FoodItem]:
        """Fetch a food by id (``None`` when unknown)."""
        return self._by_id.get(int(food_id))

    def list_foods(self, category: Optional[str] = None, diet_preference: Optional[str] = None) -> list[FoodItem]:
        """All foods, optionally filtered by category and diet compatibility."""
        items = [item for item in self._foods if category in (None, "", item.category)]
        if diet_preference:
            items = [item for item in items if item.matches_diet(diet_preference)[0]]
        return items

    def search(self, query: str, limit: int = 25) -> list[FoodItem]:
        """Rank foods by relevance to a free-text query (for the manual picker)."""
        needle = normalise(query)
        if not needle:
            return list(self._foods[:limit])

        scored: list[tuple[float, FoodItem]] = []
        for item in self._foods:
            haystacks = [normalise(item.name), *(normalise(a) for a in item.aliases)]
            best = 0.0
            for hay in haystacks:
                if not hay:
                    continue
                if hay == needle:
                    best = max(best, 100.0)
                elif needle in hay or hay in needle:
                    best = max(best, 88.0)
                else:
                    best = max(best, SequenceMatcher(None, needle, hay).ratio() * 100.0)
            if best >= 45.0:
                scored.append((best, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored[:limit]]

    # -- matching ---------------------------------------------------------- #
    def _score_against(self, query: str, item: FoodItem) -> float:
        """Best fuzzy ratio between the query and any of the item's names."""
        candidates = [normalise(item.name), *(normalise(a) for a in item.aliases)]
        best = 0.0
        for candidate in candidates:
            if not candidate:
                continue
            best = max(best, SequenceMatcher(None, query, candidate).ratio())
        return best * 100.0

    def match(self, label: str, top_k: int = 6) -> MatchResult:
        """Match a recognised food label to a dataset row.

        Applies the documented rules in order (exact → alias → model label →
        token → fuzzy) and returns a :class:`MatchResult` describing the method,
        the confidence and whether the UI must ask the user to confirm.
        """
        raw = str(label or "").strip()
        if not raw:
            return MatchResult(None, "none", 0.0, True, (), "No food label was provided.")

        query = normalise(raw)
        query_singular = singularise(query)

        # Rule 1 — exact normalised name.
        item = self._by_normalised_name.get(query) or self._by_normalised_name.get(query_singular)
        if item is not None:
            return self._result(item, "exact", 100.0, raw)

        # Rule 2 — declared alias.
        item = self._by_alias.get(query) or self._by_alias.get(query_singular)
        if item is not None:
            return self._result(item, "alias", 96.0, raw)

        # Rule 3 — curated Food-101 label mapping stored in the CSV.
        item = self._by_model_label.get(query) or self._by_model_label.get(query_singular)
        if item is not None:
            return self._result(item, "model", 92.0, raw)

        # Rule 4 — every query token appears in a name/alias.
        tokens = [token for token in query_singular.split() if token]
        if tokens and any(len(token) >= 4 for token in tokens):
            for candidate in self._foods:
                haystacks = " ".join(
                    [singularise(normalise(candidate.name)), *(singularise(normalise(a)) for a in candidate.aliases)]
                )
                if all(token in haystacks for token in tokens):
                    return self._result(candidate, "token", TOKEN_SCORE, raw)

        # Rule 5 — fuzzy ratio.
        scored = sorted(
            ((self._score_against(query_singular, candidate), candidate) for candidate in self._foods),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_score, best_item = scored[0] if scored else (0.0, None)
        alternatives = tuple((candidate, round(score, 1)) for score, candidate in scored[:top_k])

        if best_item is not None and best_score >= FUZZY_THRESHOLD * 100.0:
            capped = min(best_score, 89.0)
            return MatchResult(
                best_item,
                "fuzzy",
                round(capped, 1),
                True,
                alternatives,
                f"Closest name found was '{best_item.name}' ({best_score:.0f}% text similarity). "
                "Please confirm before trusting the nutrition values.",
            )

        # No match — never invent values.
        return MatchResult(
            None,
            "none",
            round(best_score, 1),
            True,
            alternatives,
            f"'{raw}' is not in the Indian food dataset. Pick the closest food from the list below; "
            "the nutrition engine will not guess.",
        )

    def _result(self, item: FoodItem, method: str, score: float, raw: str) -> MatchResult:
        needs_confirmation = score < CONFIDENT_SCORE
        message = {
            "exact": f"Matched '{raw}' exactly to {item.name}.",
            "alias": f"'{raw}' is a known alternative name for {item.name}.",
            "model": f"The model's label '{raw}' is mapped to {item.name}.",
        }.get(method, "")
        alternatives = tuple(
            (candidate, round(self._score_against(normalise(raw), candidate), 1))
            for candidate in self._top_alternatives(item)
        )
        return MatchResult(item, method, score, needs_confirmation, alternatives, message)

    def _top_alternatives(self, item: FoodItem, count: int = 5) -> list[FoodItem]:
        """Similar foods from the same category, for the confirmation shortlist."""
        same_category = [
            candidate
            for candidate in self._foods
            if candidate.category == item.category and candidate.food_id != item.food_id
        ]
        return same_category[:count]

    # -- calculation ------------------------------------------------------- #
    def calculate(self, food: FoodItem, portion_grams: float) -> NutritionBreakdown:
        """Scale a food to a portion size and add macro-split context.

        Args:
            food: dataset row (values per 100 g).
            portion_grams: portion weight in grams, inside
                :data:`config.PORTION_GRAMS_RANGE`.

        Raises:
            ValueError: when the portion is outside the supported range.
        """
        low, high = PORTION_GRAMS_RANGE
        if not (low <= float(portion_grams) <= high):
            raise ValueError(
                f"Portion size must be between {low:g} g and {high:g} g (got {portion_grams})."
            )

        values = food.scaled(float(portion_grams))
        macro_pct = self.macro_energy_percent(values)
        return NutritionBreakdown(
            food=food,
            portion_grams=float(portion_grams),
            values=values,
            per_100g=food.per_100g(),
            macro_energy_pct=macro_pct,
            serving_equivalents=round(float(portion_grams) / food.serving_grams, 2) if food.serving_grams else 0.0,
        )

    @staticmethod
    def macro_energy_percent(values: dict[str, Optional[float]]) -> dict[str, float]:
        """Share of energy from protein / carbohydrate / fat (percent).

        Uses 4 kcal/g for protein and carbohydrate and 9 kcal/g for fat; the
        three percentages plus the rounding residual may not sum to exactly 100.
        """
        protein = float(values.get("protein") or 0.0)
        carbs = float(values.get("carbs") or 0.0)
        fat = float(values.get("fat") or 0.0)
        energy = 4 * protein + 4 * carbs + 9 * fat
        if energy <= 0:
            return {"protein": 0.0, "carbs": 0.0, "fat": 0.0}
        return {
            "protein": round(4 * protein / energy * 100.0, 1),
            "carbs": round(4 * carbs / energy * 100.0, 1),
            "fat": round(9 * fat / energy * 100.0, 1),
        }

    def portion_presets(self, food: FoodItem) -> list[tuple[str, float]]:
        """Useful portion buttons for the UI: house serving, half, double, 100 g."""
        presets: list[tuple[str, float]] = []
        low, high = PORTION_GRAMS_RANGE

        def add(label: str, grams: float) -> None:
            grams = float(round(grams))
            if low <= grams <= high and all(abs(grams - existing) > 0.5 for _label, existing in presets):
                presets.append((label, grams))

        if food.serving_label:
            add(food.serving_label, food.serving_grams)
            add(f"2 × {food.serving_label}", food.serving_grams * 2)
        add("100 g", 100.0)
        add("50 g", 50.0)
        return presets


# --------------------------------------------------------------------------- #
# Cached accessor
# --------------------------------------------------------------------------- #
def _cached_engine_factory():
    """Cache the parsed dataset once per process (Streamlit re-runs often)."""
    try:
        import streamlit as st

        if st.runtime.exists():
            return st.cache_resource(show_spinner=False)(_build_engine)
    except Exception:  # pragma: no cover
        pass
    return _build_engine


def _build_engine(csv_path: Optional[str] = None) -> NutritionEngine:
    return NutritionEngine(Path(csv_path) if csv_path else None)


get_nutrition_engine = _cached_engine_factory()
