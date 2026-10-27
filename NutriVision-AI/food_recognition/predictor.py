"""Food prediction and confidence handling (Phase 3).

``FoodPredictor`` wraps a loaded model and turns raw class probabilities into
something the UI can present honestly:

* a human-readable label (``chicken_curry`` → "Chicken curry"),
* a confidence percentage plus a confidence *tier*,
* the top-k alternatives, with the margin between the first and second choice,
* an explicit ``is_uncertain`` flag when the model is guessing. Phase 4 uses
  that flag to ask the user to confirm/choose the food before nutrition is
  looked up.

No fallback or hardcoded labels exist anywhere in this module: if the model is
not available, the loader raises and the caller shows the error.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, UNCERTAIN_MARGIN
from food_recognition.model_loader import (
    BaseFoodModel,
    ModelUnavailableError,
    load_food_model,
    load_labels,
    resolve_backend,
)
from food_recognition.preprocessing import preprocess

logger = logging.getLogger(__name__)

#: Nicer display names for labels where plain title-casing reads badly.
LABEL_DISPLAY_OVERRIDES: dict[str, str] = {
    "baby_back_ribs": "Baby back ribs",
    "beef_carpaccio": "Beef carpaccio",
    "beef_tartare": "Beef tartare",
    "chicken_curry": "Chicken curry",
    "chicken_quesadilla": "Chicken quesadilla",
    "chicken_wings": "Chicken wings",
    "chocolate_mousse": "Chocolate mousse",
    "club_sandwich": "Club sandwich",
    "creme_brulee": "Crème brûlée",
    "croque_madame": "Croque madame",
    "cup_cakes": "Cupcakes",
    "deviled_eggs": "Deviled eggs",
    "eggs_benedict": "Eggs benedict",
    "filet_mignon": "Filet mignon",
    "fish_and_chips": "Fish and chips",
    "french_fries": "French fries",
    "french_onion_soup": "French onion soup",
    "french_toast": "French toast",
    "fried_calamari": "Fried calamari",
    "frozen_yogurt": "Frozen yogurt",
    "garlic_bread": "Garlic bread",
    "grilled_cheese_sandwich": "Grilled cheese sandwich",
    "grilled_salmon": "Grilled salmon",
    "hot_and_sour_soup": "Hot and sour soup",
    "hot_dog": "Hot dog",
    "huevos_rancheros": "Huevos rancheros",
    "ice_cream": "Ice cream",
    "lobster_bisque": "Lobster bisque",
    "lobster_roll_sandwich": "Lobster roll sandwich",
    "macaroni_and_cheese": "Macaroni and cheese",
    "miso_soup": "Miso soup",
    "onion_rings": "Onion rings",
    "pad_thai": "Pad thai",
    "panna_cotta": "Panna cotta",
    "peking_duck": "Peking duck",
    "pork_chop": "Pork chop",
    "prime_rib": "Prime rib",
    "pulled_pork_sandwich": "Pulled pork sandwich",
    "red_velvet_cake": "Red velvet cake",
    "seaweed_salad": "Seaweed salad",
    "shrimp_and_grits": "Shrimp and grits",
    "spaghetti_bolognese": "Spaghetti bolognese",
    "spaghetti_carbonara": "Spaghetti carbonara",
    "spring_rolls": "Spring rolls",
    "strawberry_shortcake": "Strawberry shortcake",
    "tuna_tartare": "Tuna tartare",
    "zucchini_boats": "Zucchini boats",
}


def format_label(raw_label: str) -> str:
    """Turn a Food-101 class key into a readable name."""
    key = str(raw_label).strip()
    if key in LABEL_DISPLAY_OVERRIDES:
        return LABEL_DISPLAY_OVERRIDES[key]
    return " ".join(word.capitalize() for word in key.replace("-", "_").split("_") if word)


def confidence_tier(confidence: float) -> str:
    """Classify a confidence percentage as ``high`` / ``medium`` / ``low``."""
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    if confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


@dataclass
class FoodPrediction:
    """The result of analysing one food photo."""

    label: str                      # display label, e.g. "Chicken curry"
    raw_label: str                  # dataset key, e.g. "chicken_curry"
    confidence: float               # 0-100 for the top-1 class
    alternatives: list[tuple[str, float]] = field(default_factory=list)   # (label, confidence)
    margin: float = 0.0             # top-1 minus top-2 confidence (percentage points)
    is_uncertain: bool = False
    tier: str = "low"
    backend: str = ""
    model_name: str = ""
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        """Plain dictionary (used for session-state storage and tests)."""
        return {
            "label": self.label,
            "raw_label": self.raw_label,
            "confidence": self.confidence,
            "alternatives": list(self.alternatives),
            "margin": self.margin,
            "is_uncertain": self.is_uncertain,
            "tier": self.tier,
            "backend": self.backend,
            "model_name": self.model_name,
            "inference_ms": self.inference_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FoodPrediction":
        """Rebuild a prediction from :meth:`to_dict` output."""
        return cls(
            label=data["label"],
            raw_label=data["raw_label"],
            confidence=float(data["confidence"]),
            alternatives=[(str(label), float(score)) for label, score in data.get("alternatives", [])],
            margin=float(data.get("margin", 0.0)),
            is_uncertain=bool(data.get("is_uncertain", False)),
            tier=str(data.get("tier", "low")),
            backend=str(data.get("backend", "")),
            model_name=str(data.get("model_name", "")),
            inference_ms=float(data.get("inference_ms", 0.0)),
        )

    @property
    def confidence_text(self) -> str:
        return f"{self.confidence:.1f}%"

    def explanation(self) -> str:
        """Short, user-facing description of how much to trust this result."""
        if self.tier == "high":
            return "The model is fairly confident about this food."
        if self.tier == "medium":
            return "The model is moderately confident — please check the label before trusting it."
        return "The model is unsure about this image. Confirm the food yourself before using it."


class FoodPredictor:
    """Turns an image into a :class:`FoodPrediction` using a loaded model."""

    def __init__(
        self,
        model: Optional[BaseFoodModel] = None,
        top_k: int = 5,
        backend: Optional[str] = None,
    ) -> None:
        self._model = model
        self._top_k = int(top_k)
        # None / "auto" defers to FOOD_MODEL_BACKEND, then to automatic selection.
        self._backend = resolve_backend(backend)

    # -- model access ------------------------------------------------------ #
    @property
    def model(self) -> BaseFoodModel:
        """The loaded model, loading it on first use."""
        if self._model is None:
            self._model = load_food_model(self._backend)
        return self._model

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, str]:
        """Model metadata for the UI (never throws; reports load failures)."""
        try:
            return self.model.describe()
        except ModelUnavailableError as exc:
            return {"Backend": "unavailable", "Reason": str(exc).splitlines()[0]}

    # -- inference --------------------------------------------------------- #
    def predict(self, image) -> FoodPrediction:
        """Analyse a PIL image. Raises :class:`ModelUnavailableError` if no model."""
        model = self.model                      # may raise: no fake results
        labels = load_labels()

        batch = preprocess(image)
        started = time.perf_counter()
        probabilities = model.predict_proba(batch)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ranked = sorted(
            ((labels[index], float(score) * 100.0) for index, score in enumerate(probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )

        top = ranked[: max(1, self._top_k)]
        top_label, top_score = top[0]
        runner_up_score = top[1][1] if len(top) > 1 else 0.0
        margin = top_score - runner_up_score
        tier = confidence_tier(top_score)
        uncertain = tier == "low" or margin < UNCERTAIN_MARGIN

        prediction = FoodPrediction(
            label=format_label(top_label),
            raw_label=top_label,
            confidence=round(top_score, 2),
            alternatives=[(format_label(label), round(score, 2)) for label, score in top],
            margin=round(margin, 2),
            is_uncertain=uncertain,
            tier=tier,
            backend=model.backend,
            model_name=model.model_name,
            inference_ms=round(elapsed_ms, 1),
        )
        logger.info(
            "Prediction | %s %.1f%% (margin %.1f, %s) in %.0f ms via %s",
            prediction.raw_label,
            prediction.confidence,
            prediction.margin,
            prediction.tier,
            prediction.inference_ms,
            prediction.backend,
        )
        return prediction

    def predict_top_k_text(self, prediction: FoodPrediction) -> str:
        """One-line summary of the ranking, e.g. for logs or captions."""
        return ", ".join(f"{label} ({score:.1f}%)" for label, score in prediction.alternatives)
