"""Glue between the two independent pipelines for the comparison page (no Streamlit imports).

``run_original`` calls the unchanged original FoodPredictor; ``run_enhanced`` calls the
enhanced predictor. Each returns a plain dict tagged with ``source`` so a result can never be
shown under the wrong model name. Neither function looks at the other's output.
"""
from __future__ import annotations

from typing import Any, Optional

from enhanced_model.predict import (EnhancedImageError, EnhancedModelUnavailableError,
                                    enhanced_model_status, load_enhanced_predictor, load_rgb)

ORIGINAL = "original"
ENHANCED = "enhanced"
ORIGINAL_NAME = "Original Model (MobileNetV2 · Food-101, 101 classes)"
ENHANCED_NAME = "Enhanced Model (EfficientNet-B0 · fine-tuned, target classes)"


def run_original(image: Any) -> dict[str, Any]:
    """Original pipeline, exactly as the Upload Meal page uses it."""
    from food_recognition import FoodPredictor, ModelUnavailableError, UnsupportedImageError
    try:
        pil = _to_pil(image)
        p = FoodPredictor().predict(pil)
    except ModelUnavailableError as exc:
        return {"source": ORIGINAL, "ok": False, "error": f"Original model unavailable: {exc}"}
    except (UnsupportedImageError, EnhancedImageError) as exc:
        return {"source": ORIGINAL, "ok": False, "error": str(exc)}
    return {"source": ORIGINAL, "ok": True, "label": p.label, "raw_label": p.raw_label,
            "confidence": p.confidence, "top": list(p.alternatives[:5]), "uncertain": p.is_uncertain,
            "reasons": [], "inference_ms": p.inference_ms}


def run_enhanced(image: Any) -> dict[str, Any]:
    try:
        p = load_enhanced_predictor().predict(image)
    except EnhancedModelUnavailableError as exc:
        return {"source": ENHANCED, "ok": False, "unavailable": True, "error": str(exc)}
    except EnhancedImageError as exc:
        return {"source": ENHANCED, "ok": False, "error": str(exc)}
    return {"source": ENHANCED, "ok": True, "label": p.label, "raw_label": p.raw_label,
            "confidence": p.confidence, "top": list(p.top_k), "uncertain": p.is_uncertain,
            "reasons": list(p.uncertainty_reasons), "inference_ms": p.inference_ms}


def _to_pil(image: Any):
    try:
        return load_rgb(image)
    except EnhancedImageError:
        raise


def history_row(name: str, actual: str, original: Optional[dict], enhanced: Optional[dict]) -> dict[str, Any]:
    def part(res: Optional[dict]):
        if not res or not res.get("ok"):
            return "", "", ""
        correct = (res["raw_label"] == actual) if actual else ""
        return res["label"], f"{res['confidence']:.1f}", correct
    ol, oc, ok = part(original)
    el, ec, ek = part(enhanced)
    return {"image": name, "actual": actual or "unknown", "original_prediction": ol, "original_confidence_%": oc,
            "original_correct": ok, "enhanced_prediction": el, "enhanced_confidence_%": ec, "enhanced_correct": ek}


def agree(original: Optional[dict], enhanced: Optional[dict]) -> Optional[bool]:
    if not (original and enhanced and original.get("ok") and enhanced.get("ok")):
        return None
    return original["raw_label"] == enhanced["raw_label"]


__all__ = ["run_original", "run_enhanced", "history_row", "agree", "enhanced_model_status",
           "ORIGINAL", "ENHANCED", "ORIGINAL_NAME", "ENHANCED_NAME"]
