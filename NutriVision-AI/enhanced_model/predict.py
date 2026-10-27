"""Runtime inference for the Enhanced Food Classifier (Model 2).

Needs only onnxruntime + NumPy + Pillow (already project dependencies). It never
imports the original ``food_recognition`` package: its own preprocessing, labels
and checkpoint are read from ``enhanced_model/artifacts`` and ``checkpoints``.
If a file is missing it raises :class:`EnhancedModelUnavailableError`; it never
falls back to the original model.
"""
from __future__ import annotations

import functools
import io
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from enhanced_model import config as cfg

logger = logging.getLogger(__name__)
MODEL_LABEL = "Enhanced Food Classifier"


class EnhancedModelUnavailableError(RuntimeError):
    """The enhanced model (checkpoint / mapping / runtime) cannot be used."""


class EnhancedImageError(ValueError):
    """The input is not a usable image (message is safe to show to users)."""


@dataclass
class EnhancedPrediction:
    label: str
    raw_label: str
    confidence: float                      # softmax score of top-1, percent (NOT calibrated)
    top_k: list[tuple[str, float]] = field(default_factory=list)   # (display label, percent)
    margin: float = 0.0
    is_uncertain: bool = False
    uncertainty_reasons: list[str] = field(default_factory=list)
    inference_ms: float = 0.0
    model_name: str = MODEL_LABEL
    source: str = "enhanced"               # lets the UI/tests verify provenance

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def display_label(raw: str) -> str:
    return " ".join(w.capitalize() for w in raw.replace("-", "_").split("_") if w)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise EnhancedModelUnavailableError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnhancedModelUnavailableError(f"{what} is unreadable ({path}): {exc}") from exc


def load_class_mapping(path: Path = cfg.CLASS_MAPPING_PATH) -> list[str]:
    data = _read_json(path, "Enhanced class mapping")
    classes = data.get("classes") if isinstance(data, dict) else data
    if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(c, str) for c in classes):
        raise EnhancedModelUnavailableError(f"Class mapping {path} must list at least two class names.")
    if len(set(classes)) != len(classes):
        raise EnhancedModelUnavailableError(f"Class mapping {path} contains duplicate names.")
    return classes


def load_preprocessing(path: Path = cfg.PREPROCESSING_PATH) -> dict[str, Any]:
    data = _read_json(path, "Enhanced preprocessing config")
    for key in ("image_size", "resize_size", "mean", "std"):
        if key not in data:
            raise EnhancedModelUnavailableError(f"Preprocessing config {path} lacks '{key}'.")
    return data


def load_thresholds(path: Path = cfg.THRESHOLDS_PATH) -> dict[str, Any]:
    """Validation-derived thresholds if calibrated, otherwise documented defaults."""
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "min_confidence": float(data["min_confidence"]),
                "min_margin": float(data["min_margin"]),
                "calibrated_on": str(data.get("calibrated_on", "validation")),
            }
        except (OSError, ValueError, KeyError):
            logger.warning("Ignoring unreadable thresholds file %s", path)
    return {
        "min_confidence": cfg.DEFAULT_MIN_CONFIDENCE,
        "min_margin": cfg.DEFAULT_MIN_MARGIN,
        "calibrated_on": "defaults (not calibrated)",
    }


# --------------------------------------------------------------------------- #
# Image handling (own implementation — independent of the original pipeline)
# --------------------------------------------------------------------------- #
def load_rgb(source: Any) -> Image.Image:
    """Return an RGB image from a PIL image, path, bytes or file-like object."""
    Image.MAX_IMAGE_PIXELS = cfg.MAX_IMAGE_PIXELS
    try:
        if isinstance(source, Image.Image):
            image = source
        else:
            if isinstance(source, (str, Path)):
                data = Path(source).read_bytes()
            elif isinstance(source, (bytes, bytearray)):
                data = bytes(source)
            else:
                data = source.read()
                if hasattr(source, "seek"):
                    source.seek(0)
            if not data:
                raise EnhancedImageError("The image file is empty.")
            if len(data) / 1048576 > cfg.MAX_UPLOAD_MB:
                raise EnhancedImageError(f"The image is larger than {cfg.MAX_UPLOAD_MB:g} MB.")
            image = Image.open(io.BytesIO(data))
            image.load()
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))   # transparency -> white
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")                                           # also handles grayscale
    except EnhancedImageError:
        raise
    except FileNotFoundError as exc:
        raise EnhancedImageError(f"Image file not found: {exc.filename}") from exc
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise EnhancedImageError(
            f"That file could not be read as an image ({exc.__class__.__name__})."
        ) from exc


def preprocess_enhanced(image: Image.Image, pre: dict[str, Any]) -> np.ndarray:
    """Resize shorter side, centre-crop, normalise -> (1, 3, H, W) float32."""
    size, resize = int(pre["image_size"]), int(pre["resize_size"])
    rgb = image
    width, height = rgb.size
    if width < 1 or height < 1:
        raise EnhancedImageError("The image has no pixels.")
    scale = resize / min(width, height)
    rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    width, height = rgb.size
    left, top = max(0, (width - size) // 2), max(0, (height - size) // 2)
    rgb = rgb.crop((left, top, left + size, top + size))
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    array = (array - np.asarray(pre["mean"], np.float32)) / np.asarray(pre["std"], np.float32)
    return np.transpose(array, (2, 0, 1))[None].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class _SafeUnpickler(pickle.Unpickler):
    """Bundle files contain only dict/list/str/float/bytes, so NO global may be loaded.
    This blocks the code-execution risk of unpickling an untrusted .pkl."""

    def find_class(self, module, name):  # noqa: D401
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from model bundle.")


def save_bundle(path: Path, onnx_path: Path, classes: list[str], pre: dict, thresholds: dict, meta: dict) -> None:
    bundle = {"format": "nutrivision-enhanced-bundle-v1", "onnx_bytes": Path(onnx_path).read_bytes(),
              "classes": list(classes), "preprocessing": dict(pre), "thresholds": dict(thresholds), "meta": dict(meta)}
    Path(path).write_bytes(pickle.dumps(bundle, protocol=4))


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            bundle = _SafeUnpickler(fh).load()
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
        raise EnhancedModelUnavailableError(f"Model bundle {Path(path).name} could not be loaded: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != "nutrivision-enhanced-bundle-v1" \
            or not isinstance(bundle.get("onnx_bytes"), bytes):
        raise EnhancedModelUnavailableError(f"{Path(path).name} is not a valid enhanced-model bundle.")
    return bundle


class EnhancedPredictor:
    """Loads the enhanced model (.pkl bundle if present, else .onnx + JSON artifacts)."""

    def __init__(self, model_path: Optional[Path] = None, mapping_path: Optional[Path] = None,
                 preprocessing_path: Optional[Path] = None, thresholds_path: Optional[Path] = None):
        # Defaults are resolved at call time so tests / deployments can point config elsewhere.
        if model_path is None:
            model_path = cfg.PICKLE_MODEL_PATH if Path(cfg.PICKLE_MODEL_PATH).is_file() else cfg.ONNX_MODEL_PATH
        model_path = Path(model_path)
        if not model_path.is_file():
            raise EnhancedModelUnavailableError(
                f"Enhanced model checkpoint not found: {model_path}\n"
                "Train it with `python -m enhanced_model.prepare_dataset` then "
                "`python -m enhanced_model.train` (see enhanced_model/README.md)."
            )
        if model_path.suffix == ".pkl":
            bundle = load_bundle(model_path)
            self.classes, self.pre = list(bundle["classes"]), dict(bundle["preprocessing"])
            self.thresholds = dict(bundle["thresholds"])
            model_source: Any = bundle["onnx_bytes"]
            if len(self.classes) < 2:
                raise EnhancedModelUnavailableError("Bundle lists fewer than two classes.")
        else:
            self.classes = load_class_mapping(mapping_path or cfg.CLASS_MAPPING_PATH)
            self.pre = load_preprocessing(preprocessing_path or cfg.PREPROCESSING_PATH)
            self.thresholds = load_thresholds(thresholds_path or cfg.THRESHOLDS_PATH)
            model_source = str(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EnhancedModelUnavailableError("onnxruntime is not installed (`pip install onnxruntime`).") from exc
        try:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(model_source, sess_options=options,
                                                 providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            raise EnhancedModelUnavailableError(f"ONNX Runtime could not load {model_path.name}: {exc}") from exc
        out_dim = self._session.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != len(self.classes):
            raise EnhancedModelUnavailableError(
                f"Class mapping has {len(self.classes)} classes but the model outputs {out_dim}. "
                "The mapping and checkpoint are from different training runs."
            )
        self._input = self._session.get_inputs()[0].name
        self.model_path = model_path

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        batch = preprocess_enhanced(image, self.pre)
        logits = self._session.run(None, {self._input: batch})[0]
        return _softmax(np.asarray(logits, dtype=np.float64))[0]

    def predict(self, source: Any, top_k: int = cfg.TOP_K) -> EnhancedPrediction:
        image = load_rgb(source)
        started = time.perf_counter()
        probs = self.predict_proba(image)
        elapsed = (time.perf_counter() - started) * 1000.0
        order = np.argsort(-probs)[: max(1, min(top_k, len(self.classes)))]
        top = [(display_label(self.classes[i]), round(float(probs[i]) * 100, 2)) for i in order]
        confidence = float(probs[order[0]]) * 100
        margin = confidence - (float(probs[order[1]]) * 100 if len(order) > 1 else 0.0)
        reasons: list[str] = []
        if confidence < self.thresholds["min_confidence"]:
            reasons.append(f"confidence {confidence:.1f}% is below {self.thresholds['min_confidence']:.0f}%")
        if margin < self.thresholds["min_margin"]:
            reasons.append(f"top two classes are only {margin:.1f} points apart")
        return EnhancedPrediction(
            label=display_label(self.classes[order[0]]), raw_label=self.classes[order[0]],
            confidence=round(confidence, 2), top_k=top, margin=round(margin, 2),
            is_uncertain=bool(reasons), uncertainty_reasons=reasons, inference_ms=round(elapsed, 1),
        )


@functools.lru_cache(maxsize=1)
def _cached_predictor() -> EnhancedPredictor:
    return EnhancedPredictor()


def load_enhanced_predictor(use_cache: bool = True) -> EnhancedPredictor:
    return _cached_predictor() if use_cache else EnhancedPredictor()


def enhanced_model_status() -> dict[str, Any]:
    """Cheap, non-throwing availability report for the UI."""
    if Path(cfg.PICKLE_MODEL_PATH).is_file():
        return {"available": True, "missing": []}
    missing = [p.name for p in (cfg.ONNX_MODEL_PATH, cfg.CLASS_MAPPING_PATH, cfg.PREPROCESSING_PATH) if not p.is_file()]
    return {"available": not missing, "missing": missing}


def predict_enhanced(image: Any, top_k: int = cfg.TOP_K) -> EnhancedPrediction:
    """Public entry point: enhanced-model prediction for one image."""
    return load_enhanced_predictor().predict(image, top_k=top_k)
