"""Pretrained food-recognition model loading (Phase 3).

Model
-----
``AlexKoff88/mobilenet_v2_food101`` — torchvision **MobileNetV2** fine-tuned on
**Food-101** (101 classes, 76.3 % top-1 on the Food-101 test split, Apache-2.0).
It is ~19 MB, runs on CPU in well under a second and needs no GPU, which is what
makes it usable on Streamlit Community Cloud.

Two interchangeable backends
---------------------------
* ``onnx``  — ONNX Runtime, ~10 MB file (or ~2.5 MB dynamic-INT8). Light on RAM,
  ideal for deployment. Create it once with ``python -m food_recognition.export_onnx``.
* ``torch`` — PyTorch + torchvision, used when no ONNX file is present; downloads
  the published checkpoint on first use.

``backend="auto"`` (default) prefers ONNX when a valid file exists, then Torch.

Honesty rules implemented here
------------------------------
* Nothing is ever fabricated: if no backend can be loaded, :class:`ModelUnavailableError`
  is raised with an actionable message and the UI shows an error — no substitute labels.
* Every loaded model must pass :func:`sanity_check` (a real inference on a
  deterministic synthetic image that verifies the output looks like a genuine
  probability distribution). This check exists because the only pre-exported
  Food-101 ONNX file found on the Hub returned **all-zero logits for every
  input, including random noise** — a broken artifact that would have silently
  produced meaningless "predictions".

Weights are cached in ``.cache/models/`` (git-ignored). Nothing is written
outside the project directory.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

from config import (
    ASSETS_DIR,
    CACHE_DIR,
    FOOD_MODEL_BACKEND,
    FOOD_MODEL_ID,
    FOOD_MODEL_REVISION,
    FOOD_NUM_CLASSES,
    FOOD_ONNX_INT8_FILENAME,
    FOOD_ONNX_FILENAME,
    FOOD_TORCH_WEIGHTS_FILENAME,
    FOOD_TORCH_WEIGHTS_REMOTE_NAME,
    FOOD101_CLASSES_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_IMAGE_SIZE,
)
from food_recognition.preprocessing import preprocess

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR: Path = CACHE_DIR / "models"
BUNDLED_MODEL_DIR: Path = ASSETS_DIR / "models"          # optional, for committing an ONNX file
HF_RESOLVE_URL: str = f"https://huggingface.co/{FOOD_MODEL_ID}/resolve/{FOOD_MODEL_REVISION}/"

#: Minimum plausible size for a downloaded checkpoint (guards truncated downloads).
MIN_WEIGHTS_BYTES: int = 5 * 1024 * 1024  # 5 MB (real files are ~9-19 MB)

#: Logits with a standard deviation below this are considered degenerate.
MIN_LOGIT_STD: float = 1e-4


class ModelUnavailableError(RuntimeError):
    """Raised when no usable food-recognition model is available."""


def resolve_backend(requested: Optional[str] = None) -> str:
    """Decide which backend to use.

    Precedence: an explicit argument, then the ``FOOD_MODEL_BACKEND`` setting
    (environment or config), then ``"auto"``. ``"auto"`` is only a *fallback*
    value — it never overrides an explicit configuration.
    """
    value = (requested or "").strip().lower()
    if not value or value == "auto":
        value = (FOOD_MODEL_BACKEND or "auto").strip().lower()
    return value or "auto"


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def load_labels(path: Optional[Path] = None) -> tuple[str, ...]:
    """Return the 101 Food-101 class names in model output order.

    The file ships with the project (``food_recognition/food101_classes.json``)
    and was taken from the dataset's own label definition
    (``ethz/food101`` on the Hugging Face Hub), which matches torchvision's
    alphabetical ordering and therefore the order this checkpoint predicts in.
    """
    labels_path = Path(path) if path else FOOD101_CLASSES_PATH
    try:
        names = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(
            f"Could not read the label list at {labels_path}: {exc}"
        ) from exc

    if not isinstance(names, list) or len(names) != FOOD_NUM_CLASSES:
        raise ModelUnavailableError(
            f"Expected {FOOD_NUM_CLASSES} class names in {labels_path}, "
            f"found {len(names) if isinstance(names, list) else 'no list'}."
        )
    return tuple(str(name) for name in names)


# --------------------------------------------------------------------------- #
# Downloads & sanity check
# --------------------------------------------------------------------------- #
def _download(url: str, destination: Path, progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Download ``url`` to ``destination`` atomically (temp file + rename)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "NutriVision-AI/0.3"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(temp_path, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"Download failed for {url} ({exc.__class__.__name__}: {exc}). "
            "Check your internet connection or place the model file in "
            f"'{MODEL_CACHE_DIR}' manually."
        ) from exc

    temp_path.replace(destination)
    logger.info("Downloaded %s -> %s", url, destination)
    return destination


def sanity_check(predict_proba: Callable[[np.ndarray], np.ndarray], backend: str) -> None:
    """Verify a freshly loaded model behaves like a real classifier.

    Runs one inference on a deterministic synthetic image and rejects models
    whose output is not a proper (non-degenerate) probability distribution.

    Raises:
        ModelUnavailableError: if the output suggests the model is broken.
    """
    labels = load_labels()
    if len(labels) != FOOD_NUM_CLASSES:  # pragma: no cover - guarded by load_labels
        raise ModelUnavailableError("Label list is incomplete.")

    # A fixed pattern (not pure noise) so the test is deterministic and repeatable.
    test_image = Image.new("RGB", (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), (128, 96, 64))
    for x in range(0, MODEL_IMAGE_SIZE, 32):
        for y in range(MODEL_IMAGE_SIZE // 2, MODEL_IMAGE_SIZE):
            test_image.putpixel((x, y), (200, 180, 120))

    batch = preprocess(test_image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    probabilities = np.asarray(predict_proba(batch), dtype=np.float64)

    if probabilities.shape != (1, FOOD_NUM_CLASSES):
        raise ModelUnavailableError(
            f"The {backend} model returned an unexpected output shape "
            f"{probabilities.shape} (expected (1, {FOOD_NUM_CLASSES}))."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ModelUnavailableError(f"The {backend} model produced non-finite scores.")
    if probabilities.min() < -1e-6 or probabilities.max() > 1.0 + 1e-6:
        raise ModelUnavailableError(f"The {backend} model did not output probabilities.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        raise ModelUnavailableError(f"The {backend} model probabilities do not sum to 1.")
    if float(probabilities.std()) < MIN_LOGIT_STD:
        raise ModelUnavailableError(
            f"The {backend} model output the same score for every class — the model file "
            "appears to be broken (this is a known problem with at least one pre-exported "
            "Food-101 ONNX file). Re-create it with `python -m food_recognition.export_onnx` "
            "or switch backend with FOOD_MODEL_BACKEND=torch."
        )
    logger.info("Sanity check passed for the %s backend (top-1 spread %.4f)", backend, float(probabilities.std()))


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class BaseFoodModel:
    """Common interface for the concrete backends."""

    backend: str = "base"
    model_name: str = FOOD_MODEL_ID
    model_path: Optional[Path] = None
    load_seconds: float = 0.0

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return FOOD_NUM_CLASSES

    def describe(self) -> dict[str, str]:
        return {
            "Backend": self.backend,
            "Model": self.model_name,
            "Classes": str(self.num_classes),
            "Device": "CPU",
            "Load time": f"{self.load_seconds:.2f} s",
            "Model file": str(self.model_path) if self.model_path else "—",
        }


class OnnxFoodModel(BaseFoodModel):
    """ONNX Runtime backend (light: ~10 MB FP32 or ~2.5 MB INT8)."""

    backend = "onnx"

    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailableError(
                "onnxruntime is not installed. Install it with `pip install onnxruntime`, "
                "or set FOOD_MODEL_BACKEND=torch."
            ) from exc

        started = time.perf_counter()
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - any ORT failure is a load failure
            raise ModelUnavailableError(
                f"ONNX Runtime could not load '{model_path.name}': {exc}. "
                "Delete the file and re-create it with `python -m food_recognition.export_onnx`."
            ) from exc

        self._input_name = self._session.get_inputs()[0].name
        self.model_path = Path(model_path)
        self.model_name = f"{FOOD_MODEL_ID} → ONNX"
        self.load_seconds = time.perf_counter() - started

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        logits = self._session.run(None, {self._input_name: batch.astype(np.float32)})[0]
        return _softmax(np.asarray(logits, dtype=np.float64))


class TorchFoodModel(BaseFoodModel):
    """PyTorch/torchvision backend (downloads the published checkpoint)."""

    backend = "torch"

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        try:
            import torch
            import torchvision.models as tvm
        except ImportError as exc:
            raise ModelUnavailableError(
                "PyTorch is not installed. Install the CPU build with:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                "or generate an ONNX model with `python -m food_recognition.export_onnx`."
            ) from exc

        started = time.perf_counter()
        self._torch = torch
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

        weights = Path(weights_path) if weights_path else ensure_torch_weights()
        model = tvm.mobilenet_v2(num_classes=FOOD_NUM_CLASSES)
        try:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Could not read the checkpoint '{weights.name}': {exc}. "
                f"Delete it and retry so it is downloaded again."
            ) from exc

        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {str(key).replace("module.", ""): value for key, value in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ModelUnavailableError(
                f"The checkpoint '{weights.name}' does not match MobileNetV2/101 "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys). "
                "Delete the file and let it download again."
            )

        model.eval()
        self._model = model
        self.model_path = weights
        self.model_name = f"{FOOD_MODEL_ID} (MobileNetV2, Food-101)"
        self.load_seconds = time.perf_counter() - started

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(batch.astype(np.float32))
        with self._torch.no_grad():
            logits = self._model(tensor).numpy()
        return _softmax(np.asarray(logits, dtype=np.float64))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Weight / model files
# --------------------------------------------------------------------------- #
def torch_weights_path() -> Path:
    """Where the PyTorch checkpoint is cached."""
    return MODEL_CACHE_DIR / FOOD_TORCH_WEIGHTS_FILENAME


def find_onnx_path() -> Optional[Path]:
    """Return a usable ONNX file.

    Full-precision is preferred by default because it matches PyTorch to ~1e-7;
    the smaller INT8 model can shift probabilities slightly on ambiguous inputs.
    Set ``FOOD_MODEL_PREFER_INT8=1`` to use the 2.5 MB quantised file instead.
    """
    fp32 = (MODEL_CACHE_DIR / FOOD_ONNX_FILENAME, BUNDLED_MODEL_DIR / FOOD_ONNX_FILENAME)
    int8 = (MODEL_CACHE_DIR / FOOD_ONNX_INT8_FILENAME, BUNDLED_MODEL_DIR / FOOD_ONNX_INT8_FILENAME)
    prefer_int8 = os.environ.get("FOOD_MODEL_PREFER_INT8", "").strip() in {"1", "true", "yes"}
    candidates = int8 + fp32 if prefer_int8 else fp32 + int8
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate
    return None


def ensure_torch_weights(progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Download the MobileNetV2 Food-101 checkpoint if it is not cached yet."""
    destination = torch_weights_path()
    if destination.is_file() and destination.stat().st_size >= MIN_WEIGHTS_BYTES:
        return destination

    # An externally provided path (useful offline) always wins.
    override = os.environ.get("FOOD_MODEL_PATH")
    if override and Path(override).is_file():
        logger.info("Using checkpoint from FOOD_MODEL_PATH=%s", override)
        return Path(override)

    logger.info(
        "Downloading %s from %s — first run only", FOOD_TORCH_WEIGHTS_REMOTE_NAME, FOOD_MODEL_ID
    )
    _download(HF_RESOLVE_URL + FOOD_TORCH_WEIGHTS_REMOTE_NAME, destination, progress)

    if destination.stat().st_size < MIN_WEIGHTS_BYTES:
        destination.unlink(missing_ok=True)
        raise ModelUnavailableError(
            "The downloaded checkpoint looks incomplete (too small). Please try again."
        )
    return destination


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def resource_cache(func: Callable):
    """Cache an expensive loader once per process.

    Uses Streamlit's ``st.cache_resource`` when running inside Streamlit (the
    model is then shared by every user session and never reloaded on reruns) and
    falls back to ``functools.lru_cache`` in scripts and tests.
    """
    try:
        import streamlit as st

        if st.runtime.exists():
            return st.cache_resource(show_spinner=False)(func)
    except Exception:  # pragma: no cover - Streamlit missing or unusual state
        pass
    return functools.lru_cache(maxsize=None)(func)


@resource_cache
def load_food_model(backend: str = "auto", force_reload: bool = False) -> BaseFoodModel:
    """Load and cache the food-recognition model.

    Args:
        backend: ``"auto"`` (default), ``"onnx"`` or ``"torch"``.
        force_reload: kept for API symmetry; pass a different ``backend`` value
            or restart the app to reload.

    Returns:
        A loaded model exposing ``predict_proba(batch) -> (n, 101)``.

    Raises:
        ModelUnavailableError: when no model can be loaded. The message explains
            exactly what to install or fix — no prediction is ever faked.
    """
    requested = resolve_backend(backend)

    if requested not in {"auto", "onnx", "torch"}:
        raise ModelUnavailableError(
            f"Unknown backend '{requested}'. Use 'auto', 'onnx' or 'torch'."
        )

    errors: list[str] = []

    candidates: Sequence[tuple[str, Callable[[], BaseFoodModel]]]
    if requested == "onnx":
        candidates = (("onnx", lambda: OnnxFoodModel(find_onnx_path() or _missing_onnx())),)
    elif requested == "torch":
        candidates = (("torch", TorchFoodModel),)
    else:
        onnx_path = find_onnx_path()
        candidates = tuple(
            filter(
                None,
                (
                    ("onnx", (lambda p=onnx_path: OnnxFoodModel(p))) if onnx_path else None,
                    ("torch", TorchFoodModel),
                ),
            )
        )

    for name, factory in candidates:
        try:
            model = factory()
            sanity_check(model.predict_proba, name)
            logger.info(
                "Food model ready | backend=%s file=%s load=%.2fs",
                name,
                model.model_path,
                model.load_seconds,
            )
            return model
        except ModelUnavailableError as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("Backend '%s' unavailable: %s", name, exc)
        except Exception as exc:  # noqa: BLE001 - never let a backend crash the app
            errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
            logger.exception("Unexpected failure while loading the '%s' backend", name)

    raise ModelUnavailableError(
        "No usable food-recognition model could be loaded. "
        + ("\n\n".join(errors) if errors else "No backend was tried.")
        + "\n\nQuick fixes:\n"
        "  • `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`"
        " (PyTorch backend), or\n"
        "  • `pip install onnxruntime` + `python -m food_recognition.export_onnx`"
        " (lighter ONNX backend)."
    )


def _missing_onnx() -> Path:
    raise ModelUnavailableError(
        "No ONNX model file was found. Create one with "
        "`python -m food_recognition.export_onnx` (needs PyTorch once), or use "
        "FOOD_MODEL_BACKEND=torch."
    )


def model_status() -> dict[str, str]:
    """Cheap, non-throwing description of what the app would load.

    Used by the UI to show model availability *before* a user uploads anything.
    """
    onnx_path = find_onnx_path()
    torch_weights = torch_weights_path()
    status = {
        "Requested backend": resolve_backend(),
        "ONNX file": f"found ({onnx_path.name}, {onnx_path.stat().st_size / 1e6:.1f} MB)"
        if onnx_path
        else "not found",
        "PyTorch weights": f"cached ({torch_weights.stat().st_size / 1e6:.1f} MB)"
        if torch_weights.is_file()
        else "not downloaded yet",
        "Model": f"{FOOD_MODEL_ID} · MobileNetV2 · Food-101 · {FOOD_NUM_CLASSES} classes",
        "Cache folder": str(MODEL_CACHE_DIR),
    }
    try:
        status["torch installed"] = "yes" if _has_module("torch") else "no"
        status["onnxruntime installed"] = "yes" if _has_module("onnxruntime") else "no"
    except Exception:  # pragma: no cover
        pass
    return status


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
