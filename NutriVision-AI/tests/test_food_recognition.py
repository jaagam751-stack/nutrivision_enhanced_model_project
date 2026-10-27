"""Phase 3 tests: upload validation, preprocessing, labels, prediction logic.

Design notes
------------
* All images are generated in memory with Pillow — no copyrighted photos are
  committed to the repository.
* Model inference is exercised through a ``StubModel`` so the unit tests stay
  fast and offline. The real model is covered by an integration test that is
  skipped automatically when neither backend is installed/available.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from config import (
    ALLOWED_IMAGE_TYPES,
    CONFIDENCE_HIGH,
    FOOD_NUM_CLASSES,
    IMAGE_MAX_PIXELS,
    MAX_UPLOAD_MB,
    MODEL_IMAGE_SIZE,
)
from food_recognition import preprocessing as prep
from food_recognition.model_loader import (
    ModelUnavailableError,
    find_onnx_path,
    load_labels,
    sanity_check,
)
from food_recognition.predictor import FoodPrediction, FoodPredictor, confidence_tier, format_label


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
def make_image(width: int = 640, height: int = 480, colour=(180, 90, 60), fmt: str = "JPEG") -> bytes:
    """Create an in-memory test image and return its encoded bytes."""
    image = Image.new("RGB", (width, height), colour)
    for x in range(0, width, 40):                       # add some texture
        for y in range(height // 2, height, 30):
            image.putpixel((x, y), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class UploadedFileStub:
    """Minimal stand-in for a Streamlit ``UploadedFile``."""

    def __init__(self, data: bytes, name: str = "meal.jpg", mime: str = "image/jpeg") -> None:
        self._data = data
        self.name = name
        self.type = mime
        self.size = len(data)
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            self._position = len(self._data)
            return self._data
        chunk = self._data[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    def seek(self, position: int) -> None:
        self._position = position

    def getvalue(self) -> bytes:
        return self._data


class StubModel:
    """Deterministic fake backend used to test the predictor logic in isolation."""

    backend = "stub"
    model_name = "stub-model"
    model_path = None
    load_seconds = 0.0

    def __init__(self, ranking: list[int] | None = None, flat: bool = False) -> None:
        self._ranking = ranking or [85, 20, 10, 0, 99]      # samosa, chicken_wings, bruschetta, ...
        self._flat = flat

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        scores = np.full((batch.shape[0], FOOD_NUM_CLASSES), 1.0 / FOOD_NUM_CLASSES)
        if not self._flat:
            scores[:] = 1e-6
            for position, index in enumerate(self._ranking):
                scores[:, index] = 0.5 if position == 0 else 0.5 / (position + 2)
        return scores / scores.sum(axis=1, keepdims=True)

    def describe(self) -> dict[str, str]:
        return {"Backend": self.backend, "Model": self.model_name}


# --------------------------------------------------------------------------- #
# Label list
# --------------------------------------------------------------------------- #
def test_label_list_has_101_entries_and_is_in_model_order() -> None:
    labels = load_labels()
    assert len(labels) == FOOD_NUM_CLASSES == 101
    # The first and last entries of the canonical Food-101 ordering.
    assert labels[0] == "apple_pie"
    assert labels[-1] == "waffles"
    assert labels[85] == "samosa"          # verified against the model output during development
    assert "chicken_curry" in labels


def test_missing_label_file_raises_clear_error(tmp_path) -> None:
    with pytest.raises(ModelUnavailableError):
        load_labels(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text('["only_one"]')
    with pytest.raises(ModelUnavailableError):
        load_labels(broken)


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
def test_valid_jpeg_and_png_are_accepted() -> None:
    jpeg = UploadedFileStub(make_image(fmt="JPEG"), "lunch.jpg", "image/jpeg")
    png = UploadedFileStub(make_image(fmt="PNG"), "dinner.png", "image/png")
    for upload in (jpeg, png):
        info = prep.validate_upload(upload, filename=upload.name)
        assert info.width == 640 and info.height == 480
        assert 0 < info.size_mb < MAX_UPLOAD_MB
    assert info.image_format == "PNG"


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(prep.UnsupportedImageError) as excinfo:
        prep.validate_upload(make_image(fmt="PNG"), filename="menu.gif")
    assert "not supported" in str(excinfo.value).lower()
    assert set(ALLOWED_IMAGE_TYPES) == {"jpg", "jpeg", "png"}


def test_renamed_file_is_rejected_by_real_format_check() -> None:
    """A BMP renamed to .jpg must be caught by the format sniff, not the suffix."""
    image = Image.new("RGB", (64, 64), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")
    with pytest.raises(prep.UnsupportedImageError) as excinfo:
        prep.validate_upload(UploadedFileStub(buffer.getvalue(), "photo.jpg"))
    assert "bmp" in str(excinfo.value).lower() or "not supported" in str(excinfo.value).lower()


def test_corrupted_file_is_rejected() -> None:
    with pytest.raises(prep.UnsupportedImageError):
        prep.validate_upload(UploadedFileStub(b"this is definitely not an image", "photo.jpg"))


def test_empty_and_oversized_files_are_rejected() -> None:
    with pytest.raises(prep.UnsupportedImageError):
        prep.validate_upload(UploadedFileStub(b"", "photo.jpg"))
    with pytest.raises(prep.UnsupportedImageError) as excinfo:
        prep.validate_upload(make_image(fmt="PNG"), filename="big.png", max_mb=0.0001)
    assert "larger than" in str(excinfo.value)


def test_missing_path_is_reported(tmp_path) -> None:
    with pytest.raises(prep.UnsupportedImageError):
        prep.validate_upload(tmp_path / "ghost.jpg")


def test_decompression_bomb_guard_is_configured() -> None:
    assert Image.MAX_IMAGE_PIXELS == IMAGE_MAX_PIXELS == 40_000_000


# --------------------------------------------------------------------------- #
# Loading & preprocessing
# --------------------------------------------------------------------------- #
def test_exif_orientation_is_applied() -> None:
    """A photo stored with orientation=6 must come back upright."""
    image = Image.new("RGB", (120, 60), (200, 30, 30))
    exif = Image.Exif()
    exif[274] = 6                            # 274 = Orientation, 6 = rotate 90° CW
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    loaded = prep.load_image(buffer.getvalue())
    assert loaded.size == (60, 120)          # dimensions swapped by the EXIF rotation


def test_preprocess_output_shape_dtype_and_range() -> None:
    batch = prep.preprocess(Image.new("RGB", (640, 480), (120, 120, 120)))
    assert batch.shape == (1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all()
    # A mid-grey image normalised with ImageNet stats stays in a sane range.
    assert -4.0 < batch.min() and batch.max() < 4.0


def test_preprocess_is_deterministic_and_handles_tiny_images() -> None:
    image = Image.new("RGB", (8, 8), (10, 200, 10))     # smaller than the crop size
    first = prep.preprocess(image)
    second = prep.preprocess(image)
    assert np.array_equal(first, second)


def test_preprocess_converts_modes_to_rgb() -> None:
    grey = Image.new("L", (300, 300), 128)
    rgba = Image.new("RGBA", (300, 300), (10, 20, 30, 255))
    assert prep.preprocess(grey).shape == (1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    assert prep.preprocess(rgba).shape == (1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)


def test_preprocess_upload_returns_image_and_batch() -> None:
    upload = UploadedFileStub(make_image(fmt="PNG"), "dish.png", "image/png")
    image, batch = prep.preprocess_upload(upload)
    assert isinstance(image, Image.Image) and image.mode == "RGB"
    assert batch.shape == (1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)


# --------------------------------------------------------------------------- #
# Labels / tiers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("chicken_curry", "Chicken curry"),
        ("pizza", "Pizza"),
        ("french_fries", "French fries"),
        ("creme_brulee", "Crème brûlée"),
        ("ice_cream", "Ice cream"),
        ("zucchini_boats", "Zucchini boats"),
    ],
)
def test_format_label(raw: str, expected: str) -> None:
    assert format_label(raw) == expected


def test_confidence_tiers_follow_the_configured_thresholds() -> None:
    assert confidence_tier(CONFIDENCE_HIGH) == "high"
    assert confidence_tier(CONFIDENCE_HIGH - 0.1) == "medium"
    assert confidence_tier(0.0) == "low"
    assert confidence_tier(100.0) == "high"


# --------------------------------------------------------------------------- #
# Predictor logic
# --------------------------------------------------------------------------- #
def test_predictor_ranks_classes_and_reports_confidence() -> None:
    predictor = FoodPredictor(model=StubModel([85, 20, 10, 0, 99]), top_k=5)
    result = predictor.predict(Image.new("RGB", (400, 300), (200, 150, 90)))

    assert result.raw_label == "samosa"
    assert result.label == "Samosa"
    assert 0 < result.confidence <= 100
    assert len(result.alternatives) == 5
    assert result.alternatives[0][0] == "Samosa"
    scores = [score for _, score in result.alternatives]
    assert scores == sorted(scores, reverse=True)
    assert result.backend == "stub" and result.inference_ms >= 0


def test_predictor_flags_low_confidence_as_uncertain() -> None:
    flat = StubModel(flat=True)          # every class equal → ~1 % confidence
    result = FoodPredictor(model=flat).predict(Image.new("RGB", (300, 300)))
    assert result.tier == "low"
    assert result.is_uncertain is True
    assert "confirm" in result.explanation().lower()


def test_predictor_flags_narrow_margins_even_with_medium_confidence() -> None:
    class CloseCall(StubModel):
        def predict_proba(self, batch: np.ndarray) -> np.ndarray:
            scores = np.full((1, FOOD_NUM_CLASSES), 1e-6)
            scores[0, 85], scores[0, 20] = 0.45, 0.44     # 1 point apart
            return scores / scores.sum()

    result = FoodPredictor(model=CloseCall()).predict(Image.new("RGB", (256, 256)))
    assert result.confidence > 40              # medium, not low
    assert result.margin < 10
    assert result.is_uncertain is True         # ...but still not trusted silently


def test_predictor_raises_when_no_model_available(monkeypatch) -> None:
    """No model → an exception, never an invented label."""
    def boom(_backend: str = "auto", force_reload: bool = False):
        raise ModelUnavailableError("simulated: backend missing")

    monkeypatch.setattr("food_recognition.predictor.load_food_model", boom)
    with pytest.raises(ModelUnavailableError):
        FoodPredictor().predict(Image.new("RGB", (300, 300)))


def test_predictor_describe_reports_failure_instead_of_raising(monkeypatch) -> None:
    def boom(_backend: str = "auto", force_reload: bool = False):
        raise ModelUnavailableError("simulated failure")

    monkeypatch.setattr("food_recognition.predictor.load_food_model", boom)
    described = FoodPredictor().describe()
    assert described["Backend"] == "unavailable"
    assert "simulated failure" in described["Reason"]


def test_prediction_roundtrip_dict() -> None:
    prediction = FoodPredictor(model=StubModel()).predict(Image.new("RGB", (320, 240)))
    restored = FoodPrediction.from_dict(prediction.to_dict())
    assert restored == prediction


# --------------------------------------------------------------------------- #
# Sanity check (the guard that caught the broken ONNX artifact)
# --------------------------------------------------------------------------- #
def test_sanity_check_accepts_a_valid_model() -> None:
    sanity_check(StubModel().predict_proba, "stub")


def test_sanity_check_rejects_degenerate_model() -> None:
    """All-zero logits (uniform output) must be rejected, not silently used."""
    def degenerate(_batch):
        return np.full((1, FOOD_NUM_CLASSES), 1.0 / FOOD_NUM_CLASSES)

    with pytest.raises(ModelUnavailableError) as excinfo:
        sanity_check(degenerate, "broken-onnx")
    assert "same score for every class" in str(excinfo.value)


def test_sanity_check_rejects_wrong_shape_and_non_finite() -> None:
    with pytest.raises(ModelUnavailableError):
        sanity_check(lambda _b: np.zeros((1, 7)), "wrong-shape")

    def not_finite(_batch):
        scores = np.full((1, FOOD_NUM_CLASSES), np.nan)
        return scores

    with pytest.raises(ModelUnavailableError):
        sanity_check(not_finite, "nan-model")


# --------------------------------------------------------------------------- #
# Real-model integration (skipped when unavailable)
# --------------------------------------------------------------------------- #
def _real_backend_ready() -> bool:
    try:
        import importlib.util

        if find_onnx_path() is not None:
            return True
        return importlib.util.find_spec("torch") is not None
    except Exception:  # pragma: no cover
        return False


@pytest.mark.skipif(not _real_backend_ready(), reason="no ONNX file and no PyTorch installed")
def test_real_model_loads_and_predicts_a_valid_distribution() -> None:
    """Integration test: load the real model once and check output sanity.

    This asserts *behaviour* (a valid probability distribution over 101 classes
    with a non-degenerate spread), not a specific label, because accuracy on
    arbitrary web photos varies — the documented per-image results live in the
    README.
    """
    from food_recognition.model_loader import load_food_model

    model = load_food_model("auto")
    batch = prep.preprocess(Image.new("RGB", (512, 512), (200, 120, 60)))
    probabilities = model.predict_proba(batch)

    assert probabilities.shape == (1, FOOD_NUM_CLASSES)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-4)
    assert probabilities.std() > 1e-4                     # not degenerate
    prediction = FoodPredictor(model=model).predict(Image.new("RGB", (512, 512), (200, 120, 60)))
    assert prediction.raw_label in load_labels()
    assert 0.0 < prediction.confidence <= 100.0
