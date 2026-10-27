"""Unit tests for the enhanced model's independent inference path (uses a synthetic ONNX test double)."""
from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image

from enhanced_model.predict import (EnhancedImageError, EnhancedModelUnavailableError, EnhancedPredictor)
from tests._enhanced_fixture import CLASSES, build_fixture


@pytest.fixture()
def predictor(tmp_path):
    p = build_fixture(tmp_path)
    return EnhancedPredictor(p["model"], p["mapping"], p["pre"], p["thresholds"])


def photo(mode="RGB", size=(320, 240)):
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).convert(mode)


def test_loads_own_class_mapping(predictor):
    assert predictor.classes == CLASSES


def test_prediction_is_valid(predictor):
    r = predictor.predict(photo())
    assert r.raw_label in CLASSES and r.source == "enhanced"
    assert math.isfinite(r.confidence) and 0 <= r.confidence <= 100
    assert len(r.top_k) == 3 and r.top_k[0][1] == pytest.approx(r.confidence, abs=0.01)
    assert sum(s for _, s in predictor.predict(photo(), top_k=5).top_k) == pytest.approx(100, abs=0.1)
    assert r.top_k == sorted(r.top_k, key=lambda t: -t[1])


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_other_image_modes(predictor, mode):
    assert predictor.predict(photo(mode)).raw_label in CLASSES


def test_accepts_bytes_and_filelike(predictor):
    buf = io.BytesIO(); photo().save(buf, "PNG")
    assert predictor.predict(buf.getvalue()).raw_label in CLASSES
    buf.seek(0); assert predictor.predict(buf).raw_label in CLASSES


def test_invalid_images_raise_clear_errors(predictor):
    with pytest.raises(EnhancedImageError):
        predictor.predict(b"not an image")
    with pytest.raises(EnhancedImageError):
        predictor.predict(b"")
    with pytest.raises(EnhancedImageError):
        predictor.predict("/nonexistent/file.jpg")


def test_missing_checkpoint_is_reported_not_faked(tmp_path):
    p = build_fixture(tmp_path)
    with pytest.raises(EnhancedModelUnavailableError, match="checkpoint not found"):
        EnhancedPredictor(tmp_path / "missing.onnx", p["mapping"], p["pre"], p["thresholds"])


def test_missing_or_mismatched_mapping(tmp_path):
    p = build_fixture(tmp_path)
    with pytest.raises(EnhancedModelUnavailableError):
        EnhancedPredictor(p["model"], tmp_path / "nope.json", p["pre"], p["thresholds"])
    p["mapping"].write_text('{"classes": ["a", "b"]}')
    with pytest.raises(EnhancedModelUnavailableError, match="different training runs"):
        EnhancedPredictor(p["model"], p["mapping"], p["pre"], p["thresholds"])


def test_uncertainty_flag_uses_thresholds(tmp_path):
    p = build_fixture(tmp_path)
    p["thresholds"].write_text('{"min_confidence": 99.9, "min_margin": 0, "calibrated_on": "test"}')
    r = EnhancedPredictor(p["model"], p["mapping"], p["pre"], p["thresholds"]).predict(photo())
    assert r.is_uncertain and r.uncertainty_reasons


def test_does_not_depend_on_original_package():
    import enhanced_model.predict as mod
    assert "food_recognition" not in open(mod.__file__).read().replace("original ``food_recognition``", "")


def test_enhanced_preprocessing_independent_config(tmp_path):
    from enhanced_model.predict import preprocess_enhanced
    out = preprocess_enhanced(photo(), {"image_size": 128, "resize_size": 150, "mean": [0, 0, 0], "std": [1, 1, 1]})
    assert out.shape == (1, 3, 128, 128)


def test_pickle_bundle_roundtrip_matches_onnx(tmp_path):
    from enhanced_model.predict import save_bundle
    p = build_fixture(tmp_path)
    pkl = tmp_path / "bundle.pkl"
    save_bundle(pkl, p["model"], CLASSES, {"image_size": 224, "resize_size": 256, "mean": [0.485, 0.456, 0.406],
                                           "std": [0.229, 0.224, 0.225]},
                {"min_confidence": 50.0, "min_margin": 10.0, "calibrated_on": "t"}, {})
    a = EnhancedPredictor(p["model"], p["mapping"], p["pre"], p["thresholds"]).predict(photo())
    b = EnhancedPredictor(pkl).predict(photo())
    assert (a.raw_label, a.confidence) == (b.raw_label, b.confidence)


def test_pickle_bundle_refuses_code_execution(tmp_path):
    import os, pickle

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned.txt"),))

    bad = tmp_path / "evil.pkl"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(EnhancedModelUnavailableError):
        EnhancedPredictor(bad)
    assert not (tmp_path / "pwned.txt").exists()


def test_status_prefers_bundle(tmp_path, monkeypatch):
    from enhanced_model import config as c
    from enhanced_model.predict import enhanced_model_status
    pkl = tmp_path / "x.pkl"; pkl.write_bytes(b"x")
    monkeypatch.setattr(c, "PICKLE_MODEL_PATH", pkl)
    assert enhanced_model_status()["available"] is True
