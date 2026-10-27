"""Regression tests: the ORIGINAL model and pipeline must be exactly as shipped."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = ROOT.parent / "image-search"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_artifacts_unchanged():
    expected = json.loads((FIXTURES / "original_artifact_hashes.json").read_text())
    for rel, digest in expected.items():
        assert sha256(ROOT / rel) == digest, f"{rel} was modified"


def test_original_class_mapping_preserved():
    from food_recognition import load_labels
    labels = load_labels()
    assert len(labels) == 101
    assert labels[0] == "apple_pie" and "pizza" in labels and "samosa" in labels and labels[-1] == "waffles"


def test_original_preprocessing_shape_and_determinism():
    from food_recognition.preprocessing import preprocess
    img = Image.new("RGB", (300, 200), (120, 60, 30))
    a, b = preprocess(img), preprocess(img)
    assert a.shape == (1, 3, 224, 224) and a.dtype == np.float32
    assert np.array_equal(a, b)


@pytest.mark.skipif(not SAMPLES.is_dir(), reason="image-search/ not present")
def test_original_predictions_match_recorded_baseline():
    from food_recognition import FoodPredictor
    baseline = json.loads((FIXTURES / "original_baseline_predictions.json").read_text())
    predictor = FoodPredictor(backend="onnx")
    for name, expected in baseline.items():
        result = predictor.predict(Image.open(SAMPLES / name))
        assert result.raw_label == expected["label"], name
        assert result.confidence == pytest.approx(expected["confidence"], abs=0.01), name
