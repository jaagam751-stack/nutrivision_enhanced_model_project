"""Integration tests for pages/7_Model_Comparison.py via streamlit.testing.v1.AppTest."""
from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from enhanced_model import config as ecfg  # noqa: E402
from enhanced_model import predict as epredict  # noqa: E402
from tests._enhanced_fixture import build_fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "pages" / "7_Model_Comparison.py")
SAMPLES = ROOT.parent / "image-search"
pytestmark = pytest.mark.skipif(not SAMPLES.is_dir(), reason="image-search/ not present")


def point_enhanced_to(monkeypatch, paths):
    monkeypatch.setattr(ecfg, "ONNX_MODEL_PATH", paths["model"])
    monkeypatch.setattr(ecfg, "CLASS_MAPPING_PATH", paths["mapping"])
    monkeypatch.setattr(ecfg, "PREPROCESSING_PATH", paths["pre"])
    monkeypatch.setattr(ecfg, "THRESHOLDS_PATH", paths["thresholds"])
    epredict._cached_predictor.cache_clear()


def run_sample(at: AppTest, mode_index: int) -> AppTest:
    at.radio[0].set_value(at.radio[0].options[mode_index])
    at.radio[1].set_value("Use a sample image")
    at.run()
    at.selectbox[0].set_value("margherita-pizza-whole-food-photo-1.jpg")
    at.run()
    [b for b in at.button if b.label.endswith("Run")][0].click()
    at.run()
    return at


def test_page_survives_missing_enhanced_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(ecfg, "ONNX_MODEL_PATH", tmp_path / "missing.onnx")
    epredict._cached_predictor.cache_clear()
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not at.exception
    assert any("Enhanced model not available" in w.value for w in at.warning)
    at = run_sample(at, 0)                       # Original Model mode still works
    assert not at.exception
    assert any(m.label == "Predicted food" and m.value == "Sushi" for m in at.metric)   # unchanged original output


def test_compare_mode_labels_each_model(monkeypatch, tmp_path):
    point_enhanced_to(monkeypatch, build_fixture(tmp_path))
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    at = run_sample(at, 2)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Original Model" in text and "Enhanced Model" in text
    assert len(at.session_state["nv_model_comparison_history"]) == 1
    row = at.session_state["nv_model_comparison_history"][0]
    assert row["original_prediction"] == "Sushi" and row["enhanced_prediction"] != ""


def test_enhanced_only_mode_never_uses_original(monkeypatch, tmp_path):
    point_enhanced_to(monkeypatch, build_fixture(tmp_path))
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    at = run_sample(at, 1)
    row = at.session_state["nv_model_comparison_history"][0]
    assert row["original_prediction"] == "" and row["enhanced_prediction"] != ""
