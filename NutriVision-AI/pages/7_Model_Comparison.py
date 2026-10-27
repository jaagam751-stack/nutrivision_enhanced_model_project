"""Model Comparison — original Food-101 classifier vs the separately trained enhanced classifier.

This page is additive: it calls the original ``FoodPredictor`` exactly as the Upload Meal page does,
and the enhanced model through its own independent inference path. Nothing is saved to the database.
"""
from __future__ import annotations

import io
import json
import logging

import pandas as pd
import streamlit as st

from config import ALLOWED_IMAGE_TYPES, APP_BUILD, APP_FOOTER_NOTE, APP_ICON, APP_NAME, CURRENT_PHASE, MAX_UPLOAD_MB
from enhanced_model import config as ecfg
from enhanced_model.compare import (ENHANCED, ENHANCED_NAME, ORIGINAL, ORIGINAL_NAME, agree,
                                    enhanced_model_status, history_row, run_enhanced, run_original)
from enhanced_model.evaluate_samples import actual_class
from food_recognition import UnsupportedImageError
from food_recognition.preprocessing import validate_upload
from utils import ui
from utils.helpers import init_session_state
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
st.set_page_config(page_title=f"{APP_NAME} · Model Comparison", page_icon=APP_ICON, layout="wide")

HISTORY_KEY = "nv_model_comparison_history"
MODES = ("Original Model", "Enhanced Model", "Compare Both Models")


def render_result(title: str, res: dict) -> None:
    ui.section(title, icon="🧠")
    if not res.get("ok"):
        st.error(res.get("error", "No result."))
        return
    st.metric("Predicted food", res["label"])
    st.metric("Model confidence", f"{res['confidence']:.1f}%",
              help="A softmax score from this model, not a guaranteed probability that it is right.")
    if res["uncertain"]:
        why = "; ".join(res["reasons"]) or "low confidence or a small gap to the runner-up"
        st.warning(f"The model is uncertain about this prediction ({why}). "
                   "Try another image with the food clearly visible.")
    st.caption("Top predictions")
    for label, score in res["top"]:
        st.progress(min(1.0, score / 100), text=f"{label} — {score:.1f}%")
    st.caption(f"Inference {res['inference_ms']:.0f} ms (model only)")


def render_enhanced_status() -> bool:
    status = enhanced_model_status()
    if status["available"]:
        return True
    st.warning(
        "**Enhanced model not available yet.** Missing: " + ", ".join(f"`{m}`" for m in status["missing"]) +
        ". The original model still works. Train the enhanced model with "
        "`python -m enhanced_model.prepare_dataset` and `python -m enhanced_model.train` "
        "(see `enhanced_model/README.md`)."
    )
    return False


def render_performance() -> None:
    ui.section("Enhanced Model: Dataset and Performance",
               "Read from the real training / evaluation files — nothing here is typed in by hand.", icon="📊")
    metrics_path = ecfg.RESULTS_DIR / "evaluation_metrics.json"
    meta_path = ecfg.DATASET_METADATA_PATH
    cfg_path = ecfg.TRAINING_CONFIG_PATH
    if not metrics_path.is_file():
        st.info("No evaluation results yet. Run `python -m enhanced_model.evaluate` after training; "
                "metrics will appear here once that has actually completed.")
        return
    m = json.loads(metrics_path.read_text())
    if cfg_path.is_file() and json.loads(cfg_path.read_text()).get("pretrained") is False:
        st.warning("This checkpoint was trained WITHOUT pretrained weights (smoke test). Do not treat it as a real model.")
    cols = st.columns(4)
    cols[0].metric("Test top-1", f"{m['top1_accuracy'] * 100:.1f}%")
    cols[1].metric("Macro F1", f"{m['macro_f1']:.3f}")
    cols[2].metric("Macro precision / recall", f"{m['macro_precision']:.3f} / {m['macro_recall']:.3f}")
    cols[3].metric("Test images", f"{m['n_images']}")
    cols = st.columns(4)
    cols[0].metric("Architecture", m.get("architecture", "—"))
    cols[1].metric("Classes", m["n_classes"])
    cols[2].metric("Model size", f"{m['model_size_mb']} MB")
    cols[3].metric("Avg inference", f"{m['avg_inference_ms_model_only']:.0f} ms")
    if m.get("top3_accuracy") is not None:
        st.caption(f"Top-3 accuracy {m['top3_accuracy'] * 100:.1f}% · expected calibration error "
                   f"{m['expected_calibration_error']:.3f} · trained {m.get('trained_at') or 'n/a'}")
    per = pd.DataFrame(m["per_class"]).T.reset_index().rename(columns={"index": "class"})
    try:
        import plotly.express as px
        st.plotly_chart(px.bar(per, x="class", y="recall", title="Per-class recall (test set)", range_y=[0, 1]),
                        use_container_width=True)
    except Exception:  # noqa: BLE001
        st.bar_chart(per.set_index("class")["recall"])
    st.dataframe(per, use_container_width=True, hide_index=True)
    cm_path = ecfg.RESULTS_DIR / "confusion_matrix.csv"
    if cm_path.is_file():
        cm = pd.read_csv(cm_path, index_col=0)
        with st.expander("Confusion matrix (rows = actual, columns = predicted)"):
            st.dataframe(cm, use_container_width=True)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        counts = pd.DataFrame(meta["counts"]).fillna(0).astype(int)
        with st.expander("Dataset distribution"):
            st.caption(f"{meta['source']} · version {meta['dataset_version']} · {meta['split_policy']}")
            st.dataframe(counts, use_container_width=True)
    comp_path = ecfg.RESULTS_DIR / "original_vs_enhanced_test.json"
    if comp_path.is_file():
        c = json.loads(comp_path.read_text())
        st.caption("Both models on the same held-out test images:")
        st.table(pd.DataFrame({
            "Model": ["Original (101 classes, as shipped)", "Original (restricted to enhanced classes)", "Enhanced"],
            "Top-1 accuracy": [f"{c['original_top1_all_101_classes'] * 100:.1f}%",
                               f"{c['original_top1_restricted_to_enhanced_classes'] * 100:.1f}%",
                               f"{c['enhanced_top1'] * 100:.1f}%"]}))
        st.caption(c["note"])


def main() -> None:
    init_session_state()
    ui.load_css()
    ui.hero(title="Model comparison",
            subtitle="Run the original Food-101 classifier, the separately trained enhanced classifier, or both on "
                     "the same photo. Results are never merged and each is labelled with the model that produced it.",
            eyebrow="Original vs Enhanced", icon="⚖️")
    enhanced_ok = render_enhanced_status()
    mode = st.radio("Model", MODES, horizontal=True, index=2 if enhanced_ok else 0)

    source = st.radio("Image source", ("Upload a photo", "Use a sample image"), horizontal=True)
    image_bytes, name, actual = None, "", ""
    if source == "Upload a photo":
        up = st.file_uploader("Choose a food photo", type=list(ALLOWED_IMAGE_TYPES),
                              help=f"JPG, JPEG or PNG up to {MAX_UPLOAD_MB:g} MB.")
        if up is not None:
            try:
                validate_upload(up, filename=up.name)
                image_bytes, name = up.getvalue(), up.name
            except UnsupportedImageError as exc:
                st.error(str(exc))
        classes = ["unknown"] + sorted(set(ecfg.TARGET_CLASSES))
        actual = st.selectbox("Actual food (optional, for the history table)", classes)
        actual = "" if actual == "unknown" else actual
    else:
        samples = sorted(p for p in ecfg.SAMPLE_IMAGE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if ecfg.SAMPLE_IMAGE_DIR.is_dir() else []
        if not samples:
            st.info(f"No sample images found in {ecfg.SAMPLE_IMAGE_DIR}.")
        else:
            choice = st.selectbox("Sample image", [p.name for p in samples])
            path = next(p for p in samples if p.name == choice)
            image_bytes, name, actual = path.read_bytes(), path.name, actual_class(path, {})
            if actual:
                st.caption(f"Actual food (from file name): **{actual}**")

    if image_bytes:
        st.image(image_bytes, caption=name, width=360)
    if st.button("🔍 Run", type="primary", disabled=image_bytes is None):
        want_o = mode in (MODES[0], MODES[2])
        want_e = mode in (MODES[1], MODES[2])
        with st.spinner("Running model(s)…"):
            orig = run_original(io.BytesIO(image_bytes)) if want_o else None
            enh = run_enhanced(io.BytesIO(image_bytes)) if want_e else None
        cols = st.columns(2 if (orig and enh) else 1, gap="large")
        if orig:
            with cols[0]:
                render_result(f"Original Model — {ORIGINAL_NAME}", orig)
        if enh:
            with cols[-1]:
                render_result(f"Enhanced Model — {ENHANCED_NAME}", enh)
        verdict = agree(orig, enh)
        if verdict is True:
            st.success("Both models predict the same class.")
        elif verdict is False:
            st.info("The models disagree. Both results are shown as produced; neither is assumed correct.")
        if orig or enh:
            st.session_state.setdefault(HISTORY_KEY, []).append(history_row(name, actual, orig, enh))

    history = st.session_state.get(HISTORY_KEY, [])
    ui.section("Prediction history (this session)", icon="🗂️")
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()
    else:
        st.caption("Run a prediction to start the history.")

    render_performance()
    st.caption("Confidence is the model's own softmax score. It is not calibrated probability of correctness, "
               "and a low score does not prove the food is outside the model's classes.")
    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


main()
