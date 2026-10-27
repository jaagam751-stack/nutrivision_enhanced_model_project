"""Evaluate the enhanced model on the isolated TEST split (through the same ONNX path the app uses).

    python -m enhanced_model.evaluate [--no-compare-original]

Writes to enhanced_model/results/: evaluation_metrics.json, classification_report.json,
confusion_matrix.csv (+ .png when matplotlib is installed), per_class_metrics.csv, and
original_vs_enhanced_test.json (same images, same labels, both models).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from enhanced_model import config as cfg
from enhanced_model.predict import EnhancedPredictor, load_rgb


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    conf, correct = probs.max(1), (probs.argmax(1) == labels)
    edges, ece = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def original_probs(image: Image.Image):
    """Original pipeline, used read-only (its own preprocessing, labels and ONNX file)."""
    from food_recognition.model_loader import load_food_model, load_labels
    from food_recognition.preprocessing import preprocess
    model = load_food_model("onnx")
    return model.predict_proba(preprocess(image))[0], list(load_labels())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    ap.add_argument("--dataset-metadata", type=Path, default=cfg.DATASET_METADATA_PATH)
    ap.add_argument("--checkpoint-dir", type=Path, default=cfg.CHECKPOINT_DIR)
    ap.add_argument("--artifact-dir", type=Path, default=cfg.ARTIFACT_DIR)
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    ap.add_argument("--no-compare-original", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=0)
    args = ap.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(args.dataset_metadata.read_text())
    root = Path(meta["source_root"])
    bundle = args.checkpoint_dir / cfg.PICKLE_MODEL_PATH.name
    predictor = EnhancedPredictor(bundle if bundle.is_file() else args.checkpoint_dir / cfg.ONNX_MODEL_PATH.name,
                                  args.artifact_dir / cfg.CLASS_MAPPING_PATH.name,
                                  args.artifact_dir / cfg.PREPROCESSING_PATH.name,
                                  args.artifact_dir / cfg.THRESHOLDS_PATH.name)
    classes = predictor.classes
    index = {c: i for i, c in enumerate(classes)}
    with open(args.manifest, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    if args.limit_per_class:
        seen: dict[str, int] = {}
        rows = [r for r in rows if not (seen.__setitem__(r["class"], seen.get(r["class"], 0) + 1)
                                        or seen[r["class"]] > args.limit_per_class)]

    probs, labels, times = [], [], []
    orig_full, orig_sub, orig_times = [], [], []
    compare = not args.no_compare_original
    if compare:
        try:
            sys.path.insert(0, str(cfg.PROJECT_DIR))
            original_probs(Image.new("RGB", (64, 64)))
        except Exception as exc:  # noqa: BLE001
            print(f"Original model not usable for comparison ({exc}); skipping.", file=sys.stderr)
            compare = False
    for r in rows:
        image = load_rgb(root / r["path"])
        t0 = time.perf_counter(); p = predictor.predict_proba(image); times.append((time.perf_counter() - t0) * 1000)
        probs.append(p); labels.append(index[r["class"]])
        if compare:
            t0 = time.perf_counter(); op, olabels = original_probs(image); orig_times.append((time.perf_counter() - t0) * 1000)
            truth = r["class"]
            orig_full.append(olabels[int(op.argmax())] == truth)
            sub_ids = [olabels.index(c) for c in classes]
            orig_sub.append(classes[int(np.argmax(op[sub_ids]))] == truth)
    P, y = np.array(probs), np.array(labels)
    pred = P.argmax(1)

    top3 = float(np.mean([yy in np.argsort(-pp)[:3] for pp, yy in zip(P, y)])) if len(classes) >= 3 else None
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    macro = precision_recall_fscore_support(y, pred, labels=range(len(classes)), average="macro", zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    onnx_size = predictor.model_path.stat().st_size
    training_cfg = {}
    tc = args.artifact_dir / cfg.TRAINING_CONFIG_PATH.name
    if tc.is_file():
        training_cfg = json.loads(tc.read_text())

    metrics = {
        "split": "test", "n_images": int(len(y)), "n_classes": len(classes), "classes": classes,
        "top1_accuracy": float(accuracy_score(y, pred)), "top3_accuracy": top3,
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "expected_calibration_error": expected_calibration_error(P, y),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                          "support": int(support[i])} for i, c in enumerate(classes)},
        "avg_inference_ms_model_only": float(np.mean(times)), "model_size_mb": round(onnx_size / 1e6, 2),
        "architecture": training_cfg.get("architecture", cfg.ARCHITECTURE),
        "pretrained": training_cfg.get("pretrained"), "trained_at": training_cfg.get("trained_at"),
        "dataset_version": meta.get("dataset_version"), "dataset_counts": meta.get("totals"),
        "confidence_note": "Softmax scores are not guaranteed probabilities; see expected_calibration_error.",
    }
    (args.results_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.results_dir / "classification_report.json").write_text(json.dumps(
        classification_report(y, pred, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0), indent=2))
    with open(args.results_dir / "confusion_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["true\\pred", *classes])
        for c, row in zip(classes, cm):
            w.writerow([c, *row.tolist()])
    with open(args.results_dir / "per_class_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class", "support", "precision", "recall", "f1"])
        for i, c in enumerate(classes):
            w.writerow([c, int(support[i]), f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}"])
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=90); ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); fig.tight_layout()
        fig.savefig(args.results_dir / "confusion_matrix.png", dpi=120); plt.close(fig)
    except ImportError:
        pass

    if compare:
        comp = {"n_images": int(len(y)), "classes": classes,
                "enhanced_top1": metrics["top1_accuracy"],
                "original_top1_all_101_classes": float(np.mean(orig_full)),
                "original_top1_restricted_to_enhanced_classes": float(np.mean(orig_sub)),
                "original_avg_inference_ms_incl_preprocessing": float(np.mean(orig_times)),
                "note": ("Same test images and labels for both models. 'all_101' is the original model as shipped; "
                         "'restricted' only lets it choose among the enhanced model's classes (a fairer, closed-set "
                         "comparison). The original was trained on Food-101's train split; these are Food-101 test images.")}
        (args.results_dir / "original_vs_enhanced_test.json").write_text(json.dumps(comp, indent=2))
        print(json.dumps(comp, indent=2))
    print(f"top1 {metrics['top1_accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | n={len(y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
