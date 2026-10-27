"""Run BOTH models on the images in image-search/ and save a comparison (real inference only).

    python -m enhanced_model.evaluate_samples

Ground truth comes from the file name (keyword -> Food-101 class), configured in
``SAMPLE_KEYWORDS`` below or an optional ``labels.csv`` (columns: filename,actual) in the folder.
These four photos are a *diagnostic* set, not a test set.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

from enhanced_model import config as cfg
from enhanced_model.predict import EnhancedModelUnavailableError, load_enhanced_predictor

SAMPLE_KEYWORDS = {"pizza": "pizza", "samosa": "samosa", "sushi": "sushi", "bruschetta": "bruschetta", "wings": "chicken_wings"}


def actual_class(path: Path, overrides: dict[str, str]) -> str:
    if path.name in overrides:
        return overrides[path.name]
    for key, cls in SAMPLE_KEYWORDS.items():
        if key in path.name.lower():
            return cls
    return ""


def run(sample_dir: Path = cfg.SAMPLE_IMAGE_DIR, out_dir: Path = cfg.RESULTS_DIR) -> list[dict]:
    sys.path.insert(0, str(cfg.PROJECT_DIR))
    from food_recognition import FoodPredictor            # original pipeline, unchanged
    original = FoodPredictor(backend="onnx")
    enhanced = load_enhanced_predictor()
    overrides = {}
    if (sample_dir / "labels.csv").is_file():
        with open(sample_dir / "labels.csv", newline="") as fh:
            overrides = {r["filename"]: r["actual"] for r in csv.DictReader(fh)}
    rows = []
    for path in sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        actual = actual_class(path, overrides)
        image = Image.open(path)
        o = original.predict(image)
        e = enhanced.predict(image)
        rows.append({
            "image": path.name, "actual": actual,
            "original_prediction": o.raw_label, "original_confidence": o.confidence,
            "original_correct": (o.raw_label == actual) if actual else "",
            "original_inference_ms": o.inference_ms,
            "enhanced_prediction": e.raw_label, "enhanced_confidence": e.confidence,
            "enhanced_top3": "; ".join(f"{l} {s:.1f}%" for l, s in e.top_k),
            "enhanced_correct": (e.raw_label == actual) if actual else "",
            "enhanced_inference_ms": e.inference_ms,
            "enhanced_uncertain": e.is_uncertain,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sample_image_predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["| Image | Actual | Original | Orig. conf. | Enhanced | Enh. conf. | Enhanced correct? |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['image']} | {r['actual']} | {r['original_prediction']} | {r['original_confidence']:.1f}% | "
                     f"{r['enhanced_prediction']} | {r['enhanced_confidence']:.1f}% | {r['enhanced_correct']} |")
    (out_dir / "sample_image_comparison.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    try:
        for row in run():
            print(row)
    except EnhancedModelUnavailableError as exc:
        raise SystemExit(f"Enhanced model unavailable: {exc}")
