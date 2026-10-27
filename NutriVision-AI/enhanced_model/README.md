# Enhanced Food Classifier (Model 2)

A second, independently trained food classifier that sits **next to** the original Food-101 MobileNetV2 model
in NutriVision AI. The original model, its labels, preprocessing and weights are untouched
(`tests/test_original_model.py` hashes them). The comparison UI is `pages/7_Model_Comparison.py`.

## Status (read this first)

| Item | State |
|---|---|
| Code: dataset prep, training, evaluation, inference, UI, tests | **Written and exercised** (see "What was actually run") |
| Real dataset (Food-101) | **Not downloaded** in the build sandbox (host not reachable) |
| Real trained checkpoint / pretrained weights | **Not produced.** No ImageNet weights could be fetched either |
| Real accuracy numbers for the enhanced model | **None yet.** Nothing in this repo claims any |

Until you run training, the page shows "Enhanced model not available" and the original model keeps working.

## Existing model (Model 1)
torchvision MobileNetV2 fine-tuned on Food-101 (101 classes, 76.3 % top-1 on the Food-101 test split), shipped as
ONNX (`assets/models/*.onnx`), 224x224 input, ImageNet statistics. Measured on `image-search/` before any change:

| Image | Actual | Original prediction | Confidence |
|---|---|---|---|
| margherita-pizza-…-1 | pizza | sushi | 41.47 % |
| margherita-pizza-…-2 | pizza | bruschetta | 47.93 % |
| plate-of-samosa-…-1 | samosa | chicken_wings | 11.14 % |
| plate-of-samosa-…-2 | samosa | samosa | 66.52 % |

## Enhanced model (Model 2) — design
* **Architecture:** EfficientNet-B0 (torchvision), ImageNet-pretrained, new head (Dropout 0.3 + Linear). ~5.3 M parameters,
  ~16 MB as ONNX; chosen over ResNet-50 / ConvNeXt-T because it trains on CPU/small GPUs and serves fast on CPU.
  Not assumed better: only `evaluate.py` can show that.
* **Classes (configurable):** pizza, samosa, sushi, bruschetta, chicken_wings (priority) plus hamburger, french_fries,
  club_sandwich, spaghetti_bolognese, greek_salad, fried_rice, dumplings, spring_rolls, tacos, pancakes. Edit
  `TARGET_CLASSES` in `config.py` or set `ENHANCED_CLASSES=a,b,c`. Classes absent from the dataset are skipped with a warning.
  (Food-101 folder names; e.g. "burger" is `hamburger`, "pasta" is covered by `spaghetti_bolognese`, "sandwich" by `club_sandwich`.)
* **Training:** stage 1 head only (3 epochs, AdamW 1e-3), stage 2 full fine-tune (≤12 epochs, AdamW 1e-4, cosine decay), label
  smoothing 0.1, early stopping on **validation** accuracy, seed 42, resumable (`--resume`).
* **Augmentation:** random-resized-crop (0.5–1.0), horizontal flip, ±10° rotation, mild brightness/contrast/saturation.
* **Inference:** training writes both `checkpoints/enhanced_food_classifier.onnx` and a single-file bundle `checkpoints/enhanced_food_classifier.pkl` (ONNX bytes + classes + preprocessing + thresholds; the app prefers the `.pkl`). The bundle is loaded with a restricted unpickler that refuses any code object, but still only load `.pkl` files you trained yourself. Exported to ONNX and run with onnxruntime (already a project dependency). Own preprocessing
  (`artifacts/preprocessing_config.json`), own class mapping, own thresholds. Transparent PNG → white background;
  grayscale → RGB; EXIF orientation applied.

## Dataset and splits
Source: Food-101 (ETH Zurich), class subset. Licence: non-commercial research use of images scraped from foodspotting.com —
verify before redistributing. Images are **not copied**; `datasets/enhanced_food_dataset/manifest.csv` lists them.

* **Test** = Food-101's official test images (250/class), never used for training or model selection, and also unseen by the
  original model, so the two models can be compared fairly.
* **Train / validation** = official train images (750/class) split 85/15 per class, fixed seed.
* This differs from a 70/15/15 pool split on purpose: pooling would put the original model's training images in the test set.
* Near-duplicates (64-bit dHash, Hamming ≤ 4) are removed: train images duplicating a test image are dropped, and duplicates
  inside train are dropped, so none cross splits. Hashing is not exhaustive; Food-101 labels are noisy by design.

## Commands
```bash
pip install -r requirements.txt -r requirements-training.txt     # CPU torch first, see requirements-training.txt
# 1. dataset (≈4.7 GB download) — or point at an existing extracted food-101/ folder
python -m enhanced_model.prepare_dataset --download --food101-root ./datasets/raw
#    python -m enhanced_model.prepare_dataset --food101-root /path/to/food-101
# 2. train (writes checkpoints/*.onnx and artifacts/*.json)
python -m enhanced_model.train                  # add --resume to continue
# 3. evaluate on the held-out test split (+ original model on the same images)
python -m enhanced_model.evaluate
# 4. the four sample photos, both models -> results/sample_image_predictions.csv
python -m enhanced_model.evaluate_samples
# 5. app
streamlit run app.py          # open "Model comparison" in the sidebar
# tests
pytest tests/test_original_model.py tests/test_enhanced_model.py tests/test_streamlit_integration.py
```
Run everything from the `NutriVision-AI/` folder. Env vars: `ENHANCED_CLASSES`, `ENHANCED_DATASET_DIR`, `ENHANCED_NUM_WORKERS`.

## Evaluation outputs (`results/`)
`evaluation_metrics.json` (top-1, top-3, macro P/R/F1, per-class P/R/F1/support, expected calibration error, mean inference ms,
model size), `classification_report.json`, `confusion_matrix.csv/.png`, `per_class_metrics.csv`,
`original_vs_enhanced_test.json` (original as shipped over 101 classes **and** restricted to the enhanced classes),
`sample_image_predictions.csv` + `sample_image_comparison.md`.
The four sample images are a *diagnostic set*, not evidence of general accuracy. Do not tune on them.

## Uncertainty and confidence
Scores are softmax outputs, labelled "Model confidence", not guaranteed probabilities (ECE is reported). Training writes
`artifacts/uncertainty_thresholds.json`, choosing the lowest confidence threshold at which accepted **validation** predictions
reach 90 % accuracy; if unreachable, documented defaults (50 % / 10-point margin) are kept and flagged. A warning is advice,
not proof of an unknown food; no unknown-food detector has been trained.

## What was actually run in the build sandbox
* Original model re-run on the four images: matches the table above exactly.
* Full pipeline smoke test (prepare → train → ONNX export + PyTorch/ONNX parity → evaluate → resume → predict) on **synthetic
  random images, 3 classes, no pretrained weights**, written to a temp folder. It proves the code runs, nothing more.
* Tests: 19 new tests pass; the existing suite gives the same 35 failures with and without these changes (missing `xgboost`
  and `google-genai` in the sandbox) — no regressions.

## Troubleshooting
* "Enhanced model not available": train first, or copy `enhanced_food_classifier.onnx` + `artifacts/*.json` into place.
* "different training runs": `class_mapping.json` and the ONNX file come from different trainings — re-export together.
* Download errors: place a Food-101 copy locally and use `--food101-root`.
* Slow training on CPU: reduce `--finetune-epochs` or classes; a GPU is used automatically if present.

## Known limitations / future work
Closed-set classifier; Food-101 only (studio-style web photos, not phone photos of mixed plates); no calibration beyond
thresholding; no multi-dish detection. Ideas: add phone-photo data with usable licences, temperature scaling, an
"other food" class, `st.cache_resource` wrapping if load time matters (currently `lru_cache`).
