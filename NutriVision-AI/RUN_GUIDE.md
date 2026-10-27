# NutriVision AI: How to Run the Project (Original + Enhanced Model)

Everything below is run from the `NutriVision-AI/` folder (the one containing `app.py`).
Python 3.10 to 3.12 is recommended.

---

## Fastest way (one command)

This installs everything, trains a small demo-sized enhanced model, evaluates it and opens the app.

**Windows:** double-click `run_quick.bat` (or run it in a terminal from the `NutriVision-AI/` folder).
**macOS / Linux:** `bash run_quick.sh`

What it does, in order (finished steps are skipped when you run it again):

1. Downloads Food-101 once if you don't have it (about 4.7 GB, so this is the slowest part, roughly 10 to 30 minutes on a normal connection).
   Already have it? Run `python quick_start.py --food101-root "D:\path\to\food-101"`.
2. Trains the **quick profile**: 5 classes (pizza, samosa, sushi, bruschetta, chicken_wings), 300 training images per class, 5 epochs.
   On an ordinary laptop CPU I'd expect roughly 20 to 40 minutes. That is an estimate, I could not time it here. A GPU makes it a few minutes.
3. Evaluates on held-out test images and runs both models on your four sample photos.
4. Launches the app. Open **Model comparison** in the sidebar.

Useful flags: `--skip-app` (don't launch), `--retrain` (train again), `--food101-root PATH`.

The quick model is a demo. Expect lower accuracy than a full run (`python -m enhanced_model.train`, Part B below).
Whatever it scores, report the numbers shown in the app / `enhanced_model/results/`, and do not describe it as better than the original unless those results show it.

---

## Part A: Run the app (original model only, about 5 minutes)

This works immediately. No training is needed and the original model ships in `assets/models/`.

```bash
cd NutriVision-AI
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create the demo database (SQLite, no account needed):

```bash
python database/init_db.py --sqlite --sample-data
```

Start the app:

```bash
streamlit run app.py
```

Open http://localhost:8501. In the sidebar you will see **Upload Meal** (original model, unchanged)
and the new **Model Comparison** page.

> The Model Comparison page does not need a profile or a database. The other pages do.

On the Model Comparison page before the enhanced model is trained, you will see
"Enhanced model not available". This is expected. The **Original Model** mode still works.

---

## Part B: Train the enhanced model (needed once)

### 1. Install training packages

CPU PyTorch first, then the extras:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-training.txt
```

(If you have an NVIDIA GPU, install the matching CUDA build of PyTorch instead. Training uses it automatically.)

### 2. Get Food-101 and build the dataset manifest

Option 1: let the script download it (about 4.7 GB, needs internet):

```bash
python -m enhanced_model.prepare_dataset --download --food101-root ./datasets/raw
```

Option 2: you already have Food-101 extracted (a folder containing `images/` and `meta/`):

```bash
python -m enhanced_model.prepare_dataset --food101-root /path/to/food-101
```

This creates `datasets/enhanced_food_dataset/manifest.csv` and `dataset_metadata.json`.
Check the printed counts per split and the number of duplicates removed.

To change the classes (default is 5 priority classes plus 10 extras):

```bash
# Windows PowerShell
$env:ENHANCED_CLASSES="pizza,samosa,sushi,bruschetta,chicken_wings"
# macOS / Linux
export ENHANCED_CLASSES=pizza,samosa,sushi,bruschetta,chicken_wings
```

Set it before steps 2, 3 and 4, and keep it the same in all of them.

### 3. Train

```bash
python -m enhanced_model.train
```

* Stage 1 trains the new head, stage 2 fine-tunes the whole network, with early stopping on validation accuracy.
* On CPU expect roughly 1 to 3 hours for 15 classes. A GPU takes minutes.
* If interrupted, continue with `python -m enhanced_model.train --resume`.
* For a quick trial first: `python -m enhanced_model.train --limit-per-class 50 --head-epochs 1 --finetune-epochs 1`
  (trial results are not meaningful, and you should retrain fully afterwards).

Output: `enhanced_model/checkpoints/enhanced_food_classifier.pkl` (single-file bundle the app loads) and `enhanced_food_classifier.onnx`, plus the JSON files in `enhanced_model/artifacts/`.
Only load `.pkl` files you trained yourself.

### 4. Evaluate

```bash
python -m enhanced_model.evaluate
python -m enhanced_model.evaluate_samples
```

* `evaluate` scores the held-out test images and compares both models on the same images.
  Results go to `enhanced_model/results/`.
* `evaluate_samples` runs both models on your four photos in `image-search/` and writes
  `sample_image_predictions.csv` and `sample_image_comparison.md`.

Use the numbers from these files in your report. The four photos are only a diagnostic check,
and the test-set numbers are the real accuracy estimate.

---

## Part C: Use the comparison page

```bash
streamlit run app.py
```

Sidebar, then **Model comparison**:

1. Choose a mode: **Original Model**, **Enhanced Model**, or **Compare Both Models**.
2. Choose **Upload a photo** (JPG or PNG up to 10 MB) or **Use a sample image**.
3. Press **Run**.
4. Results appear side by side, each labelled with its model. A warning shows when a model is uncertain.
5. The session history table at the bottom has a **Download CSV** button.
6. **Enhanced Model: Dataset and Performance** (bottom) fills in automatically after `evaluate` has run.

"Model confidence" is the model's own score. It is not a guarantee that the prediction is correct.

---

## Part D: Run the tests

```bash
pip install pytest
pip install onnx          # only needed for the enhanced-model unit tests
pytest tests/test_original_model.py tests/test_enhanced_model.py tests/test_streamlit_integration.py
```

Expected: 22 passed. These confirm the original model files and predictions are unchanged and that the
enhanced inference and the comparison page behave correctly (the tests use a tiny synthetic stand-in, not your trained model).

Full suite: `pytest tests`. The PNIE and Gemini tests need `xgboost` and `google-genai`
from `requirements.txt` and will fail if those are not installed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: enhanced_model` | You are not in the `NutriVision-AI/` folder. `cd` there and use `python -m enhanced_model.<name>`. |
| "Enhanced model not available" on the page | Training has not been run, or `checkpoints/enhanced_food_classifier.pkl` (or the `.onnx`) is missing. Do Part B. |
| "different training runs" error | The ONNX file and `artifacts/class_mapping.json` come from different trainings. Re-run `train`. |
| `Food-101 not found under ...` | `--food101-root` must contain `meta/train.txt` and `images/`. |
| Download fails or is slow | Download Food-101 manually, extract it, and use the `--food101-root` option. |
| Out of memory in training | `python -m enhanced_model.train --batch-size 16`. |
| Original page asks for a profile | Run `python database/init_db.py --sqlite --sample-data`, or create a profile in the app. |
| Port in use | `streamlit run app.py --server.port 8502` |

---

## Where things are

| Path | Purpose |
|---|---|
| `app.py`, `pages/` | The Streamlit app (`pages/7_Model_Comparison.py` is new) |
| `food_recognition/`, `assets/models/` | Original model (unchanged) |
| `enhanced_model/` | New model: dataset prep, training, evaluation, inference, README |
| `enhanced_model/results/` | Real metrics after you evaluate |
| `datasets/` | Generated dataset manifest (git-ignored) |
| `tests/` | Regression, enhanced-model and page tests |
