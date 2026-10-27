"""One command to get the whole demo running: data -> quick training -> evaluation -> app.

    python quick_start.py                       # downloads Food-101 once if needed, trains quickly, opens the app
    python quick_start.py --food101-root D:\\food-101   # you already have Food-101
    python quick_start.py --skip-app            # do everything except launch Streamlit
    python quick_start.py --retrain             # ignore an existing trained model

Quick profile: the 5 priority classes (pizza, samosa, sushi, bruschetta, chicken_wings), 300 training
images per class, 5 epochs. It is a demo-sized model: expect lower accuracy than a full training run.
Every step is skipped if its output already exists, so re-running is cheap.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from enhanced_model import config as cfg  # noqa: E402

PRIORITY = ",".join(cfg.PRIORITY_CLASSES)


def run(step: str, *args: str) -> None:
    print(f"\n=== {step} ===", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, *args], cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"\nStep failed: {step}. Fix the error above, then re-run `python quick_start.py` "
                         "(finished steps are skipped).")
    print(f"--- done in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--food101-root", type=Path, default=HERE / "datasets" / "raw")
    ap.add_argument("--skip-app", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    args = ap.parse_args()

    root = args.food101_root
    have_food = (root / "meta" / "train.txt").is_file() or (root / "food-101" / "meta" / "train.txt").is_file()
    if not cfg.MANIFEST_PATH.is_file() or args.retrain:
        extra = [] if have_food else ["--download"]
        if extra:
            print("Food-101 not found: downloading it once (about 4.7 GB, 10-30 minutes depending on your connection).")
        run("1/5 Prepare dataset", "-m", "enhanced_model.prepare_dataset", "--food101-root", str(root),
            "--classes", PRIORITY, *extra)
    else:
        print("1/5 Dataset manifest exists, skipping.")

    if not cfg.PICKLE_MODEL_PATH.is_file() or args.retrain:
        run("2/5 Train (quick profile)", "-m", "enhanced_model.train", "--quick",
            *(["--no-pretrained"] if args.no_pretrained else []))
    else:
        print("2/5 Trained model exists, skipping (use --retrain to redo).")

    if not (cfg.RESULTS_DIR / "evaluation_metrics.json").is_file() or args.retrain:
        run("3/5 Evaluate on held-out test images", "-m", "enhanced_model.evaluate")
    else:
        print("3/5 Evaluation results exist, skipping.")

    if not (cfg.RESULTS_DIR / "sample_image_predictions.csv").is_file() or args.retrain:
        run("4/5 Run both models on the sample photos", "-m", "enhanced_model.evaluate_samples")
    else:
        print("4/5 Sample comparison exists, skipping.")

    if args.skip_app:
        print("\nAll set. Launch later with:  streamlit run app.py   (sidebar -> Model comparison)")
        return 0
    print("\n=== 5/5 Launching the app: open the 'Model comparison' page in the sidebar ===")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=HERE)


if __name__ == "__main__":
    raise SystemExit(main())
