"""Configuration for the Enhanced Food Classifier (Model 2).

Everything is independent of the original model: its own classes, preprocessing,
checkpoints and results. Edit ``TARGET_CLASSES`` (or set ``ENHANCED_CLASSES``,
comma separated, in the environment) to change the class list, then re-run
``prepare_dataset.py`` and ``train.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIR: Final[Path] = PACKAGE_DIR.parent

CHECKPOINT_DIR: Final[Path] = PACKAGE_DIR / "checkpoints"
ARTIFACT_DIR: Final[Path] = PACKAGE_DIR / "artifacts"
RESULTS_DIR: Final[Path] = PACKAGE_DIR / "results"
DATASET_DIR: Final[Path] = Path(
    os.environ.get("ENHANCED_DATASET_DIR", PROJECT_DIR / "datasets" / "enhanced_food_dataset")
)
SAMPLE_IMAGE_DIR: Final[Path] = PROJECT_DIR.parent / "image-search"
if not SAMPLE_IMAGE_DIR.is_dir():                       # e.g. repo layout without the parent folder
    SAMPLE_IMAGE_DIR = PROJECT_DIR / "image-search"

# Files produced by training and consumed by inference.
ONNX_MODEL_PATH: Final[Path] = CHECKPOINT_DIR / "enhanced_food_classifier.onnx"
# Single-file bundle (ONNX bytes + classes + preprocessing + thresholds); preferred at runtime when present.
PICKLE_MODEL_PATH: Final[Path] = CHECKPOINT_DIR / "enhanced_food_classifier.pkl"
TORCH_CHECKPOINT_PATH: Final[Path] = CHECKPOINT_DIR / "enhanced_food_classifier_best.pt"
LAST_CHECKPOINT_PATH: Final[Path] = CHECKPOINT_DIR / "enhanced_food_classifier_last.pt"
CLASS_MAPPING_PATH: Final[Path] = ARTIFACT_DIR / "class_mapping.json"
PREPROCESSING_PATH: Final[Path] = ARTIFACT_DIR / "preprocessing_config.json"
TRAINING_CONFIG_PATH: Final[Path] = ARTIFACT_DIR / "training_config.json"
THRESHOLDS_PATH: Final[Path] = ARTIFACT_DIR / "uncertainty_thresholds.json"
MANIFEST_PATH: Final[Path] = DATASET_DIR / "manifest.csv"
DATASET_METADATA_PATH: Final[Path] = DATASET_DIR / "dataset_metadata.json"

# --------------------------------------------------------------------------- #
# Classes (Food-101 folder names). The first five are the priority classes from
# the observed failures; the rest are optional extras that exist in Food-101.
# Only classes that really exist in the dataset can be used.
# --------------------------------------------------------------------------- #
PRIORITY_CLASSES: Final[tuple[str, ...]] = ("pizza", "samosa", "sushi", "bruschetta", "chicken_wings")
EXTRA_CLASSES: Final[tuple[str, ...]] = (
    "hamburger", "french_fries", "club_sandwich", "spaghetti_bolognese", "greek_salad",
    "fried_rice", "dumplings", "spring_rolls", "tacos", "pancakes",
)
_env_classes = os.environ.get("ENHANCED_CLASSES", "").strip()
TARGET_CLASSES: Final[tuple[str, ...]] = (
    tuple(c.strip() for c in _env_classes.split(",") if c.strip())
    if _env_classes
    else PRIORITY_CLASSES + EXTRA_CLASSES
)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DATASET_NAME: Final[str] = "Food-101 (ETH Zurich), class subset"
DATASET_LICENSE: Final[str] = (
    "Food-101 images were scraped from foodspotting.com; the dataset is provided for "
    "non-commercial research use. Check the original terms before redistributing."
)
DATASET_VERSION: Final[str] = "enhanced-v1"
SPLIT_SEED: Final[int] = 42
VAL_FRACTION_OF_TRAIN: Final[float] = 0.15   # carved out of Food-101's official train split
DUPLICATE_HASH_DISTANCE: Final[int] = 4      # dHash Hamming distance treated as near-duplicate
MIN_IMAGES_PER_CLASS: Final[int] = 200

# --------------------------------------------------------------------------- #
# Model + preprocessing (consumed by BOTH training and inference)
# --------------------------------------------------------------------------- #
ARCHITECTURE: Final[str] = "efficientnet_b0"
PRETRAINED_SOURCE: Final[str] = "torchvision EfficientNet_B0_Weights.IMAGENET1K_V1 (ImageNet-1k)"
IMAGE_SIZE: Final[int] = 224
RESIZE_SIZE: Final[int] = 256
MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42
BATCH_SIZE: Final[int] = 32
HEAD_EPOCHS: Final[int] = 3          # stage 1: backbone frozen
FINETUNE_EPOCHS: Final[int] = 12     # stage 2: whole network, lower LR
HEAD_LR: Final[float] = 1e-3
FINETUNE_LR: Final[float] = 1e-4
WEIGHT_DECAY: Final[float] = 1e-4
LABEL_SMOOTHING: Final[float] = 0.1
EARLY_STOP_PATIENCE: Final[int] = 4
NUM_WORKERS: Final[int] = int(os.environ.get("ENHANCED_NUM_WORKERS", "2"))

# --------------------------------------------------------------------------- #
# Uncertainty (defaults used only until calibrate_thresholds has written
# artifacts/uncertainty_thresholds.json from VALIDATION data)
# --------------------------------------------------------------------------- #
DEFAULT_MIN_CONFIDENCE: Final[float] = 50.0   # percent
DEFAULT_MIN_MARGIN: Final[float] = 10.0       # percentage points
TOP_K: Final[int] = 3
MAX_UPLOAD_MB: Final[float] = 10.0
MAX_IMAGE_PIXELS: Final[int] = 40_000_000
