"""Central configuration for NutriVision AI.

Every module imports its settings from here, so colours, file paths, profile
options, validation ranges, nutrition constants, database/API settings and the
PNIE / coach tuning parameters are defined in exactly one place. The comments say
which phase introduced each block; nothing here is a placeholder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Final, Optional, Tuple

# --------------------------------------------------------------------------- #
# Filesystem paths
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
NUTRITION_DIR: Final[Path] = BASE_DIR / "nutrition"
FOOD_RECOGNITION_DIR: Final[Path] = BASE_DIR / "food_recognition"
CACHE_DIR: Final[Path] = BASE_DIR / ".cache"           # downloaded model cache

STYLE_SHEET: Final[Path] = ASSETS_DIR / "style.css"
NUTRITION_CSV: Final[Path] = NUTRITION_DIR / "food_nutrition.csv"
ENV_FILE: Final[Path] = BASE_DIR / ".env"

# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
APP_NAME: Final[str] = "NutriVision AI"
APP_TAGLINE: Final[str] = "AI-Powered Food Recognition & Predictive Nutrition Analysis"
APP_ICON: Final[str] = "🥗"
APP_BUILD: Final[str] = "0.8.0"
CURRENT_PHASE: Final[str] = "Phase 8 · Profile, polish & deployment"

# --------------------------------------------------------------------------- #
# Profile options (single source of truth for widgets + validation)
# --------------------------------------------------------------------------- #
GENDERS: Final[Tuple[str, ...]] = ("Female", "Male", "Other")

# Keys are the exact labels shown in the UI and stored in the database.
ACTIVITY_LEVELS: Final[Dict[str, Dict[str, Any]]] = {
    "Sedentary (little or no exercise)": {
        "multiplier": 1.20,
        "description": "Desk job, minimal intentional movement.",
    },
    "Lightly active (light exercise 1-3 days/week)": {
        "multiplier": 1.375,
        "description": "Short walks, light workouts or household activity.",
    },
    "Moderately active (moderate exercise 3-5 days/week)": {
        "multiplier": 1.55,
        "description": "Regular gym, sports or active commuting.",
    },
    "Very active (hard exercise 6-7 days/week)": {
        "multiplier": 1.725,
        "description": "Daily training or physically demanding work.",
    },
    "Extra active (very hard exercise & physical job)": {
        "multiplier": 1.90,
        "description": "Athlete-level training plus manual labour.",
    },
}

DIET_PREFERENCES: Final[Tuple[str, ...]] = (
    "Vegetarian",
    "Non-Vegetarian",
    "Eggetarian",
    "Vegan",
    "Jain",
)

FITNESS_GOALS: Final[Tuple[str, ...]] = ("Weight Loss", "Muscle Gain", "Maintain Weight")

# Goal-specific calorie adjustment applied on top of maintenance calories (TDEE).
GOAL_CALORIE_FACTOR: Final[Dict[str, float]] = {
    "Weight Loss": 0.80,      # ~20 % deficit
    "Muscle Gain": 1.15,      # ~15 % surplus
    "Maintain Weight": 1.00,
}

# Protein target in grams per kilogram of body weight.
GOAL_PROTEIN_PER_KG: Final[Dict[str, float]] = {
    "Weight Loss": 1.6,
    "Muscle Gain": 1.8,
    "Maintain Weight": 1.2,
}

# --------------------------------------------------------------------------- #
# Input validation ranges
# --------------------------------------------------------------------------- #
# These ranges are enforced twice: Streamlit's number_input min/max stops
# out-of-range entry in the browser, and utils.helpers.validate_profile_inputs()
# re-checks every value server-side (it is also used by init_db.py and tests).
AGE_RANGE: Final[Tuple[int, int]] = (10, 100)
HEIGHT_RANGE_CM: Final[Tuple[float, float]] = (100.0, 250.0)
WEIGHT_RANGE_KG: Final[Tuple[float, float]] = (25.0, 300.0)
NAME_MAX_LENGTH: Final[int] = 60

# Safety floor so a large deficit never produces an unrealistically low target.
MIN_SAFE_CALORIES: Final[Dict[str, float]] = {
    "Female": 1200.0,
    "Male": 1500.0,
    "Other": 1300.0,
}

# --------------------------------------------------------------------------- #
# Health constants
# --------------------------------------------------------------------------- #
# WHO Asian-Indian BMI cut-offs (the app targets Indian food habits, so the
# Asian-Indian bands are used instead of the generic WHO bands).
BMI_BANDS: Final[Tuple[Tuple[float, float, str], ...]] = (
    (0.0, 18.5, "Underweight"),
    (18.5, 23.0, "Normal"),
    (23.0, 25.0, "Overweight"),
    (25.0, 1000.0, "Obese"),
)
BMI_TONE: Final[Dict[str, str]] = {
    "Underweight": "warn",
    "Normal": "good",
    "Overweight": "warn",
    "Obese": "bad",
}

# Fibre recommendation: 14 g per 1000 kcal (USDA / ICMR-RDA aligned rule of thumb).
FIBER_G_PER_1000_KCAL: Final[float] = 14.0

# --------------------------------------------------------------------------- #
# Food recognition (Phase 3)
# --------------------------------------------------------------------------- #
# Pretrained MobileNetV2 fine-tuned on Food-101 (101 classes). Chosen because it
# is small (~19 MB checkpoint), CPU-friendly, needs no GPU and has a real food
# taxonomy — unlike a generic ImageNet model, which has almost no Indian dishes.
FOOD_MODEL_ID: Final[str] = "AlexKoff88/mobilenet_v2_food101"
FOOD_MODEL_REVISION: Final[str] = "main"
FOOD_TORCH_WEIGHTS_FILENAME: Final[str] = "mobilenet_v2_food101.pt"   # local cache name
# The Hub file is named differently from our cache name — keep both explicit.
FOOD_TORCH_WEIGHTS_REMOTE_NAME: Final[str] = "pytorch_model.bin"
FOOD_ONNX_FILENAME: Final[str] = "mobilenet_v2_food101.onnx"
FOOD_ONNX_INT8_FILENAME: Final[str] = "mobilenet_v2_food101_int8.onnx"
FOOD_NUM_CLASSES: Final[int] = 101
FOOD101_CLASSES_PATH: Final[Path] = FOOD_RECOGNITION_DIR / "food101_classes.json"

# "auto" prefers ONNX when a valid file exists, then falls back to PyTorch.
# Override with the FOOD_MODEL_BACKEND environment variable ("onnx" | "torch").
FOOD_MODEL_BACKEND: Final[str] = os.environ.get("FOOD_MODEL_BACKEND", "auto")

# Preprocessing constants: MobileNetV2 expects 224x224 with ImageNet statistics.
MODEL_IMAGE_SIZE: Final[int] = 224
MODEL_RESIZE_SIZE: Final[int] = 256
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Upload validation
ALLOWED_IMAGE_TYPES: Final[tuple[str, ...]] = ("jpg", "jpeg", "png")
MAX_UPLOAD_MB: Final[float] = 10.0
IMAGE_MAX_PIXELS: Final[int] = 40_000_000   # decompression-bomb guard (~40 MP)

# Confidence handling. Below CONFIDENCE_MEDIUM, or when the top-1 and top-2
# scores are closer than UNCERTAIN_MARGIN percentage points, the UI asks the user
# to confirm the food instead of trusting the label (used by Phase 4 too).
CONFIDENCE_HIGH: Final[float] = 70.0
CONFIDENCE_MEDIUM: Final[float] = 40.0
UNCERTAIN_MARGIN: Final[float] = 10.0

# --------------------------------------------------------------------------- #
# Nutrition engine (Phase 4)
# --------------------------------------------------------------------------- #
# Portion sizes the engine accepts (grams). The lower bound keeps "0 g" and
# negative values out; the upper bound stops a typo from producing an absurd meal.
PORTION_GRAMS_RANGE: Final[Tuple[float, float]] = (1.0, 2000.0)

#: Portion used when a food has no usable serving weight in the dataset.
DEFAULT_PORTION_GRAMS: Final[float] = 100.0

# --------------------------------------------------------------------------- #
# Dashboard & charts (Phase 5)
# --------------------------------------------------------------------------- #
#: Dark-theme palette shared by the CSS and every Plotly chart.
COLOR_ACCENT: Final[str] = "#7c5cff"
COLOR_ACCENT_2: Final[str] = "#22d3ee"
COLOR_GOOD: Final[str] = "#34d399"
COLOR_WARN: Final[str] = "#fbbf24"
COLOR_BAD: Final[str] = "#f87171"
COLOR_TEXT: Final[str] = "#e8ecf5"
COLOR_MUTED: Final[str] = "#97a1b8"
COLOR_GRID: Final[str] = "rgba(255,255,255,0.07)"
MACRO_COLORS: Final[Dict[str, str]] = {
    "protein": "#7c5cff",
    "carbs": "#22d3ee",
    "fat": "#fbbf24",
}

CHART_HEIGHT: Final[int] = 330
CHART_HEIGHT_SMALL: Final[int] = 280

#: Default window for the "recent trend" charts, and the broader range selector.
TREND_WINDOW_DAYS: Final[int] = 14
HISTORY_RANGE_OPTIONS: Final[Tuple[str, ...]] = (
    "Last 7 days",
    "Last 14 days",
    "Last 30 days",
    "Last 90 days",
    "All time",
    "Custom range",
)

#: How long a "same food, same portion" meal is treated as a duplicate save.
DUPLICATE_GUARD_SECONDS: Final[int] = 120

# --------------------------------------------------------------------------- #
# PNIE — Predictive Nutrition Intelligence Engine (Phase 6)
# --------------------------------------------------------------------------- #
# Exactly ONE XGBoost regression model per user, answering ONE question:
# "what will my average daily calories be next week?" Everything else the
# Predictions page shows (trends, direction, statistics, error bars) is plain
# arithmetic over the same saved meals — no second model, no invented numbers.
PNIE_MODEL_NAME: Final[str] = "xgboost-next-week-calories"
PNIE_MODEL_VERSION: Final[str] = "1.0.0"
PNIE_RANDOM_STATE: Final[int] = 42

#: Feature window (calendar days of history *before* a prediction origin) and
#: forecast horizon (calendar days being predicted).
PNIE_LOOKBACK_DAYS: Final[int] = 14
PNIE_HORIZON_DAYS: Final[int] = 7
#: Distance between two consecutive training origins. One row per week keeps the
#: rows far enough apart to be worth evaluating chronologically.
PNIE_ORIGIN_STRIDE_DAYS: Final[int] = 7
#: How far back the daily series is built (bounds training time and memory).
PNIE_MAX_SERIES_DAYS: Final[int] = 365

#: Minimum history before the engine reports anything at all.
PNIE_MIN_HISTORY_DAYS: Final[int] = 14   # span between first and last logged day
PNIE_MIN_LOGGED_DAYS: Final[int] = 7     # days carrying at least one meal
#: A training window is only usable when its target week has this many logged
#: days; otherwise the "next week average" would describe days never recorded.
PNIE_MIN_TARGET_LOGGED_DAYS: Final[int] = 3
#: Weekly windows needed before a model is trained. Below this the engine
#: reports descriptive statistics instead of fabricating a forecast.
PNIE_MIN_TRAINING_ROWS: Final[int] = 8
#: Chronological hold-out used to report an honest error estimate (never a
#: random split — a random split would let the model peek at the future).
PNIE_TEST_FRACTION: Final[float] = 0.25
PNIE_MIN_TEST_ROWS: Final[int] = 2

#: A meal log this old makes the forecast describe the past, so it is flagged.
PNIE_STALE_AFTER_DAYS: Final[int] = 7

#: A forecast this close to the recent average (kcal) is reported as "stable".
PNIE_STABLE_BAND_KCAL: Final[float] = 50.0
#: Safety guard: the raw model output may not deviate more than this fraction
#: from the recent average before it is clamped (and the clamp is disclosed).
PNIE_MAX_DEVIATION: Final[float] = 0.60
#: Absolute sanity bounds in kcal/day applied to every prediction and to
#: whatever is written to the ``predictions`` table.
PNIE_PREDICTION_KCAL_BOUNDS: Final[Tuple[float, float]] = (600.0, 6000.0)

#: XGBoost hyper-parameters. Deliberately small trees: with a handful of weekly
#: rows a deep model would memorise noise instead of learning the trend.
PNIE_XGB_PARAMS: Final[Dict[str, Any]] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.06,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 1,          # tiny datasets on shared hosting: one thread is plenty
    "verbosity": 0,
}

#: How long "the same forecast" counts as already stored (mirrors the meal guard).
PNIE_DUPLICATE_GUARD_SECONDS: Final[int] = 300

# --------------------------------------------------------------------------- #
# Gemini 2.5 Flash AI coach (Phase 7)
# --------------------------------------------------------------------------- #
# The coach is an *optional* feature: without a key the rest of the app works
# exactly as before and the AI Coach page explains how to configure it.
GEMINI_MODEL_NAME: Final[str] = "gemini-2.5-flash"

#: Request timeout in MILLISECONDS (the google-genai HttpOptions unit).
GEMINI_REQUEST_TIMEOUT_MS: Final[int] = 45_000
GEMINI_TEMPERATURE: Final[float] = 0.6
GEMINI_TOP_P: Final[float] = 0.95
GEMINI_MAX_OUTPUT_TOKENS: Final[int] = 1_200
#: ``None`` = let the API use the model's default thinking behaviour. Set to 0
#: (in code or via the environment) to disable thinking for faster replies.
GEMINI_THINKING_BUDGET: Final[Optional[int]] = None

#: Upper bound on the prompt we send. A snapshot this size is already far more
#: than the model needs; the cap stops a pathological history from being sent.
GEMINI_MAX_PROMPT_CHARS: Final[int] = 8_000
#: How many recent meals / most-frequent foods are described in the prompt.
GEMINI_MAX_MEALS_IN_CONTEXT: Final[int] = 8
GEMINI_MAX_TOP_FOODS: Final[int] = 6
#: The coach is asked for at most this many words (kept short and readable).
GEMINI_TARGET_WORDS: Final[int] = 200

#: Frequency guard: the same question asked again within this window reuses the
#: stored answer instead of spending another API call.
COACH_DUPLICATE_GUARD_SECONDS: Final[int] = 120
#: A stored recommendation older than this is flagged as possibly out of date.
COACH_STALE_AFTER_DAYS: Final[int] = 7
#: Below this many logged days the prompt tells the model the history is thin.
COACH_MIN_LOGGED_DAYS: Final[int] = 3

# --------------------------------------------------------------------------- #
# Profile page & deployment polish (Phase 8)
# --------------------------------------------------------------------------- #
#: Destructive actions ask the user to type this word first. A typed word is
#: harder to trigger by accident than a second click, and it is the same
#: "explicit action" rule the rest of the app follows.
PROFILE_DELETE_CONFIRM_WORD: Final[str] = "DELETE"
#: Shown next to the build number in the footer of every page.
APP_FOOTER_NOTE: Final[str] = "Estimates for general wellness — not medical advice."

# --------------------------------------------------------------------------- #
# Secrets (used from Phase 2 onwards; declared here so keys never change)
# --------------------------------------------------------------------------- #
SECRET_DATABASE_URL: Final[str] = "DATABASE_URL"          # Neon PostgreSQL (Phase 2)
SECRET_GEMINI_API_KEY: Final[str] = "GEMINI_API_KEY"      # Gemini 2.5 Flash (Phase 7)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_environment() -> None:
    """Load a local ``.env`` file when python-dotenv is available.

    On Streamlit Community Cloud there is no ``.env`` file: secrets are read
    from ``st.secrets`` inside :func:`get_secret`.
    """
    try:
        from dotenv import load_dotenv  # imported lazily so the app still runs without it
    except ImportError:  # pragma: no cover - dependency is in requirements.txt
        return
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)


def get_secret(key: str, default: Any = None) -> Any:
    """Return a secret from Streamlit Secrets (cloud) or the environment (local).

    Never logs the value, so API keys and database URLs stay out of log output.
    """
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            try:
                if key in secrets:
                    return secrets[key]
            except Exception:  # missing secrets.toml raises a Streamlit error
                pass
    except Exception:
        pass
    return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
# Runtime settings resolved from the environment / Streamlit secrets
# --------------------------------------------------------------------------- #
load_environment()

# The connection string is read at import time. Phase 1 ran without a database,
# so an empty value is a normal, expected state: the app then shows a setup
# panel instead of crashing. ``database.connection`` re-reads the environment on
# every call, which is what allows switching backends at runtime.
DATABASE_URL: Final[str] = str(get_secret(SECRET_DATABASE_URL, "") or "")

#: SQLite file used ONLY when the developer explicitly opts into the fallback.
SQLITE_URL: Final[str] = f"sqlite:///{BASE_DIR / 'nutrivision.db'}"
