"""Food recognition package (Phase 3).

Public surface used by the Streamlit pages::

    from food_recognition import FoodPredictor, ModelUnavailableError
    from food_recognition.preprocessing import preprocess_upload
"""

from food_recognition.model_loader import (
    ModelUnavailableError,
    find_onnx_path,
    load_food_model,
    load_labels,
    model_status,
    sanity_check,
)
from food_recognition.predictor import (
    FoodPrediction,
    FoodPredictor,
    confidence_tier,
    format_label,
)
from food_recognition.preprocessing import (
    ImageInfo,
    UnsupportedImageError,
    load_image,
    load_image_from_upload,
    preprocess,
    preprocess_upload,
    validate_upload,
)

__all__ = [
    "ModelUnavailableError",
    "UnsupportedImageError",
    "FoodPredictor",
    "FoodPrediction",
    "ImageInfo",
    "confidence_tier",
    "find_onnx_path",
    "format_label",
    "load_food_model",
    "load_image",
    "load_image_from_upload",
    "load_labels",
    "model_status",
    "preprocess",
    "preprocess_upload",
    "sanity_check",
    "validate_upload",
]
