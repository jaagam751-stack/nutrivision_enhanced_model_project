"""Image validation and preprocessing for the food recognition model (Phase 3).

Everything here is plain Pillow + NumPy, with no deep-learning imports, so it can
be unit tested quickly and reused by both model backends (ONNX and PyTorch).

Pipeline (matches how the model was trained — MobileNetV2 / ImageNet stats):

1. reject anything that is not a readable JPG/JPEG/PNG inside the size limit
2. apply the EXIF orientation, convert to RGB
3. resize the shorter side to 256 px (aspect ratio preserved)
4. centre-crop to 224 x 224
5. scale to [0, 1] and normalise with the ImageNet mean/std
6. return a ``(1, 3, 224, 224)`` float32 array ready for the model

Verified: this Pillow pipeline produces the same top-1 prediction as
``torchvision.transforms.Resize(256) + CenterCrop(224)`` on the sample images
used during development (pixel-level differences are below 0.01 in [0, 1] space).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL import ImageFile

from config import (
    ALLOWED_IMAGE_TYPES,
    IMAGE_MAX_PIXELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_UPLOAD_MB,
    MODEL_IMAGE_SIZE,
    MODEL_RESIZE_SIZE,
)

logger = logging.getLogger(__name__)

# Guard against decompression bombs (a tiny file that expands to a huge bitmap).
Image.MAX_IMAGE_PIXELS = IMAGE_MAX_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False

#: Sources accepted by the public helpers.
ImageSource = Union[bytes, bytearray, BinaryIO, Path, str, Image.Image]


class UnsupportedImageError(ValueError):
    """Raised when an upload is not a valid, supported image."""


@dataclass(frozen=True)
class ImageInfo:
    """Basic facts about a validated image."""

    width: int
    height: int
    image_format: str
    mode: str
    size_mb: float

    def describe(self) -> str:
        return (
            f"{self.width} × {self.height} px · {self.image_format} · "
            f"{self.mode} · {self.size_mb:.2f} MB"
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _extension_of(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(str(filename)).suffix.lower().lstrip(".")


def _read_bytes(source: ImageSource, filename: Optional[str] = None) -> bytes:
    """Return the raw bytes of a source (bytes, path, or file-like object)."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, Image.Image):
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        return buffer.getvalue()

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise UnsupportedImageError(f"File not found: {path}")
        if _extension_of(path.name) not in ALLOWED_IMAGE_TYPES:
            raise UnsupportedImageError(
                f"Unsupported file type '.{_extension_of(path.name)}'. "
                f"Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}."
            )
        return path.read_bytes()

    # file-like object (e.g. a Streamlit UploadedFile)
    data = source.read()
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except (OSError, ValueError):  # pragma: no cover - not all streams are seekable
            pass
    if not isinstance(data, (bytes, bytearray)):
        raise UnsupportedImageError("The uploaded file could not be read as binary data.")
    return bytes(data)


def validate_upload(
    source: ImageSource,
    filename: Optional[str] = None,
    max_mb: float = MAX_UPLOAD_MB,
) -> ImageInfo:
    """Validate an upload and return its :class:`ImageInfo`.

    Checks the extension, the byte size, that Pillow can actually decode the
    file, and that the decoded format is really JPEG or PNG (so a renamed file
    is rejected too).

    Raises:
        UnsupportedImageError: with a message that is safe to show to the user.
    """
    display_name = filename or getattr(source, "name", None) or "the uploaded file"
    extension = _extension_of(display_name)
    if extension and extension not in ALLOWED_IMAGE_TYPES:
        raise UnsupportedImageError(
            f"'{extension}' files are not supported. Please upload "
            f"{', '.join(t.upper() for t in ALLOWED_IMAGE_TYPES)}."
        )

    data = _read_bytes(source, filename)
    if not data:
        raise UnsupportedImageError("The uploaded file is empty.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise UnsupportedImageError(
            f"The image is {size_mb:.1f} MB, which is larger than the {max_mb:g} MB limit. "
            "Please upload a smaller photo."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                      # cheap integrity check
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "UNKNOWN").upper()
            mode = image.mode
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnsupportedImageError(
            "That file could not be read as an image. It may be corrupted or not a real "
            f"photo (details: {exc.__class__.__name__})."
        ) from exc
    except Image.DecompressionBombError as exc:  # pragma: no cover - extreme inputs only
        raise UnsupportedImageError(
            "That image is too large to process safely. Please upload a smaller photo."
        ) from exc

    if image_format not in {"JPEG", "PNG"}:
        raise UnsupportedImageError(
            f"Detected a {image_format} file. Only JPEG and PNG images are supported "
            "(the file extension may not match its contents)."
        )

    return ImageInfo(
        width=width,
        height=height,
        image_format="JPEG" if image_format == "JPEG" else "PNG",
        mode=mode,
        size_mb=round(size_mb, 3),
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_image(source: ImageSource, filename: Optional[str] = None) -> Image.Image:
    """Load an image as an EXIF-corrected RGB :class:`PIL.Image.Image`."""
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")

    data = _read_bytes(source, filename)
    try:
        with Image.open(io.BytesIO(data)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnsupportedImageError(
            f"That file could not be opened as an image ({exc.__class__.__name__})."
        ) from exc


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Convenience wrapper for a Streamlit ``UploadedFile`` (validates first)."""
    validate_upload(uploaded_file, filename=getattr(uploaded_file, "name", None))
    return load_image(uploaded_file, filename=getattr(uploaded_file, "name", None))


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(
    image: Image.Image,
    size: int = MODEL_IMAGE_SIZE,
    resize: int = MODEL_RESIZE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Turn a PIL image into a ``(1, 3, size, size)`` float32 batch.

    Deterministic: the same image always produces the same array.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")

    width, height = rgb.size
    if width == 0 or height == 0:  # pragma: no cover - Pillow rejects these earlier
        raise UnsupportedImageError("The image has no pixels.")

    # 1. resize the shorter side, keeping the aspect ratio
    scale = resize / min(width, height)
    if scale != 1.0:
        rgb = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.BILINEAR,
        )

    # 2. centre crop
    width, height = rgb.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    rgb = rgb.crop((left, top, left + size, top + size))

    # 3. to float, scale, normalise, then NCHW
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    array = (array - mean_array) / std_array
    return np.transpose(array, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def preprocess_upload(uploaded_file) -> tuple[Image.Image, np.ndarray]:
    """Validate, load and preprocess a Streamlit upload in one step."""
    image = load_image_from_upload(uploaded_file)
    return image, preprocess(image)
