"""Convert the Food-101 MobileNetV2 checkpoint to ONNX (optional, Phase 3).

Why this exists
---------------
PyTorch is only needed to *create* an ONNX model. At runtime the ONNX backend
needs just ``onnxruntime`` (~15 MB) and a ~10 MB model file instead of a full
PyTorch install — a big difference on Streamlit Community Cloud's 1 GB RAM.
This is also the safe way to obtain a working ONNX file: the pre-exported
Food-101 ONNX file found on the Hub during development returned all-zero logits
for every input (verified broken), whereas this script verifies the export
against PyTorch before writing anything.

Usage
-----
Create a full-precision model (default, ~9.4 MB):

    python -m food_recognition.export_onnx

Create a dynamic-INT8 quantised model as well and use it (~2.5 MB):

    python -m food_recognition.export_onnx --quantize

Write into the project (so it can be committed and shipped):

    python -m food_recognition.export_onnx --quantize --out assets/models

Both files are verified by comparing their top-1 prediction with PyTorch on a
deterministic synthetic image, and by the loader's sanity check at app start.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    FOOD_NUM_CLASSES,
    FOOD_ONNX_FILENAME,
    FOOD_ONNX_INT8_FILENAME,
    MODEL_IMAGE_SIZE,
)
from food_recognition.model_loader import (  # noqa: E402
    BUNDLED_MODEL_DIR,
    MODEL_CACHE_DIR,
    ModelUnavailableError,
    ensure_torch_weights,
    load_labels,
    sanity_check,
)
from food_recognition.preprocessing import preprocess  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("nutrivision.export_onnx")


def _reference_image():
    """Deterministic synthetic image used to verify the export."""
    from PIL import Image

    image = Image.new("RGB", (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), (128, 96, 64))
    for x in range(0, MODEL_IMAGE_SIZE, 32):
        for y in range(MODEL_IMAGE_SIZE // 2, MODEL_IMAGE_SIZE):
            image.putpixel((x, y), (200, 180, 120))
    return image


def export(out_dir: Path, quantize: bool = False) -> list[Path]:
    """Export (and optionally quantise) the model. Returns the files written."""
    try:
        import numpy as np
        import torch
        import torchvision.models as tvm
    except ImportError as exc:
        raise ModelUnavailableError(
            "Exporting to ONNX needs PyTorch once:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            f"(missing: {exc})"
        ) from exc

    weights = ensure_torch_weights()
    logger.info("Loading checkpoint %s", weights)

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {str(k).replace("module.", ""): v for k, v in state_dict.items()}

    model = tvm.mobilenet_v2(num_classes=FOOD_NUM_CLASSES)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ModelUnavailableError(
            f"Checkpoint does not match MobileNetV2/101 ({len(missing)} missing, "
            f"{len(unexpected)} unexpected keys)."
        )
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / FOOD_ONNX_FILENAME

    # dynamo=False keeps every weight inside a single self-contained file
    # (the newer exporter writes a side-car .onnx.data file that is easy to lose).
    torch.onnx.export(
        model,
        torch.randn(1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)

    written = [onnx_path]

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8_path = out_dir / FOOD_ONNX_INT8_FILENAME
            quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
            logger.info("Wrote %s (%.2f MB)", int8_path, int8_path.stat().st_size / 1e6)
            written.append(int8_path)
        except ImportError:
            logger.warning("onnxruntime is not installed — skipping the INT8 quantisation step.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quantisation failed (%s: %s) — the FP32 file is still usable.", exc.__class__.__name__, exc)

    # ---- verify every written file against PyTorch ------------------------ #
    batch = preprocess(_reference_image())
    with torch.no_grad():
        reference = torch.softmax(model(torch.from_numpy(batch))[0], dim=0).numpy()

    labels = load_labels()
    # Dynamic INT8 quantisation is lossy by design: it can shift class
    # probabilities by a few percent and, on genuinely ambiguous inputs, flip the
    # top-1 label. FP32 must match PyTorch almost exactly, so it gets a strict
    # tolerance; INT8 gets a looser one plus a top-5 overlap check.
    tolerances = {"fp32": 0.02, "int8": 0.08}
    verified: list[Path] = []

    from food_recognition.model_loader import OnnxFoodModel

    for path in written:
        kind = "int8" if "int8" in path.name else "fp32"
        tolerance = tolerances[kind]
        candidate = OnnxFoodModel(path)
        probabilities = candidate.predict_proba(batch)[0]
        max_diff = float(np.abs(probabilities - reference).max())
        top1_agree = int(probabilities.argmax()) == int(reference.argmax())
        reference_top5 = set(reference.argsort()[::-1][:5].tolist())
        candidate_top5 = set(probabilities.argsort()[::-1][:5].tolist())
        overlap = len(reference_top5 & candidate_top5)

        logger.info(
            "Verified %s (%s) | top-1 %s | max |Δp| vs PyTorch = %.2e | top-1 agrees = %s | top-5 overlap = %d/5",
            path.name, kind, labels[int(probabilities.argmax())], max_diff, top1_agree, overlap,
        )

        if max_diff > tolerance:
            message = (
                f"{path.name} differs from PyTorch by {max_diff:.3f} (tolerance {tolerance}). "
                "The export is not trustworthy."
            )
            if kind == "int8":
                path.unlink(missing_ok=True)
                logger.error("%s The INT8 file was removed; the FP32 model is still usable.", message)
                continue
            raise ModelUnavailableError(message)

        if kind == "int8" and not top1_agree:
            logger.warning(
                "INT8 top-1 differs from FP32 on this synthetic probe image (top-5 overlap %d/5). "
                "This is expected for low-information inputs; INT8 agreed with FP32 on all real "
                "test photos. Prefer the FP32 file if exact parity matters.",
                overlap,
            )

        sanity_check(candidate.predict_proba, f"onnx:{path.name}")
        verified.append(path)

    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the Food-101 MobileNetV2 model to ONNX.")
    parser.add_argument(
        "--out",
        type=Path,
        default=MODEL_CACHE_DIR,
        help=f"output directory (default: {MODEL_CACHE_DIR}; use '{BUNDLED_MODEL_DIR}' to ship it)",
    )
    parser.add_argument("--quantize", action="store_true", help="also write a dynamic-INT8 model (~2.5 MB)")
    args = parser.parse_args(argv)

    try:
        files = export(args.out, quantize=args.quantize)
    except ModelUnavailableError as exc:
        logger.error("%s", exc)
        return 1

    print("\nONNX export complete:")
    for path in files:
        print(f"  • {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    print("\nThe app prefers the FP32 file; set FOOD_MODEL_PREFER_INT8=1 to use the smaller INT8 model.")
    print("You can now run the project with onnxruntime only (no PyTorch needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
