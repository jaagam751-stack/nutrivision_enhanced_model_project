"""Builds a TINY synthetic ONNX classifier + artifacts for unit tests.

This is a test double that exercises the real inference code. It is NOT a trained food model
and must never be shipped as one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

CLASSES = ["pizza", "samosa", "sushi", "bruschetta", "chicken_wings"]


def build_fixture(directory: Path, classes=CLASSES, seed: int = 0) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 1.0, (3, len(classes))).astype(np.float32)
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "W"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(classes)])],
        initializer=[numpy_helper.from_array(weights, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    paths = {
        "model": directory / "model.onnx", "mapping": directory / "class_mapping.json",
        "pre": directory / "preprocessing_config.json", "thresholds": directory / "thresholds.json",
    }
    onnx.save(model, str(paths["model"]))
    paths["mapping"].write_text(json.dumps({"classes": list(classes)}))
    paths["pre"].write_text(json.dumps({"image_size": 224, "resize_size": 256,
                                        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}))
    return paths
