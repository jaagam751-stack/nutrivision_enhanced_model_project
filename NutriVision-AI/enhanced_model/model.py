"""Architecture of the enhanced model: torchvision EfficientNet-B0 with a new head (training only)."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from enhanced_model import config as cfg


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(in_features, num_classes))
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def export_onnx(model: nn.Module, path, num_classes: int) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)
