"""Manifest-based PyTorch dataset + augmentation for the enhanced model (training only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from enhanced_model import config as cfg


def read_manifest(path: Path = cfg.MANIFEST_PATH) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m enhanced_model.prepare_dataset` first.")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def dataset_info(meta_path: Path = cfg.DATASET_METADATA_PATH) -> dict:
    return json.loads(Path(meta_path).read_text())


def build_transforms(train: bool) -> T.Compose:
    norm = T.Normalize(cfg.MEAN, cfg.STD)
    if train:
        # Realistic variation only: no vertical flips, no heavy colour shifts.
        return T.Compose([
            T.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize(cfg.RESIZE_SIZE), T.CenterCrop(cfg.IMAGE_SIZE), T.ToTensor(), norm])


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, classes: list[str], train: bool):
        self.rows, self.root = rows, Path(root)
        self.index = {c: i for i, c in enumerate(classes)}
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(self.root / row["path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), self.index[row["class"]]
