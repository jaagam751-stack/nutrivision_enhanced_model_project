"""Build the enhanced-model dataset manifest from Food-101 (nothing is fabricated).

Usage
-----
    # Food-101 already on disk (folder containing images/ and meta/):
    python -m enhanced_model.prepare_dataset --food101-root /data/food-101
    # or let torchvision download it (~4.7 GB, needs internet):
    python -m enhanced_model.prepare_dataset --download --food101-root ./datasets/raw

Splits (reproducible, seed in config.py)
----------------------------------------
* test       = Food-101's OFFICIAL test images of the chosen classes (250/class). They
               were never used to train the ORIGINAL model either, so both models can be
               compared fairly on them.
* train/val  = Food-101's official train images, split 85/15 per class with a fixed seed.
* Near-duplicates (64-bit dHash, Hamming distance <= DUPLICATE_HASH_DISTANCE) are removed:
  any train/val image that is a near-duplicate of a test image is dropped from train/val,
  and duplicates inside train/val are dropped (first one kept), so none cross splits.

Images are not copied; ``manifest.csv`` lists them and ``dataset_metadata.json``
records counts, sources, licence and limitations.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from enhanced_model import config as cfg

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path) -> int | None:
    """64-bit difference hash; None if the image is unreadable or tiny."""
    try:
        with Image.open(path) as im:
            im.load()
            if min(im.size) < 32:
                return None
            gray = im.convert("L").resize((9, 8), Image.BILINEAR)
    except Exception:  # noqa: BLE001 - any decode failure marks the file invalid
        return None
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming_min(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """For each query hash, the minimum Hamming distance to any hash in pool."""
    result = np.full(len(query), 64, dtype=np.int32)
    if len(pool) == 0:
        return result
    pool_bytes = pool.view(np.uint8).reshape(len(pool), 8)
    for i, q in enumerate(query):
        x = np.bitwise_xor(pool_bytes, np.array([q], dtype=np.uint64).view(np.uint8))
        result[i] = int(_POPCOUNT[x].sum(axis=1).min())
    return result


def read_split_file(root: Path, name: str, classes: set[str]) -> list[tuple[str, str]]:
    path = root / "meta" / f"{name}.txt"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. --food101-root must contain images/ and meta/.")
    rows = []
    for line in path.read_text().splitlines():
        rel = line.strip()
        if rel and rel.split("/")[0] in classes:
            rows.append((rel.split("/")[0], f"images/{rel}.jpg"))
    return rows


def locate_root(arg: Path) -> Path:
    for candidate in (arg, arg / "food-101"):
        if (candidate / "meta" / "train.txt").is_file():
            return candidate
    raise SystemExit(f"Food-101 not found under {arg}. Use --download or point to the extracted folder.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--food101-root", type=Path, required=True)
    ap.add_argument("--download", action="store_true", help="download Food-101 with torchvision first")
    ap.add_argument("--classes", help="comma separated override of TARGET_CLASSES")
    args = ap.parse_args(argv)

    if args.download:
        from torchvision.datasets import Food101
        args.food101_root.mkdir(parents=True, exist_ok=True)
        Food101(root=str(args.food101_root), split="train", download=True)
    root = locate_root(args.food101_root)

    available = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    wanted = tuple(c.strip() for c in args.classes.split(",")) if args.classes else cfg.TARGET_CLASSES
    unknown = [c for c in wanted if c not in available]
    if unknown:
        print(f"Skipping classes not present in this dataset: {unknown}", file=sys.stderr)
    classes = [c for c in wanted if c in available]
    if len(classes) < 2:
        raise SystemExit("Need at least two valid classes.")

    train_rows = read_split_file(root, "train", set(classes))
    test_rows = read_split_file(root, "test", set(classes))
    print(f"{len(classes)} classes | official train {len(train_rows)} | official test {len(test_rows)}")

    def hashed(rows):
        kept, hashes, bad = [], [], 0
        for cls, rel in rows:
            h = dhash(root / rel)
            if h is None:
                bad += 1
                continue
            kept.append((cls, rel))
            hashes.append(h)
        return kept, np.array(hashes, dtype=np.uint64), bad

    test_kept, test_h, test_bad = hashed(test_rows)
    train_kept, train_h, train_bad = hashed(train_rows)

    # 1) drop train images that are near-duplicates of any test image (no leakage into test)
    d_to_test = _hamming_min(train_h, test_h)
    leak = d_to_test <= cfg.DUPLICATE_HASH_DISTANCE
    # 2) drop duplicates inside train (keep first)
    seen: list[int] = []
    dup_in_train = np.zeros(len(train_kept), dtype=bool)
    seen_arr = np.empty(0, dtype=np.uint64)
    for i in range(len(train_kept)):
        if leak[i]:
            continue
        if len(seen_arr) and _hamming_min(train_h[i:i + 1], seen_arr)[0] <= cfg.DUPLICATE_HASH_DISTANCE:
            dup_in_train[i] = True
            continue
        seen_arr = np.append(seen_arr, train_h[i])
    keep = ~(leak | dup_in_train)

    by_class: dict[str, list[str]] = defaultdict(list)
    for (cls, rel), k in zip(train_kept, keep):
        if k:
            by_class[cls].append(rel)

    rng = random.Random(cfg.SPLIT_SEED)
    manifest: list[dict[str, str]] = []
    for cls in classes:
        rels = sorted(by_class[cls])
        rng.shuffle(rels)
        n_val = int(round(len(rels) * cfg.VAL_FRACTION_OF_TRAIN))
        for j, rel in enumerate(rels):
            manifest.append({"path": rel, "class": cls, "split": "validation" if j < n_val else "train"})
    for cls, rel in test_kept:
        manifest.append({"path": rel, "class": cls, "split": "test"})

    counts = {s: Counter(r["class"] for r in manifest if r["split"] == s) for s in ("train", "validation", "test")}
    thin = [c for c in classes if min(counts[s][c] for s in counts) < cfg.MIN_IMAGES_PER_CLASS // 4]
    cfg.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "class", "split"])
        writer.writeheader()
        writer.writerows(manifest)

    meta = {
        "dataset_version": cfg.DATASET_VERSION,
        "created": date.today().isoformat(),
        "source": cfg.DATASET_NAME,
        "source_root": str(root),
        "license": cfg.DATASET_LICENSE,
        "classes": classes,
        "classes_requested_but_missing": unknown,
        "split_seed": cfg.SPLIT_SEED,
        "split_policy": ("test = Food-101 official test split; train/validation = official train split "
                         f"divided {int((1 - cfg.VAL_FRACTION_OF_TRAIN) * 100)}/{int(cfg.VAL_FRACTION_OF_TRAIN * 100)} per class"),
        "counts": {s: dict(counts[s]) for s in counts},
        "totals": {s: sum(counts[s].values()) for s in counts},
        "removed": {
            "unreadable_or_tiny": train_bad + test_bad,
            "near_duplicate_of_test": int(leak.sum()),
            "near_duplicate_within_train": int(dup_in_train.sum()),
        },
        "known_limitations": [
            "Food-101 photos come from one website; labels in the training split are noisy by design.",
            "Images are not copied; the manifest points into the Food-101 folder.",
            "Near-duplicate detection uses a 64-bit perceptual hash and is not exhaustive.",
            "The ORIGINAL model was trained on Food-101's train split; the test split here is unseen by both models.",
            *( [f"Very few images for: {thin}"] if thin else [] ),
        ],
    }
    cfg.DATASET_METADATA_PATH.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["totals"]), "removed:", meta["removed"])
    print(f"Wrote {cfg.MANIFEST_PATH} and {cfg.DATASET_METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
