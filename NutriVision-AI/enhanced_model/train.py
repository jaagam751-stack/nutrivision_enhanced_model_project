"""Independent training pipeline for the enhanced model.

    python -m enhanced_model.train                 # full run with ImageNet-pretrained EfficientNet-B0
    python -m enhanced_model.train --resume        # continue from checkpoints/..._last.pt

Stage 1 trains only the new classifier head; stage 2 fine-tunes the whole network at a lower
learning rate with cosine decay. Early stopping and "best checkpoint" selection use the
VALIDATION split only; the test split is never touched here.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from enhanced_model import config as cfg
from enhanced_model.dataset import ManifestDataset, dataset_info, read_manifest
from enhanced_model.model import build_model, export_onnx, set_backbone_trainable
from enhanced_model.predict import save_bundle


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(y); total += len(y); correct += (out.argmax(1) == y).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


def collect_probs(model, loader, device="cpu"):
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def calibrate_thresholds(probs: np.ndarray, labels: np.ndarray, target_accuracy: float = 0.90) -> dict:
    """Pick the smallest confidence threshold whose accepted VALIDATION predictions reach target accuracy."""
    conf = probs.max(1) * 100
    correct = probs.argmax(1) == labels
    best = None
    for t in range(10, 96, 1):
        accepted = conf >= t
        if accepted.sum() < max(10, 0.1 * len(labels)):
            break
        acc = float(correct[accepted].mean())
        if acc >= target_accuracy:
            best = {"min_confidence": float(t), "accepted_accuracy": round(acc, 4),
                    "coverage": round(float(accepted.mean()), 4)}
            break
    note = "calibrated on validation split"
    if best is None:
        best = {"min_confidence": cfg.DEFAULT_MIN_CONFIDENCE, "accepted_accuracy": None, "coverage": None}
        note = f"target accuracy {target_accuracy:.0%} not reachable on validation; default threshold kept"
    return {**best, "min_margin": cfg.DEFAULT_MIN_MARGIN, "target_accuracy": target_accuracy, "calibrated_on": note}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="demo profile: 300 train images/class, 2 head + 3 fine-tune epochs (minutes, not hours)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--head-epochs", type=int, default=cfg.HEAD_EPOCHS)
    ap.add_argument("--finetune-epochs", type=int, default=cfg.FINETUNE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--limit-per-class", type=int, default=0, help="debug: cap images per class and split")
    ap.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    ap.add_argument("--dataset-metadata", type=Path, default=cfg.DATASET_METADATA_PATH)
    ap.add_argument("--checkpoint-dir", type=Path, default=cfg.CHECKPOINT_DIR)
    ap.add_argument("--artifact-dir", type=Path, default=cfg.ARTIFACT_DIR)
    args = ap.parse_args(argv)
    if args.quick:
        args.head_epochs, args.finetune_epochs = 2, 3
        args.limit_per_class = args.limit_per_class or 300

    seed_everything(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True); args.artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / cfg.TORCH_CHECKPOINT_PATH.name
    last_path = args.checkpoint_dir / cfg.LAST_CHECKPOINT_PATH.name
    onnx_path = args.checkpoint_dir / cfg.ONNX_MODEL_PATH.name

    info = dataset_info(args.dataset_metadata)
    classes = info["classes"]
    root = Path(info["source_root"])
    rows = read_manifest(args.manifest)

    def split_rows(name):
        out, seen = [], {}
        for r in rows:
            if r["split"] != name:
                continue
            seen[r["class"]] = seen.get(r["class"], 0) + 1
            if not args.limit_per_class or seen[r["class"]] <= args.limit_per_class:
                out.append(r)
        return out

    mk = lambda name, train, shuffle: DataLoader(  # noqa: E731
        ManifestDataset(split_rows(name), root, classes, train), batch_size=args.batch_size,
        shuffle=shuffle, num_workers=cfg.NUM_WORKERS)
    train_loader, val_loader = mk("train", True, True), mk("validation", False, False)
    print(f"device={device} classes={len(classes)} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    total_epochs = args.head_epochs + args.finetune_epochs

    def make_optimizer(stage):
        set_backbone_trainable(model, stage == 2)
        lr = cfg.HEAD_LR if stage == 1 else cfg.FINETUNE_LR
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.finetune_epochs)) if stage == 2 else None
        return opt, sched

    start_epoch, best_acc, bad_epochs, history = 0, -1.0, 0, []
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); start_epoch, best_acc = ck["epoch"], ck["best_acc"]
        bad_epochs, history = ck["bad_epochs"], ck["history"]
        print(f"Resumed from epoch {start_epoch}")

    stage = 1 if start_epoch < args.head_epochs else 2
    optimizer, scheduler = make_optimizer(stage)
    if args.resume and last_path.is_file() and ck.get("stage") == stage:
        optimizer.load_state_dict(ck["optimizer"])
    started = time.time()
    for epoch in range(start_epoch, total_epochs):
        if epoch == args.head_epochs and stage == 1:
            stage = 2; optimizer, scheduler = make_optimizer(2); bad_epochs = 0
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = run_epoch(model, val_loader, criterion, None, device)
        if scheduler: scheduler.step()
        history.append({"epoch": epoch + 1, "stage": stage, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        print(f"epoch {epoch + 1}/{total_epochs} stage{stage} train {tl:.3f}/{ta:.3f} val {vl:.3f}/{va:.3f}", flush=True)
        if va > best_acc:
            best_acc, bad_epochs = va, 0
            torch.save({"model": model.state_dict(), "classes": classes, "val_acc": va, "epoch": epoch + 1}, best_path)
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "stage": stage,
                    "best_acc": best_acc, "bad_epochs": bad_epochs, "history": history}, last_path)
        if stage == 2 and bad_epochs >= cfg.EARLY_STOP_PATIENCE:
            print("Early stopping."); break

    # ---- export the BEST checkpoint + artifacts ------------------------------------------------
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model = build_model(len(classes), pretrained=False); model.load_state_dict(best["model"]); model.eval()
    export_onnx(model, onnx_path, len(classes))
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = torch.randn(2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    diff = float(np.abs(model(probe).detach().numpy() - sess.run(None, {"input": probe.numpy()})[0]).max())
    if diff > 1e-3:
        raise RuntimeError(f"ONNX export does not match PyTorch (max diff {diff:.2e}).")

    (args.artifact_dir / cfg.CLASS_MAPPING_PATH.name).write_text(json.dumps(
        {"classes": classes, "index_to_class": {str(i): c for i, c in enumerate(classes)}}, indent=2))
    (args.artifact_dir / cfg.PREPROCESSING_PATH.name).write_text(json.dumps(
        {"image_size": cfg.IMAGE_SIZE, "resize_size": cfg.RESIZE_SIZE, "mean": list(cfg.MEAN), "std": list(cfg.STD),
         "color_mode": "RGB", "interpolation": "bilinear"}, indent=2))

    probs, labels = collect_probs(model, val_loader)
    thresholds = calibrate_thresholds(probs, labels)
    (args.artifact_dir / cfg.THRESHOLDS_PATH.name).write_text(json.dumps(thresholds, indent=2))
    # One-file bundle (.pkl) for fast, single-file loading at runtime (no torch needed).
    pkl_path = args.checkpoint_dir / cfg.PICKLE_MODEL_PATH.name
    save_bundle(pkl_path, onnx_path, classes,
                json.loads((args.artifact_dir / cfg.PREPROCESSING_PATH.name).read_text()),
                {"min_confidence": thresholds["min_confidence"], "min_margin": thresholds["min_margin"],
                 "calibrated_on": thresholds["calibrated_on"]},
                {"architecture": cfg.ARCHITECTURE, "dataset_version": info["dataset_version"],
                 "best_val_accuracy": float(best_acc), "trained_at": datetime.now().isoformat(timespec="seconds"),
                 "pretrained": not args.no_pretrained})

    (args.artifact_dir / cfg.TRAINING_CONFIG_PATH.name).write_text(json.dumps({
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": cfg.ARCHITECTURE, "pretrained": not args.no_pretrained,
        "pretrained_source": cfg.PRETRAINED_SOURCE if not args.no_pretrained else "none (random init, smoke test)",
        "dataset_version": info["dataset_version"], "classes": classes, "image_size": cfg.IMAGE_SIZE,
        "batch_size": args.batch_size, "head_epochs": args.head_epochs, "finetune_epochs": args.finetune_epochs,
        "epochs_run": len(history), "optimizer": "AdamW", "head_lr": cfg.HEAD_LR, "finetune_lr": cfg.FINETUNE_LR,
        "weight_decay": cfg.WEIGHT_DECAY, "label_smoothing": cfg.LABEL_SMOOTHING, "scheduler": "CosineAnnealingLR (stage 2)",
        "seed": cfg.SEED, "limit_per_class": args.limit_per_class, "device": device,
        "best_val_accuracy": best_acc, "best_epoch": best["epoch"], "train_seconds": round(time.time() - started, 1),
        "onnx_max_abs_diff_vs_torch": diff, "history": history,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__, "onnxruntime": ort.__version__},
    }, indent=2))
    print(f"Done. best val acc {best_acc:.4f}. ONNX: {onnx_path} | bundle: {pkl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
