"""Train a reproducible U-Net control or augmentation run on AI4Mars."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Image as HFImage, load_dataset
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 4
IGNORE_INDEX = 255
CLASS_NAMES = ["Soil", "Bedrock", "Sand", "Big rock"]
# The inspected AI4Mars label-mask encoding is 0=Soil, 1=Bedrock,
# 2=Sand, 3=Big rock; all other values are ignored.
LABEL_ID_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Compact grayscale-input U-Net with four terrain logits."""
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 4, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec1 = DoubleConv(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def decode_image(value):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.copy()
    if value.get("bytes") is not None:
        return Image.open(BytesIO(value["bytes"])).copy()
    if value.get("path"):
        return Image.open(value["path"]).copy()
    raise ValueError("Image field has neither bytes nor a path")


def map_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Map documented source IDs to model IDs; ignore unknown/void values."""
    mapped = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for source_id, target_id in LABEL_ID_MAPPING.items():
        mapped[raw_mask == source_id] = target_id
    return mapped


class MarsDataset(Dataset):
    def __init__(self, data, indices, size, augment=False):
        self.data = data
        self.indices = indices
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.data[self.indices[index]]
        image = decode_image(row["image"]).convert("L").resize(
            self.size, Image.Resampling.BILINEAR
        )
        raw_mask = np.asarray(decode_image(row["label_mask"]).convert("L"), dtype=np.uint8)
        mask = Image.fromarray(map_mask(raw_mask), mode="L").resize(
            self.size, Image.Resampling.NEAREST
        )

        if self.augment:
            # Paired geometry: exactly the same flip and rotation for image/mask.
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle = random.uniform(-8.0, 8.0)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            mask = mask.rotate(
                angle, resample=Image.Resampling.NEAREST, fillcolor=IGNORE_INDEX
            )
            # Photometric transforms affect the image only.
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))

        x = np.asarray(image, dtype=np.float32).copy() / 255.0
        y = np.asarray(mask, dtype=np.int64).copy()
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)


def prepare_split(data, dataset_id, seed, split_path):
    fingerprint = getattr(data, "_fingerprint", None)
    if split_path.exists():
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id or manifest.get("seed") != seed:
            raise ValueError(
                f"Split manifest {split_path} does not match dataset/seed; choose a new --split-file."
            )
        if manifest.get("dataset_fingerprint") != fingerprint:
            raise ValueError(
                f"Dataset fingerprint changed since {split_path} was created; rebuild the split explicitly."
            )
        train_indices = manifest["train_indices"]
        val_indices = manifest["validation_indices"]
        return manifest, train_indices, val_indices

    raw = data.select_columns(["image", "label_mask", "has_labels"]).cast_column(
        "image", HFImage(decode=False)
    ).cast_column("label_mask", HFImage(decode=False))
    usable, bad_images, missing_masks, empty_masks = [], [], [], []
    for idx in range(len(raw)):
        row = raw[idx]
        if not row["has_labels"] or row["label_mask"] is None:
            missing_masks.append(idx)
            continue
        try:
            image = decode_image(row["image"])
            image.close()
            mask = decode_image(row["label_mask"]).convert("L")
            valid = np.isin(np.asarray(mask), list(LABEL_ID_MAPPING)).any()
            mask.close()
            if valid:
                usable.append(idx)
            else:
                empty_masks.append(idx)
        except Exception as exc:
            print(f"Skipping unreadable record {idx}: {exc}")
            bad_images.append(idx)
    if len(usable) < 2:
        raise RuntimeError("Dataset has fewer than two usable labeled records")

    train_indices, val_indices = train_test_split(
        usable, test_size=0.20, random_state=seed, shuffle=True
    )
    manifest = {
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "seed": seed,
        "split_method": "fixed random record-level 80/20 train/validation split",
        "grouping_note": "The source dataset schema has no scene/group identifier; records are split by index.",
        "label_id_mapping": {str(k): v for k, v in LABEL_ID_MAPPING.items()},
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "total_records": len(data),
        "usable_records": len(usable),
        "excluded_bad_image_indices": bad_images,
        "excluded_missing_mask_indices": missing_masks,
        "excluded_empty_mask_indices": empty_masks,
        "train_indices": [int(i) for i in train_indices],
        "validation_indices": [int(i) for i in val_indices],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, train_indices, val_indices


def metrics_from_confusion(cm):
    tp = cm.diag().double()
    union = cm.sum(1).double() + cm.sum(0).double() - tp
    present = union > 0
    iou = torch.full((cm.shape[0],), float("nan"), dtype=torch.float64)
    iou[present] = tp[present] / union[present]
    total = int(cm.sum().item())
    accuracy = tp.sum().item() / total if total else float("nan")
    miou = iou[present].mean().item() if present.any() else float("nan")
    per_class = [float(v) if torch.isfinite(v) else None for v in iou]
    return {
        "pixel_accuracy": accuracy,
        "miou": miou,
        "per_class_iou": per_class,
        "valid_pixels": total,
        "confusion_matrix": cm.tolist(),
    }


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, batches = 0.0, 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for images, masks in loader:
        non_blocking = device.type == "cuda"
        images = images.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item()
        batches += 1
        predictions = logits.argmax(1)
        valid = (masks >= 0) & (masks < NUM_CLASSES) & (masks != IGNORE_INDEX)
        if valid.any():
            bins = torch.bincount(
                (masks[valid] * NUM_CLASSES + predictions[valid]).cpu(),
                minlength=NUM_CLASSES**2,
            )
            cm += bins.reshape(NUM_CLASSES, NUM_CLASSES)
    metrics = metrics_from_confusion(cm)
    return total_loss / max(batches, 1), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="hassanjbara/AI4MARS")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ai4mars"))
    parser.add_argument("--split-file", type=Path, default=Path("outputs/splits/ai4mars_seed0.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--augment", action="store_true", help="Use paired geometric and image-only photometric augmentation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size % 16:
        parser.error("epochs/batch-size must be positive; image-size must be divisible by 16")
    if args.workers < 0:
        parser.error("workers cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; output: {args.output_dir.resolve()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_dataset(args.dataset_id, split="train")
    split_manifest, train_indices, val_indices = prepare_split(
        data, args.dataset_id, args.seed, args.split_file
    )
    # Copy the exact indices into each run folder so it remains self-describing.
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    image_size = (args.image_size, args.image_size)
    train_ds = MarsDataset(data, train_indices, image_size, augment=args.augment)
    val_ds = MarsDataset(data, val_indices, image_size)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    print(
        f"Samples: {len(train_ds)} train / {len(val_ds)} validation; "
        f"augmentation={'on' if args.augment else 'off'}"
    )

    config = {
        "dataset_id": args.dataset_id,
        "split_file": str(args.split_file),
        "seed": args.seed,
        "image_size": list(image_size),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "label_id_mapping": LABEL_ID_MAPPING,
        "ignore_index": IGNORE_INDEX,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_channels": args.base_channels,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "augmentation": "paired_flip_rotate8_brightness_contrast" if args.augment else "none",
        "device": str(device),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    model = UNet(num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    start_epoch, best_miou = 1, -1.0
    history_path = args.output_dir / "training_history.json"
    history = []
    latest_path = args.output_dir / "latest.pth"
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_val_miou", -1.0)
        history = checkpoint.get("history", [])
        print(f"Resuming at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_metrics = run_epoch(model, val_loader, device, criterion)
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_miou": val_metrics["miou"],
            "val_per_class_iou": val_metrics["per_class_iou"],
            "seconds": elapsed,
        }
        history.append(record)
        improved = np.isfinite(val_metrics["miou"]) and val_metrics["miou"] > best_miou
        if improved:
            best_miou = val_metrics["miou"]
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_miou": best_miou,
            "history": history,
            "config": config,
        }
        torch.save(state, latest_path)
        if improved:
            torch.save(state, args.output_dir / "best.pth")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | mIoU {val_metrics['miou']:.4f} | "
            f"acc {val_metrics['pixel_accuracy']:.4f} | {elapsed/60:.1f} min"
        )
    print(f"Finished. Best validation mIoU: {best_miou:.4f}; checkpoints in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
