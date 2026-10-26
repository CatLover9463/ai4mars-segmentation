"""Train a compact U-Net on AI4Mars terrain masks."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Image as HFImage, load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1, self.enc2, self.enc3 = DoubleConv(in_channels, c), DoubleConv(c, c*2), DoubleConv(c*2, c*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c*4, c*8)
        self.up3, self.dec3 = nn.ConvTranspose2d(c*8, c*4, 2, 2), DoubleConv(c*8, c*4)
        self.up2, self.dec2 = nn.ConvTranspose2d(c*4, c*2, 2, 2), DoubleConv(c*4, c*2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c*2, c, 2, 2), DoubleConv(c*2, c)
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
    if value.get("bytes") is not None:
        return Image.open(BytesIO(value["bytes"])).copy()
    if value.get("path"):
        return Image.open(value["path"]).copy()
    raise ValueError("Image field has neither bytes nor a path")


class MarsDataset(Dataset):
    def __init__(self, data, indices, size):
        self.data, self.indices, self.size = data, indices, size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.data[self.indices[index]]
        image = row["image"].convert("L").resize(self.size, Image.Resampling.BILINEAR)
        mask = row["label_mask"].convert("L").resize(self.size, Image.Resampling.NEAREST)
        x = np.asarray(image, dtype=np.float32).copy() / 255.0
        y = np.asarray(mask, dtype=np.int64).copy()
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)


def metrics_from_confusion(cm):
    tp = cm.diag().double()
    union = cm.sum(1).double() + cm.sum(0).double() - tp
    present = union > 0
    iou = torch.full((cm.shape[0],), float("nan"), dtype=torch.float64)
    iou[present] = tp[present] / union[present]
    total = cm.sum().item()
    return (tp.sum().item() / total if total else float("nan"),
            iou[present].mean().item() if present.any() else float("nan"))


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, batches = 0.0, 0
    cm = torch.zeros(4, 4, dtype=torch.long)
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
        preds = logits.argmax(1)
        valid = (masks >= 0) & (masks < 4) & (masks != 255)
        if valid.any():
            bins = torch.bincount((masks[valid] * 4 + preds[valid]).cpu(), minlength=16)
            cm += bins.reshape(4, 4)
    accuracy, miou = metrics_from_confusion(cm)
    return total_loss / max(batches, 1), accuracy, miou


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", default="hassanjbara/AI4MARS")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ai4mars"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="Use 0 for the most reliable Windows setup")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument("--resume", action="store_true", help="Resume from outputs/latest.pth")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size % 16:
        p.error("epochs and batch-size must be positive; image-size must be divisible by 16")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        # Input shapes are fixed, so autotuning can select faster cuDNN kernels.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; output: {args.output_dir.resolve()}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_dataset(args.dataset_id, split="train")
    raw = data.select_columns(["image", "label_mask", "has_labels"]).cast_column("image", HFImage(decode=False)).cast_column("label_mask", HFImage(decode=False))
    usable = []
    for idx in range(len(raw)):
        row = raw[idx]
        if not row["has_labels"] or row["label_mask"] is None: continue
        try:
            image = decode_image(row["image"]); image.close()
            mask = decode_image(row["label_mask"]).convert("L")
            valid = np.any(np.asarray(mask) != 255); mask.close()
            if valid: usable.append(idx)
        except Exception as exc:
            print(f"Skipping unreadable record {idx}: {exc}")
    if len(usable) < 2: raise RuntimeError("Dataset has fewer than two usable labeled records")
    train_idx, val_idx = train_test_split(usable, test_size=.2, random_state=args.seed, shuffle=True)
    train_ds = MarsDataset(data, train_idx, (args.image_size, args.image_size))
    val_ds = MarsDataset(data, val_idx, (args.image_size, args.image_size))
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} validation")

    model = UNet(num_classes=4, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    start_epoch, best_miou = 1, -1.0
    history_path = args.output_dir / "training_history.json"
    history = []
    latest_path = args.output_dir / "latest.pth"
    if args.resume:
        if not latest_path.exists(): raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt.get("scaler_state_dict", {})); start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_val_miou", -1.0); history = ckpt.get("history", [])
        print(f"Resuming at epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_acc, val_miou = run_epoch(model, val_loader, device, criterion)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "val_pixel_accuracy": val_acc, "val_miou": val_miou,
                  "seconds": time.perf_counter() - t0}
        history.append(record)
        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                 "best_val_miou": max(best_miou, val_miou), "history": history,
                 "config": vars(args) | {"output_dir": str(args.output_dir)}}
        torch.save(state, latest_path)
        if np.isfinite(val_miou) and val_miou > best_miou:
            best_miou = val_miou
            state["best_val_miou"] = best_miou
            torch.save(state, args.output_dir / "best.pth")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | mIoU {val_miou:.4f} | acc {val_acc:.4f} | {record['seconds']/60:.1f} min")
    print(f"Finished. Best validation mIoU: {best_miou:.4f}; checkpoints in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
