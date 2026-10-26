"""Evaluate frozen checkpoints once on AI4Mars' strict test_min3 labels.

Run only after model/configuration selection is complete. The test_min1/min2/min3
splits are alternative annotation-agreement versions of the same test images,
not independent partitions; this script evaluates only test_min3.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import Image as HFImage, load_dataset
from torch.utils.data import DataLoader

from train import (
    CLASS_NAMES,
    IGNORE_INDEX,
    LABEL_ID_MAPPING,
    NUM_CLASSES,
    MarsDataset,
    UNet,
    metrics_from_confusion,
)

PALETTE = np.array(
    [[156, 112, 73], [126, 132, 137], [220, 183, 110], [200, 70, 55]],
    dtype=np.uint8,
)


def labeled_test_indices(data):
    raw = data.select_columns(["image", "label_mask", "has_labels"]).cast_column(
        "image", HFImage(decode=False)
    ).cast_column("label_mask", HFImage(decode=False))
    usable, excluded = [], {"missing_image": [], "missing_mask": []}
    for idx in range(len(raw)):
        row = raw[idx]
        if row["image"] is None:
            excluded["missing_image"].append(idx)
        elif not row["has_labels"] or row["label_mask"] is None:
            excluded["missing_mask"].append(idx)
        else:
            usable.append(idx)
    if not usable:
        raise RuntimeError("The selected official test split contains no usable labeled samples")
    return usable, excluded


def overlay(image, labels):
    base = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    colors = PALETTE[np.clip(labels, 0, NUM_CLASSES - 1)].astype(np.float32)
    valid = labels != IGNORE_INDEX
    result = base.copy()
    result[valid] = 0.52 * base[valid] + 0.48 * colors[valid]
    return result.clip(0, 255).astype(np.uint8)


def evaluate_once(model, loader, indices, device, criterion, example_ids):
    model.eval()
    total_loss, batches, position = 0.0, 0, 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    target_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    examples = []
    example_ids = set(example_ids)
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            loss = criterion(logits, masks)
            predictions = logits.argmax(1)
            total_loss += loss.item()
            batches += 1
            valid = (masks >= 0) & (masks < NUM_CLASSES) & (masks != IGNORE_INDEX)
            if valid.any():
                target = masks[valid].cpu()
                predicted = predictions[valid].cpu()
                target_counts += torch.bincount(target, minlength=NUM_CLASSES)
                bins = torch.bincount(
                    target * NUM_CLASSES + predicted, minlength=NUM_CLASSES**2
                )
                cm += bins.reshape(NUM_CLASSES, NUM_CLASSES)
            for offset in range(len(images)):
                dataset_index = indices[position + offset]
                if dataset_index in example_ids:
                    image = (images[offset, 0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    truth = masks[offset].cpu().numpy().astype(np.uint8)
                    prediction = predictions[offset].cpu().numpy().astype(np.uint8)
                    examples.append((dataset_index, image, truth, prediction))
            position += len(images)
    return total_loss / max(batches, 1), metrics_from_confusion(cm), target_counts, examples


def save_test_examples(examples, model_name, output_dir):
    columns = ["Input", "Reference labels", "Prediction"]
    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 1.25 * len(examples)), squeeze=False)
    rows = []
    for row, (index, image, truth, prediction) in enumerate(examples):
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(overlay(image, truth))
        axes[row, 2].imshow(overlay(image, prediction))
        valid = truth != IGNORE_INDEX
        class_ious = []
        for class_id in range(NUM_CLASSES):
            target_class = (truth == class_id) & valid
            predicted_class = (prediction == class_id) & valid
            union = np.logical_or(target_class, predicted_class).sum()
            if union:
                class_ious.append(np.logical_and(target_class, predicted_class).sum() / union)
        image_iou = float(np.mean(class_ious)) if class_ious else None
        rows.append({"test_index": int(index), "mean_iou_present_classes": image_iou})
        iou_label = "n/a" if image_iou is None else f"{image_iou:.2f}"
        axes[row, 0].text(0.02, 0.02, f"test {index} | IoU {iou_label}", transform=axes[row, 0].transAxes,
                          color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
        for ax in axes[row]:
            ax.axis("off")
        if row == 0:
            for ax, title in zip(axes[row], columns):
                ax.set_title(title, fontsize=10)
    fig.suptitle(f"AI4Mars {model_name} | final test_min3 examples", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    path = output_dir / "test_prediction_overlays.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs=1, type=Path, required=True,
                   help="Exactly one checkpoint, selected using validation metrics")
    p.add_argument("--dataset-id", default="hassanjbara/AI4MARS")
    p.add_argument("--output", type=Path, default=Path("results/final_test_metrics.json"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    qualitative_path = args.output.parent / "test_prediction_overlays.png"
    if args.output.exists() or qualitative_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite held-out test output under {args.output.parent}; evaluate once only."
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    test_split = "test_min3"
    test_data = load_dataset(args.dataset_id, split=test_split)
    usable, excluded = labeled_test_indices(test_data)
    results = {}
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Include the same training-derived majority predictor in the final report.
    # This keeps the test comparison like-for-like without using test labels to
    # choose the majority class.
    baseline_path = Path("results/majority_baseline_192.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    majority_id = int(baseline["majority_class_id"])
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        image_size = config.get("image_size", [256, 256])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        if tuple(image_size) != tuple(baseline["image_size"]):
            raise ValueError(
                "Selected checkpoint resolution must match the resolution used for the saved majority baseline."
            )
        base_channels = config.get("base_channels", 32)
        model = UNet(num_classes=NUM_CLASSES, base_channels=base_channels).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_dataset = MarsDataset(test_data, usable, tuple(image_size))
        loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        model_name = "augmentation" if config.get("augmentation", "none") != "none" else "control"
        example_ids = np.random.default_rng(7).choice(usable, size=min(8, len(usable)), replace=False).tolist()
        test_loss, metrics, target_counts, examples = evaluate_once(
            model, loader, usable, device, criterion, example_ids
        )
        baseline_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        baseline_cm[:, majority_id] = target_counts
        results["majority_baseline"] = {
            "majority_class": CLASS_NAMES[majority_id],
            "majority_class_id": majority_id,
            **metrics_from_confusion(baseline_cm),
        }
        figure_path, example_rows = save_test_examples(examples, model_name, args.output.parent)
        results[model_name] = {
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint.get("epoch", checkpoint.get("best_epoch")),
            "validation_miou": checkpoint.get("best_val_miou"),
            "test_loss": test_loss,
            "qualitative_figure": str(figure_path),
            "qualitative_examples": example_rows,
            **metrics,
        }
        del model

    report = {
        "dataset_id": args.dataset_id,
        "test_split": test_split,
        "test_partition_record_count": len(test_data),
        "usable_test_record_count": len(usable),
        "excluded_test_record_counts": {k: len(v) for k, v in excluded.items()},
        "class_names": CLASS_NAMES,
        "label_id_mapping": {str(k): v for k, v in LABEL_ID_MAPPING.items()},
        "ignore_index": IGNORE_INDEX,
        "device": str(device),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved one-shot test report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
