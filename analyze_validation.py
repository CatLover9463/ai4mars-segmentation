"""Compare frozen AI4Mars checkpoints and save validation-only figures."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader

from train import (
    CLASS_NAMES,
    IGNORE_INDEX,
    LABEL_ID_MAPPING,
    NUM_CLASSES,
    MarsDataset,
    UNet,
    decode_image,
    prepare_split,
    run_epoch,
)

PALETTE = np.array(
    [[156, 112, 73], [126, 132, 137], [220, 183, 110], [200, 70, 55]],
    dtype=np.uint8,
)


def display_mask(mask):
    rgb = np.full((*mask.shape, 3), 230, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        rgb[mask == class_id] = color
    return rgb


def overlay(image, mask, alpha=0.52):
    base = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    colors = display_mask(mask).astype(np.float32)
    valid = mask != IGNORE_INDEX
    result = base.copy()
    result[valid] = (1 - alpha) * base[valid] + alpha * colors[valid]
    return result.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    parser.add_argument("--split-file", type=Path, default=Path("outputs/splits/ai4mars_seed0.json"))
    parser.add_argument("--dataset-id", default="hassanjbara/AI4MARS")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data = load_dataset(args.dataset_id, split="train")
    manifest, train_indices, val_indices = prepare_split(
        data, args.dataset_id, 0, args.split_file
    )
    if len(val_indices) != manifest["total_records"] - len(manifest["excluded_bad_image_indices"]) - len(manifest["excluded_missing_mask_indices"]) - len(manifest["excluded_empty_mask_indices"]) - len(train_indices):
        raise ValueError("Validation indices do not agree with the saved split manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    results, predictions = {}, {}
    for name, checkpoint_path in (("control", args.control), ("augmentation", args.augmentation)):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        size = config.get("image_size", [192, 192])
        if isinstance(size, int):
            size = [size, size]
        model = UNet(num_classes=NUM_CLASSES, base_channels=config.get("base_channels", 32)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_dataset = MarsDataset(data, val_indices, tuple(size))
        loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        loss, metrics = run_epoch(model, loader, device, criterion)
        results[name] = {
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint.get("epoch"),
            "checkpoint_best_validation_miou": checkpoint.get("best_val_miou"),
            "validation_loss": loss,
            **metrics,
        }
        model.eval()
        selected = np.random.default_rng(0).choice(val_indices, size=min(args.examples, len(val_indices)), replace=False)
        samples = MarsDataset(data, selected.tolist(), tuple(size))
        model_predictions = []
        with torch.inference_mode():
            for image, _ in DataLoader(samples, batch_size=args.batch_size, shuffle=False):
                logits = model(image.to(device))
                model_predictions.extend(logits.argmax(1).cpu().numpy())
        predictions[name] = model_predictions
        del model

    # All requested model comparisons use only the saved validation partition.
    selected = np.random.default_rng(0).choice(val_indices, size=min(args.examples, len(val_indices)), replace=False)
    sample_dataset = MarsDataset(data, selected.tolist(), (192, 192))
    columns = ["Input", "Reference", "Control", "Augmentation"]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(12, 1.42 * len(selected)), squeeze=False)
    for row, (index, (image_tensor, mask_tensor)) in enumerate(zip(selected, sample_dataset)):
        image = (image_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        truth = mask_tensor.numpy().astype(np.uint8)
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(overlay(image, truth))
        axes[row, 2].imshow(overlay(image, predictions["control"][row]))
        axes[row, 3].imshow(overlay(image, predictions["augmentation"][row]))
        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].text(0.02, 0.02, f"val index {index}", transform=axes[row, 0].transAxes,
                          color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
        if row == 0:
            for ax, title in zip(axes[row], columns):
                ax.set_title(title, fontsize=10)
    fig.suptitle("AI4Mars validation examples | overlay colors: " + ", ".join(CLASS_NAMES), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    figure_path = args.output_dir / "validation_overlays.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_model = max(results, key=lambda name: results[name]["miou"])
    example_scores = []
    demo_dir = args.output_dir / "demo_examples"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for row, (index, (image_tensor, mask_tensor)) in enumerate(zip(selected, sample_dataset)):
        truth = mask_tensor.numpy().astype(np.uint8)
        prediction = predictions[selected_model][row]
        valid = truth != IGNORE_INDEX
        class_ious = []
        for class_id in range(NUM_CLASSES):
            target_class = (truth == class_id) & valid
            predicted_class = (prediction == class_id) & valid
            union = np.logical_or(target_class, predicted_class).sum()
            if union:
                class_ious.append(np.logical_and(target_class, predicted_class).sum() / union)
        example_scores.append(float(np.mean(class_ious)) if class_ious else float("nan"))

    best_row = int(np.nanargmax(example_scores))
    worst_row = int(np.nanargmin(example_scores))
    chosen_examples = {"success": best_row, "failure": worst_row}
    demo_indices = {}
    for label, row in chosen_examples.items():
        image_tensor, mask_tensor = sample_dataset[row]
        image = (image_tensor.squeeze(0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        mask = mask_tensor.numpy().astype(np.uint8)
        Image.fromarray(image, mode="L").save(demo_dir / f"{label}_input.png")
        Image.fromarray(mask, mode="L").save(demo_dir / f"{label}_mask.png")
        demo_indices[label] = {
            "validation_index": int(selected[row]),
            "sample_mean_iou": example_scores[row],
            "selected_model": selected_model,
        }

    report = {
        "dataset_id": args.dataset_id,
        "split_file": str(args.split_file),
        "split_fingerprint": manifest["dataset_fingerprint"],
        "validation_record_count": len(val_indices),
        "class_names": CLASS_NAMES,
        "label_id_mapping": {str(k): v for k, v in LABEL_ID_MAPPING.items()},
        "ignore_index": IGNORE_INDEX,
        "results": results,
        "qualitative_figure": str(figure_path),
        "qualitative_validation_indices": [int(i) for i in selected],
        "demo_examples": demo_indices,
    }
    report_path = args.output_dir / "validation_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved validation figure to {figure_path.resolve()}")


if __name__ == "__main__":
    main()
