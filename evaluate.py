"""Evaluate frozen checkpoints once on AI4Mars' strict test_min3 labels.

Run only after model/configuration selection is complete. The test_min1/min2/min3
splits are alternative annotation-agreement versions of the same test images,
not independent partitions; this script evaluates only test_min3 by default.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Image as HFImage, load_dataset
from torch.utils.data import DataLoader

from train import (
    CLASS_NAMES,
    IGNORE_INDEX,
    LABEL_ID_MAPPING,
    NUM_CLASSES,
    MarsDataset,
    UNet,
    decode_image,
    run_epoch,
)


def labeled_test_indices(data):
    raw = data.select_columns(["image", "label_mask", "has_labels"]).cast_column(
        "image", HFImage(decode=False)
    ).cast_column("label_mask", HFImage(decode=False))
    usable, excluded = [], {"bad_image": [], "missing_mask": [], "empty_mask": []}
    for idx in range(len(raw)):
        row = raw[idx]
        if not row["has_labels"] or row["label_mask"] is None:
            excluded["missing_mask"].append(idx)
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
                excluded["empty_mask"].append(idx)
        except Exception:
            excluded["bad_image"].append(idx)
    if not usable:
        raise RuntimeError("The selected official test split contains no usable labeled samples")
    return usable, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--dataset-id", default="hassanjbara/AI4MARS")
    p.add_argument("--output", type=Path, default=Path("results/final_test_metrics.json"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; held-out test results should be produced once."
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

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        image_size = config.get("image_size", [256, 256])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
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
        test_loss, metrics = run_epoch(model, loader, device, criterion)
        results[checkpoint_path.parent.name] = {
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint.get("epoch", checkpoint.get("best_epoch")),
            "validation_miou": checkpoint.get("best_val_miou"),
            "test_loss": test_loss,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved one-shot test report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
