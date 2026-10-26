"""Compute the majority-class baseline on the fixed training/validation split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Image as HFImage, load_dataset
from PIL import Image

from train import (
    CLASS_NAMES,
    IGNORE_INDEX,
    LABEL_ID_MAPPING,
    NUM_CLASSES,
    decode_image,
    map_mask,
    metrics_from_confusion,
    prepare_split,
)


def class_counts(data, indices, size):
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for idx in indices:
        mask = decode_image(data[idx]["label_mask"]).convert("L")
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        labels = map_mask(np.asarray(mask, dtype=np.uint8))
        valid = labels != IGNORE_INDEX
        if valid.any():
            counts += torch.bincount(
                torch.from_numpy(labels[valid].astype(np.int64)),
                minlength=NUM_CLASSES,
            )
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", default="hassanjbara/AI4MARS")
    p.add_argument("--split-file", type=Path, default=Path("outputs/splits/ai4mars_seed0.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--output", type=Path, default=Path("results/majority_baseline_192.json"))
    args = p.parse_args()
    if args.image_size < 16 or args.image_size % 16:
        p.error("image-size must be divisible by 16")

    data = load_dataset(args.dataset_id, split="train")
    manifest, train_indices, val_indices = prepare_split(
        data, args.dataset_id, args.seed, args.split_file
    )
    training_counts = class_counts(data, train_indices, args.image_size)
    majority_id = int(training_counts.argmax().item())
    val_counts = class_counts(data, val_indices, args.image_size)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    cm[:, majority_id] = val_counts
    metrics = metrics_from_confusion(cm)
    result = {
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "image_size": [args.image_size, args.image_size],
        "split_file": str(args.split_file),
        "split_method": manifest["split_method"],
        "class_names": CLASS_NAMES,
        "label_id_mapping": {str(k): v for k, v in LABEL_ID_MAPPING.items()},
        "ignore_index": IGNORE_INDEX,
        "majority_class_id": majority_id,
        "majority_class": CLASS_NAMES[majority_id],
        "training_class_pixel_counts": training_counts.tolist(),
        "validation_class_pixel_counts": val_counts.tolist(),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved baseline result to {args.output.resolve()}")


if __name__ == "__main__":
    main()
