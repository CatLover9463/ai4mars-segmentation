"""Run a saved AI4Mars U-Net checkpoint on one rover image."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import CLASS_NAMES, NUM_CLASSES, UNet

PALETTE = np.array(
    [[156, 112, 73], [126, 132, 137], [220, 183, 110], [200, 70, 55]],
    dtype=np.uint8,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a rover image")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ai4mars_best.pth"))
    parser.add_argument("--output", type=Path, default=Path("prediction_overlay.png"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    size = config.get("image_size", [192, 192])
    if isinstance(size, int):
        size = [size, size]
    model = UNet(num_classes=NUM_CLASSES, base_channels=config.get("base_channels", 32)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(args.image) as opened:
        original = opened.convert("L")
        resized = original.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy() / 255.0)[None, None]
        with torch.inference_mode():
            prediction = model(tensor.to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)

        prediction = Image.fromarray(prediction, mode="L").resize(
            original.size, Image.Resampling.NEAREST
        )
        labels = np.asarray(prediction, dtype=np.uint8)
        rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)
        colors = PALETTE[labels]
        overlay = (0.52 * rgb.astype(np.float32) + 0.48 * colors.astype(np.float32)).clip(0, 255).astype(np.uint8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(args.output)

    counts = np.bincount(labels.ravel(), minlength=NUM_CLASSES)
    total = int(counts.sum())
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {counts[class_id] / total:.1%} of image pixels")
    print(f"Saved colored prediction overlay to {args.output.resolve()}")


if __name__ == "__main__":
    main()
