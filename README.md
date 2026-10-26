# AI4Mars Terrain Segmentation

A reproducible four-class Mars rover terrain segmentation project. The main experiment compares the same compact U-Net with no augmentation and with paired, label-safe augmentation.

## Environment

Use Python 3.10 or newer. This project uses `uv`; on Windows/Linux, `pyproject.toml` pins PyTorch to its CUDA 13.0 wheel index so an NVIDIA GPU can be used.

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Keep the NVIDIA driver current. The project also runs on CPU, though training will be slower.

## Dataset and split

The public dataset is `hassanjbara/AI4MARS`. Hugging Face downloads and caches it on first use. The four source mask IDs map to Soil (0), Bedrock (1), Sand (2), and Big rock (3); void and unknown values map to ignore index 255. The dataset schema has no scene/group identifier, so the train/validation split is record-level and fixed by seed. The exact indices and dataset fingerprint are versioned in `outputs/splits/ai4mars_seed0.json` and copied into each run folder.

Training uses the dataset's `train` partition for an 80/20 train/validation split. The official `test_min1`, `test_min2`, and `test_min3` variants contain the same test images with different levels of annotation agreement, so the final one-shot evaluation uses only `test_min3` (the strictest agreement version). Do not inspect any test metrics while tuning. See the [dataset card](https://huggingface.co/datasets/hassanjbara/AI4MARS) for the variant definitions.

## Matched experiment

Both runs use the same seed, split, 192×192 image size, U-Net width, optimizer, batch size, worker count, and 20-epoch cap. The lower resolution was chosen to reduce local training time while preserving the full model width; report it as an experimental choice.

Compute the majority-class baseline first. This also creates the shared split manifest used by both model runs:

```powershell
uv run python baseline.py --image-size 192 --seed 0
```

The baseline summary is saved to `results/majority_baseline_192.json`.

Control, without augmentation:

```powershell
uv run python train.py --epochs 20 --batch-size 16 --workers 2 --image-size 192 --output-dir outputs/ai4mars_192_b16
```

Augmentation, using the same saved split:

```powershell
uv run python train.py --epochs 20 --batch-size 16 --workers 2 --image-size 192 --augment --output-dir outputs/ai4mars_192_b16_aug
```

The augmentation applies the same horizontal flip and small rotation to each image/mask pair, then applies modest brightness/contrast changes to the image only. Validation mIoU selects the best checkpoint; pixel accuracy is supplementary. Review the per-class IoU and confusion matrix as well as mIoU.

## Final test evaluation

After choosing control or augmentation by validation mIoU and freezing all settings, evaluate that selected checkpoint once. The report also evaluates the training-derived majority baseline on the same test split:

```powershell
uv run python evaluate.py --checkpoints models/ai4mars_best.pth --output results/final_test_metrics.json
```

The evaluator uses only `test_min3` and refuses to overwrite an existing report to help avoid repeated test-set use. It reports per-class IoU, mIoU, pixel accuracy, valid pixel count, and confusion matrix for the selected model and majority baseline. Do not use test results to revise the model; if they change a design decision, treat that limitation honestly.

## Outputs and resume

Each run saves `best.pth`, `latest.pth`, `training_history.json`, `run_config.json`, and its split manifest under `outputs/`. Those local training artifacts are git-ignored. Keep concise result summaries and report figures under `results/` for version control. Resume a run with `--resume`; `--epochs` remains the total target epoch count.

The notebook files contain exploratory analysis and the majority-class baseline. The standalone `train.py` and `evaluate.py` paths are the reproducible local training/evaluation workflow.
