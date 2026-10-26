# AI4Mars Terrain Segmentation

Train a compact U-Net to segment Mars rover imagery into Soil, Bedrock, Sand, and Big rock.

## Local training (Windows, macOS, or Linux)

1. Install Python 3.10 or newer.
2. In this project folder, sync dependencies with uv. This project pins PyTorch to its CUDA 13.0 wheel index on Windows/Linux so the NVIDIA GPU is available:

   ```powershell
   uv sync
   ```

   Then use `uv run` for commands so they use the synced environment. The CUDA PyTorch wheel includes its runtime; keep your NVIDIA driver current.
3. Start training:

   ```powershell
   uv run python train.py --epochs 20 --batch-size 8 --workers 2
   ```

The script downloads/caches `hassanjbara/AI4MARS` through Hugging Face Datasets on first run. It uses CUDA when available, then Apple MPS, otherwise CPU. CPU training can be slow; use `--batch-size 2` or `--image-size 128` if memory is tight. On Windows, `--workers 0` is the reliable default.

Checkpoints and `training_history.json` are written to `outputs/ai4mars/`. `latest.pth` is saved each epoch and `best.pth` tracks the best validation mIoU. Resume an interrupted run with:

```powershell
uv run python train.py --epochs 20 --batch-size 8 --workers 2 --resume
```

`--epochs` is the total target epoch count when resuming. Use `uv run python train.py --help` to see all options.

## Dataset

AI4Mars: https://data.nasa.gov/
Hugging Face dataset: `hassanjbara/AI4MARS`

The dataset is not stored in this repository. The notebook remains available for exploration; `train.py` is the standalone local training path and does not require Google Colab or Drive.
