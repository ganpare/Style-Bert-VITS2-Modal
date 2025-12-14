# Style-Bert-VITS2 Modal Training (Forked)

[日本語ガイド](/README_MODAL_JP.md)

This repository is a fork of [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2), adapted for training on [Modal](https://modal.com/).

## 🚀 Features

- **Serverless Training**: Run training on Modal's GPU infrastructure without managing servers.
- **Easy Setup**: Automated environment setup with `uv` and `modal`.
- **Custom Dataset**: Train with your own dataset easily.

## 🛠 Prerequisites

- [Modal Account](https://modal.com/)
- [uv](https://github.com/astral-sh/uv) (Recommended for fast Python package management)
- Python 3.10+

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Style-Bert-VITS2
   ```

2. **Setup Environment:**
   ```bash
   # Create a virtual environment with Python 3.10
   uv venv .venv --python 3.10
   source .venv/bin/activate

   # Install Modal
   uv pip install modal

   # Authenticate with Modal
   modal setup
   ```

## 📂 Dataset Preparation

Prepare your dataset in the following structure:

```
dataset/
├── wavs/           # Directory containing .wav files
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── esd.list        # Transcription file
```

Compress the `dataset` folder into `dataset.zip`:

```bash
zip -r dataset.zip dataset/
```

**Note**: `dataset.zip` will be created automatically if the `dataset/` folder exists when you run `run_training.sh`.

## ⚙️ Customize Training Settings

Edit the fixed values at the top of `train_sbv2.py`:

```python
# ==================== Fixed Settings ====================
DEFAULT_MODEL_NAME = "your_model"      # Model name (used for Volume save path)
MODAL_GPU = "H100"                     # GPU type: "A10G", "A100", "H100", etc.
TRAIN_BATCH_SIZE = 24                  # Batch size for training
PREPROCESS_BATCH_SIZE = 4              # Batch size for preprocessing
TRAIN_EPOCHS = 100                     # Total number of epochs
SNAPSHOT_EVERY_EPOCHS = 5              # Snapshot interval (save every N epochs)
YOMI_ERROR = "skip"                    # Behavior on reading errors: "raise" or "skip"
# ======================================================
```

**Important Settings:**
- `DEFAULT_MODEL_NAME`: Model identifier. Also used for Volume save path
- `MODAL_GPU`: GPU to use. `H100` recommended for faster training
- `TRAIN_EPOCHS`: Total epochs to train
- `SNAPSHOT_EVERY_EPOCHS`: Checkpoints are saved to Volume at this interval
- `TRAIN_BATCH_SIZE` vs `PREPROCESS_BATCH_SIZE`: Former for training loop, latter for preprocessing (BERT generation, etc.)

## 🏋️‍♂️ Training

1. **Run Training:**
   ```bash
   bash run_training.sh
   ```
   
   This script automatically:
   - Creates Modal Volume (`sbv2-vol`)
   - Generates `dataset.zip` from `dataset/` (if it doesn't exist)
   - Uploads `dataset.zip` to Volume
   - Starts training pipeline on Modal in **detached mode**
   - Retrieves training log (`training.log`)

2. **Monitor Progress:**
   
   Training runs in the background (detached mode). To monitor progress:
   
   ```bash
   # Check on Modal dashboard
   # Or stream logs from command line
   modal app logs <app-id>
   ```
   
   The URL to the dashboard is displayed when you run `run_training.sh`.

3. **Resume Training (Checkpoint Continuation):**
   
   The system automatically resumes training:
   - Checkpoints are saved to Volume every `SNAPSHOT_EVERY_EPOCHS`
   - Running `run_training.sh` again automatically resumes from the last checkpoint
   - **Important**: Do NOT delete the model's save path with `modal volume rm` if you want to continue training
   
   ```bash
   # ❌ This will reset training (start from scratch)
   modal volume rm -r sbv2-vol trained_models/your_model
   
   # ✅ Don't delete if you want to continue training
   ```

## 📥 Download Model

After training completes, download the model from the Modal Volume:

```bash
# List files in the volume (change model name to match your settings)
modal volume ls sbv2-vol trained_models/your_model

# Download to local directory
mkdir -p volume_dump/your_model
modal volume get sbv2-vol trained_models/your_model volume_dump/your_model --force
```

**Downloaded Files Include:**
- `checkpoints/G_*.pth`, `D_*.pth`, `WD_*.pth` - Training checkpoints
- `model_assets/*.safetensors` - Inference model
- `model_assets/style_vectors.npy` - Style vectors
- `config.json`, `train.list`, `val.list` - Configuration files

### Windows (PowerShell) One-liner

You can run listing → mkdir → download → local listing in one command:

```powershell
modal volume ls sbv2-vol trained_models/your_model; `
   New-Item -ItemType Directory -Force -Path .\volume_dump\your_model | Out-Null; `
   modal volume get sbv2-vol trained_models/your_model .\volume_dump\your_model --force; `
   Get-ChildItem -Recurse .\volume_dump\your_model | Select-Object -First 80 FullName,Length | Format-Table -AutoSize
```

## 💾 Volume Persistence

**Important**: Checkpoints saved to Modal Volume persist after the training job ends.

- ✅ Data in Volume persists after training completion
- ✅ Automatically resumes from checkpoint on next run
- ⚠️ Deleting with `modal volume rm` makes recovery impossible
- 💡 To train multiple models in parallel, change `DEFAULT_MODEL_NAME`

## 🧹 Cleanup

To remove the virtual environment:
```bash
rm -rf .venv
```

## 📄 License

This project follows the license of the original [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) repository.
