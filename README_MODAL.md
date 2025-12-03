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

## 🏋️‍♂️ Training

1. **Run Training:**
   ```bash
   ./run_training.sh
   ```
   This script will:
   - Create a Modal Volume (`sbv2-vol`).
   - Upload `dataset.zip`.
   - Start the training pipeline on Modal.
   - Download the training log (`training_error.txt`) if an error occurs.

2. **Monitor Progress:**
   You can monitor the training progress in your terminal or on the Modal dashboard.

3. **Customize Training:**
   Edit `train_sbv2.py` to change parameters like epochs:
   ```python
   @app.local_entrypoint()
   def main():
       # Change epochs here (e.g., epochs=50 for full training)
       train_pipeline.remote("dataset.zip", epochs=2) 
   ```

## 📥 Download Model

After training completes, download the model from the Modal Volume:

```bash
# List files in the volume
modal volume ls sbv2-vol trained_models/MyStyleModel

# Download to local directory
mkdir -p models
modal volume get sbv2-vol trained_models/MyStyleModel models/
```

## 🧹 Cleanup

To remove the virtual environment:
```bash
rm -rf .venv
```

## 📄 License

This project follows the license of the original [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) repository.
