#!/bin/bash

# Modalのバイナリパスを定義（環境に応じて解決）
if [ -x "./.venv/bin/modal" ]; then
    MODAL_BIN="./.venv/bin/modal"
elif [ -x "./.venv/Scripts/modal.exe" ]; then
    MODAL_BIN="./.venv/Scripts/modal.exe"
else
    MODAL_BIN="modal"
fi

# dataset.zip が無い場合、dataset/ があれば自動で圧縮する
if [ ! -f "dataset.zip" ] && [ -d "dataset" ]; then
    echo "dataset.zip が見つからないため、dataset/ を dataset.zip に圧縮します..."
    if command -v python >/dev/null 2>&1; then
        python - <<'PY'
import zipfile
from pathlib import Path

dataset_dir = Path("dataset")
zip_path = Path("dataset.zip")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in dataset_dir.rglob("*"):
        if p.is_file():
            zf.write(p, p.as_posix())

print(f"Created {zip_path} from {dataset_dir}/")
PY
    elif command -v zip >/dev/null 2>&1; then
        zip -r -q dataset.zip dataset
    else
        # Windows/PowerShell フォールバック: Compress-Archive を使用
        if command -v pwsh >/dev/null 2>&1; then
            pwsh -NoLogo -NoProfile -Command "Compress-Archive -Path 'dataset/*' -DestinationPath 'dataset.zip' -Force" || {
                echo "エラー: PowerShell (pwsh) で圧縮に失敗しました。";
                exit 1;
            }
        elif command -v powershell >/dev/null 2>&1; then
            powershell -NoLogo -NoProfile -Command "Compress-Archive -Path 'dataset/*' -DestinationPath 'dataset.zip' -Force" || {
                echo "エラー: PowerShell で圧縮に失敗しました。";
                exit 1;
            }
        else
            echo "エラー: dataset.zip を作れません。python, zip, または PowerShell が必要です。"
            exit 1
        fi
    fi
fi

# dataset.zipが存在するか確認
if [ ! -f "dataset.zip" ]; then
    echo "エラー: dataset.zip が見つかりません！"
    echo "学習データを準備して dataset.zip に圧縮してください。"
    echo "フォルダ構成:"
    echo "dataset/"
    echo "  ├── wavs/"
    echo "  └── esd.list"
    exit 1
fi

# Volumeの作成（既に存在していても問題ありません）
# Pythonスクリプト内でも作成されますが、データをアップロードするために先に作成します。

echo "Modalボリューム 'sbv2-vol' を作成しています..."
$MODAL_BIN volume create sbv2-vol || true

echo "dataset.zip をアップロードしています..."
# 既存の /dataset.zip があれば消してからput（上書き不可のため）
$MODAL_BIN volume rm sbv2-vol /dataset.zip >/dev/null 2>&1 || true
$MODAL_BIN volume put sbv2-vol dataset.zip /dataset.zip

echo "学習を開始します..."
$MODAL_BIN run --detach train_sbv2.py

echo "ログを確認します..."
$MODAL_BIN volume get sbv2-vol training_error.txt . --force
