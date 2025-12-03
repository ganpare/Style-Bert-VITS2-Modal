#!/bin/bash

# Modalのバイナリパスを定義
MODAL_BIN="./.venv/bin/modal"

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
$MODAL_BIN volume create sbv2-vol

echo "dataset.zip をアップロードしています..."
$MODAL_BIN volume put sbv2-vol dataset.zip /dataset.zip

echo "学習を開始します..."
$MODAL_BIN run train_sbv2.py

echo "ログを確認します..."
$MODAL_BIN volume get sbv2-vol training_error.txt .
