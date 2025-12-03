# Style-Bert-VITS2 Modal トレーニングガイド (フォーク版)

[English Guide](/README_MODAL.md)

このリポジトリは [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) のフォークであり、[Modal](https://modal.com/) を使用したトレーニングに対応しています。

## 🚀 特徴

- **サーバーレス・トレーニング**: サーバー管理不要で、ModalのGPUインフラを使ってトレーニングを実行できます。
- **簡単セットアップ**: `uv` と `modal` を使って環境構築を自動化。
- **カスタムデータセット**: 自分のデータセットを使った学習が簡単にできます。

## 🛠 前提条件

- [Modal アカウント](https://modal.com/)
- [uv](https://github.com/astral-sh/uv) (Pythonパッケージ管理に推奨)
- Python 3.10以上

## 📦 インストール

1. **リポジトリのクローン:**
   ```bash
   git clone <your-repo-url>
   cd Style-Bert-VITS2
   ```

2. **環境セットアップ:**
   ```bash
   # Python 3.10の仮想環境を作成
   uv venv .venv --python 3.10
   source .venv/bin/activate

   # Modalのインストール
   uv pip install modal

   # Modalの認証
   modal setup
   ```

## 📂 データセットの準備

以下の構造でデータセットを準備してください：

```
dataset/
├── wavs/           # .wavファイルを含むディレクトリ
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── esd.list        # 転写ファイル（書き起こしデータ）
```

`dataset` フォルダを `dataset.zip` に圧縮します：

```bash
zip -r dataset.zip dataset/
```

## 🏋️‍♂️ トレーニング

1. **トレーニングの実行:**
   ```bash
   ./run_training.sh
   ```
   このスクリプトは以下の処理を行います：
   - Modal Volume (`sbv2-vol`) の作成
   - `dataset.zip` のアップロード
   - Modal上でのトレーニングパイプラインの開始
   - エラー発生時のログ (`training_error.txt`) のダウンロード

2. **進捗の監視:**
   ターミナルまたはModalのダッシュボードで進捗を確認できます。

3. **トレーニングのカスタマイズ:**
   `train_sbv2.py` を編集して、エポック数などのパラメータを変更できます：
   ```python
   @app.local_entrypoint()
   def main():
       # エポック数をここで変更 (例: 本番学習なら epochs=50)
       train_pipeline.remote("dataset.zip", epochs=2) 
   ```

## 📥 モデルのダウンロード

トレーニング完了後、Modal Volumeからモデルをダウンロードします：

```bash
# Volume内のファイル一覧を確認
modal volume ls sbv2-vol trained_models/MyStyleModel

# ローカルディレクトリにダウンロード
mkdir -p models
modal volume get sbv2-vol trained_models/MyStyleModel models/
```

## 🧹 クリーンアップ

仮想環境を削除する場合：
```bash
rm -rf .venv
```

## 📄 ライセンス

このプロジェクトは、オリジナルの [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) リポジトリのライセンスに従います。
