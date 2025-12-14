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

**注意**: `dataset.zip` は自動的に作成されます。`dataset/` フォルダが存在すれば、`run_training.sh` が自動的にzip化します。

## ⚙️ トレーニング設定のカスタマイズ

`train_sbv2.py` のファイル冒頭にある固定値を編集してください：

```python
# ==================== 固定設定 ====================
DEFAULT_MODEL_NAME = "your_model"      # モデル名（Volume保存先にも使用）
MODAL_GPU = "H100"                     # GPU種類: "A10G", "A100", "H100" など
TRAIN_BATCH_SIZE = 24                  # 学習時のバッチサイズ
PREPROCESS_BATCH_SIZE = 4              # 前処理時のバッチサイズ
TRAIN_EPOCHS = 100                     # 総エポック数
SNAPSHOT_EVERY_EPOCHS = 5              # スナップショット間隔（何エポックごとに保存するか）
YOMI_ERROR = "skip"                    # 読みエラー時の動作: "raise" or "skip"
# ================================================
```

**重要な設定項目:**
- `DEFAULT_MODEL_NAME`: モデルの識別名。Volumeの保存先パスにも使用されます
- `MODAL_GPU`: 使用するGPU。高速化したい場合は `H100` を推奨
- `TRAIN_EPOCHS`: 学習する総エポック数
- `SNAPSHOT_EVERY_EPOCHS`: この間隔でチェックポイントをVolumeに保存します
- `TRAIN_BATCH_SIZE` vs `PREPROCESS_BATCH_SIZE`: 前者は学習本体、後者は前処理（BERT生成等）で使用

## 🏋️‍♂️ トレーニング

1. **トレーニングの実行:**
   ```bash
   bash run_training.sh
   ```
   
   このスクリプトは以下の処理を自動実行します：
   - Modal Volume (`sbv2-vol`) の作成
   - `dataset/` から `dataset.zip` を自動生成（存在しない場合）
   - `dataset.zip` のVolumeへのアップロード
   - **デタッチモード**でModal上のトレーニングパイプラインを開始
   - トレーニングログの取得（`training.log`）

2. **進捗の監視:**
   
   トレーニングはバックグラウンドで実行されます（デタッチモード）。進捗確認方法：
   
   ```bash
   # Modalダッシュボードで確認
   # または、コマンドラインでログをストリーム表示
   modal app logs <app-id>
   ```
   
   `run_training.sh` 実行時に表示されるURLからもダッシュボードで確認できます。

3. **継続学習（チェックポイントからの再開）:**
   
   このシステムは自動的に学習を継続できます：
   - `SNAPSHOT_EVERY_EPOCHS` ごとにチェックポイントがVolumeに保存されます
   - 次回 `run_training.sh` を実行すると、自動的に前回の続きから学習を再開します
   - **重要**: 継続学習したい場合、`modal volume rm` でモデルの保存先を削除しないでください
   
   ```bash
   # ❌ これをやると継続不可（最初からになる）
   modal volume rm -r sbv2-vol trained_models/your_model
   
   # ✅ 継続学習したいなら削除しない
   ```

## 📥 モデルのダウンロード

トレーニング完了後、Modal Volumeからモデルをダウンロードします：

```bash
# Volume内のファイル一覧を確認（モデル名は設定に合わせて変更）
modal volume ls sbv2-vol trained_models/your_model

# ローカルディレクトリにダウンロード
mkdir -p volume_dump/your_model
modal volume get sbv2-vol trained_models/your_model volume_dump/your_model --force
```

**ダウンロードされる主なファイル:**
- `checkpoints/G_*.pth`, `D_*.pth`, `WD_*.pth` - 学習中のチェックポイント
- `model_assets/*.safetensors` - 推論用モデル
- `model_assets/style_vectors.npy` - スタイルベクトル
- `config.json`, `train.list`, `val.list` - 設定ファイル

## 💾 Volume永続化について

**重要**: Modal Volumeに保存されたチェックポイントは、学習ジョブが終了しても残ります。

- ✅ 学習完了後もVolume内のデータは永続化されます
- ✅ 次回実行時に自動的にチェックポイントから継続できます
- ⚠️ `modal volume rm` で削除すると復元できなくなります
- 💡 複数モデルを並行して学習する場合は `DEFAULT_MODEL_NAME` を変えてください

## 🧹 クリーンアップ

仮想環境を削除する場合：
```bash
rm -rf .venv
```

## 📄 ライセンス

このプロジェクトは、オリジナルの [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) リポジトリのライセンスに従います。
