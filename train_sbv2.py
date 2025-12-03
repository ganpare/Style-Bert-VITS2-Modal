import modal
import os

# アプリ名定義
app = modal.App("style-bert-vits2-trainer")

# データの永続化領域
vol = modal.Volume.from_name("sbv2-vol", create_if_missing=True)

# 環境構築
# Style-Bert-VITS2に必要なライブラリを全て含めたイメージを作成
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel")
    # tzdata prompt回避
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "MKL_THREADING_LAYER": "GNU"})
    # システム依存パッケージ (espeak-ngが必須)
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "espeak-ng",
        "unzip",
        "pkg-config",
        "ffmpeg",
        "libavformat-dev",
        "libavdevice-dev",
        "libavcodec-dev",
        "libavfilter-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libavutil-dev",
    )
    # Python依存パッケージ
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "librosa",
        "matplotlib",
        "phonemizer",
        "pyopenjtalk",
        "pydantic",
        "transformers==4.38.2",
        "vector_quantize_pytorch",
        "safetensors",
        "num2words",
        "jieba",       # 中国語対応が含まれるため必要
        "cn2an"
    )
    # litagin02氏のリポジトリをクローン
    .run_commands("git clone https://github.com/litagin02/Style-Bert-VITS2.git /root/sbv2")
    # 追加のrequirementsをインストール
    # requirements.txtを直接使うとビルドエラーが出ることがあるため、必要なものを明示的にインストール
    .pip_install(
        "accelerate==0.27.2",
        # "av", # 依存関係解決のため一時的に除外
        "cmudict",
        # "faster-whisper", # 依存関係解決のため一時的に除外
        "pyannote.audio==3.1.1",  # style_gen.pyで必要
        "g2p_en",
        "GPUtil",
        "gradio>=4.32",
        "loguru",
        "nltk<=3.8.1",
        "numpy<2",
        "onnx",
        "onnxconverter-common",
        "onnxruntime-gpu",
        "onnxsim-prebuilt",
        "protobuf==4.25",
        "psutil",
        "punctuators",
        # "pyannote.audio>=3.1.0", # 依存関係解決のため一時的に除外
        "pyloudnorm",
        "pyopenjtalk-dict",
        "pypinyin",
        "pyworld-prebuilt",
        "stable_ts",
        "tensorboard",
        "umap-learn",
    )
)

# 定数設定
REPO_PATH = "/root/sbv2"
DATA_DIR = "/data"
MODEL_NAME = "MyStyleModel" # 任意のモデル名

@app.function(
    image=image,
    gpu="A10G",        # 学習にはA10G推奨
    volumes={DATA_DIR: vol},
    timeout=86400,     # 24時間
)
def train_pipeline(zip_filename: str, epochs: int = 2):
    import subprocess
    import shutil
    import yaml
    import json
    from pathlib import Path

    # 1. データの展開
    print(">>> Unzipping dataset...")
    zip_path = f"{DATA_DIR}/{zip_filename}"
    dataset_root = f"{DATA_DIR}/dataset_raw"
    
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    
    # unzipコマンドまたはpythonで解凍
    # apt_installでunzipを入れたのでコマンドが使えます
    subprocess.run(["unzip", "-q", zip_path, "-d", dataset_root], check=True)
    
    # 展開後のパス調整
    # dataset.zipを展開すると、dataset/wavs... となることを想定
    train_data_dir = f"{dataset_root}/dataset" 
    
    # もしzip直下にwavsがある場合などのケア（簡易的）
    if not os.path.exists(train_data_dir) and os.path.exists(f"{dataset_root}/wavs"):
        train_data_dir = dataset_root

    os.chdir(REPO_PATH)

    # 2. 前処理 (Resample, Transcribe, BERT feature extraction)
    
    # config.jsonの自動生成（簡易版）
    # 本来は config_template.json をベースに書き換えます
    config_path = f"{REPO_PATH}/Configs/config.json"
    
    print(">>> Starting Preprocessing...")
    
    internal_data_path = f"{REPO_PATH}/Data/{MODEL_NAME}"
    os.makedirs(internal_data_path, exist_ok=True)
    
    # wavsとesd.listを所定の位置にコピー
    subprocess.run(["cp", "-r", f"{train_data_dir}/wavs", f"{internal_data_path}/raw"])
    subprocess.run(["cp", f"{train_data_dir}/esd.list", f"{internal_data_path}/esd.list"])

    # 前処理実行 (Resample -> Spectrogram -> BERT)
    
    # step1: 初期化とconfig生成
    cmd_init = [
        "python", "initialize.py",
        "--dataset_root", f"{REPO_PATH}/Data"
    ]
    subprocess.run(cmd_init, check=True)

    # step2: 前処理 (BERT抽出含む)
    # jp_extra版の場合、bert抽出が必要です
    # preprocess_jp_extra.py は存在しないため、preprocess_all.py を使用します
    cmd_preprocess = [
        "python", "preprocess_all.py",
        "--model_name", MODEL_NAME,
        "--batch_size", "4",
        "--val_per_lang", "5", # 検証用データの数
        "--use_jp_extra"       # JP-Extraモデルを使用
    ]
    # BERTモデルのダウンロードが走るため初回は時間がかかります
    subprocess.run(cmd_preprocess, check=True)

    print(">>> Preprocessing Done. Starting Training...")
    
    # Debug: Check files in Data directory and read log if config is missing
    config_file = f"Data/{MODEL_NAME}/config.json"
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found!")
        print(f"Contents of {internal_data_path}:")
        if os.path.exists(internal_data_path):
            files = os.listdir(internal_data_path)
            print(files)
            # Find and read log file
            for f in files:
                if f.startswith("preprocess_") and f.endswith(".log"):
                    print(f">>> Reading {f}:")
                    with open(f"{internal_data_path}/{f}", "r", encoding="utf-8") as lf:
                        print(lf.read())
        else:
            print(f"{internal_data_path} does not exist!")
        raise FileNotFoundError(f"{config_file} not found. Preprocessing failed.")

    # 3. 学習実行
    # configパスは Data/{MODEL_NAME}/config.json に生成されています
    cmd_train = [
        "python", "train_ms_jp_extra.py",
        "--model", MODEL_NAME, # train_ms_jp_extra.pyの引数は --model
        "--config", f"Data/{MODEL_NAME}/config.json",
    ]
    
    # epochsを反映させるために config.json を編集
    with open(f"Data/{MODEL_NAME}/config.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    config_data["train"]["epochs"] = epochs
    config_data["train"]["batch_size"] = 4 # VRAMに合わせて調整
    
    with open(f"Data/{MODEL_NAME}/config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # train_ms_jp_extra.py がカレントディレクトリにモデルディレクトリがあることを期待している可能性があるため、シンボリックリンクを作成
    if not os.path.exists(MODEL_NAME):
        os.symlink(f"Data/{MODEL_NAME}", MODEL_NAME)

    print(">>> Starting Training Loop...")
    try:
        # リアルタイムで出力を表示するためcapture_output=Falseに変更
        subprocess.run(cmd_train, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(">>> Training Failed!")
        print(f"Training exited with status code: {e.returncode}")
        raise e
    
    # 4. モデルの保存
    # 学習済みモデル (model_assets/{MODEL_NAME}/*.safetensors) をVolumeに退避
    output_model_dir = f"{REPO_PATH}/model_assets/{MODEL_NAME}"
    backup_dir = f"{DATA_DIR}/trained_models/{MODEL_NAME}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f">>> Copying trained models to {backup_dir}...")
    # 最新のsafetensors等をコピー
    subprocess.run(f"cp -r {output_model_dir}/* {backup_dir}/", shell=True)
    print(">>> Training Complete!")

@app.local_entrypoint()
def main():
    # dataset.zipを渡して学習開始
    train_pipeline.remote("dataset.zip", epochs=2)
