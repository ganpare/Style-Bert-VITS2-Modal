import modal
import os
import re

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
DEFAULT_MODEL_NAME = "rumine"  # ここを編集してモデル名を固定


def _validate_model_name(model_name: str) -> str:
    # パスやコマンドに混ざると危険な文字を避ける（Modal上でもローカルでも同じルール）
    # 例: "CharA_20251214" / "my-model.v1" などはOK
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", model_name):
        raise ValueError(
            "MODEL_NAME must match: [A-Za-z0-9][A-Za-z0-9._-]{0,63} (no spaces or slashes)"
        )
    return model_name

# ===== ここを編集して実行設定を固定（環境変数は使わない） =====
# GPU（例: "A10G", "A100", "H100" など。台数は "H100:2" のように指定）
MODAL_GPU = "H100"
# 学習設定
TRAIN_BATCH_SIZE = 24
PREPROCESS_BATCH_SIZE = 4
TRAIN_EPOCHS = 100
YOMI_ERROR = "skip"
SNAPSHOT_EVERY_EPOCHS = 5
# Volume からの復元は常に有効
RESUME_FROM_VOLUME = "1"

@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={DATA_DIR: vol},
    timeout=86400,     # 24時間
)
def train_pipeline(zip_filename: str, model_name: str, epochs: int = 2):
    import subprocess
    import shutil
    import yaml
    import json
    from pathlib import Path
    import re

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
    
    model_name = _validate_model_name(model_name)
    internal_data_path = f"{REPO_PATH}/Data/{model_name}"
    os.makedirs(internal_data_path, exist_ok=True)
    
    # wavsとesd.listを所定の位置にコピー
    subprocess.run(["cp", "-r", f"{train_data_dir}/wavs", f"{internal_data_path}/raw"])
    subprocess.run(["cp", f"{train_data_dir}/esd.list", f"{internal_data_path}/esd.list"])

    # esd.list の音声パスを補正
    # preprocess_text.py の --correct_path は「Data/<model>/wavs/<utt>」を前提にするため、
    # 先頭が "wavs/" の場合は二重になって Audio not found になり得ます。
    # よくある入力例:
    # - "wavs/0001.wav|spk|JP|text" -> "0001.wav|..."
    # - "0001.wav|spk|JP|text" -> そのまま
    esd_path = Path(internal_data_path) / "esd.list"
    if esd_path.exists():
        fixed_lines: list[str] = []
        changed = 0
        for raw_line in esd_path.read_text(encoding="utf-8").splitlines(True):
            line = raw_line.rstrip("\n")
            if not line.strip():
                fixed_lines.append(raw_line)
                continue
            parts = line.split("|")
            if len(parts) < 1:
                fixed_lines.append(raw_line)
                continue
            utt = parts[0]

            # Windows絶対パス（例: C:\\...）は触らない（Modal/Linux上では基本的に使えないが、意図せぬ破壊を避ける）
            looks_windows_abs = bool(re.match(r"^[A-Za-z]:[\\\\/]", utt))
            if not (utt.startswith("/") or looks_windows_abs):
                utt_norm = utt.replace("\\\\", "/")
                if utt_norm.startswith("./"):
                    utt_norm = utt_norm[2:]
                if utt_norm.startswith("dataset/"):
                    utt_norm = utt_norm[len("dataset/"):]
                if utt_norm.startswith("wavs/"):
                    utt_norm = utt_norm[len("wavs/"):]
                # 先頭以外のディレクトリ構造は保持（例: styleA/0001.wav）
                if utt_norm != utt:
                    parts[0] = utt_norm
                    changed += 1
            fixed_lines.append("|".join(parts) + "\n")
        if changed:
            esd_path.write_text("".join(fixed_lines), encoding="utf-8")
            print(f">>> Fixed esd.list paths: {changed} lines")

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
        "--model_name", model_name,
        "--batch_size", str(PREPROCESS_BATCH_SIZE),
        "--val_per_lang", "5", # 検証用データの数
        "--yomi_error", YOMI_ERROR,
        "--use_jp_extra"       # JP-Extraモデルを使用
    ]
    # BERTモデルのダウンロードが走るため初回は時間がかかります
    subprocess.run(cmd_preprocess, check=True)

    # 前処理が内部で失敗しても exit code 0 の場合があるため、最低限の成果物を検証
    train_list_path = Path(internal_data_path) / "train.list"
    val_list_path = Path(internal_data_path) / "val.list"
    wavs_dir = Path(internal_data_path) / "wavs"

    def _nonempty(path: Path) -> bool:
        try:
            return path.exists() and path.stat().st_size > 0
        except Exception:
            return False

    npy_count = 0
    if wavs_dir.exists():
        npy_count = sum(1 for _ in wavs_dir.rglob("*.npy"))

    if not _nonempty(train_list_path):
        raise RuntimeError(
            f"Preprocess seems failed: {train_list_path} is missing/empty. "
            f"Try setting YOMI_ERROR=skip (or use) to avoid stopping on rare chars."
        )
    if npy_count == 0:
        raise RuntimeError(
            f"Preprocess seems failed: no style .npy files found under {wavs_dir}. "
            f"Try setting YOMI_ERROR=skip (or use), or check Data/{model_name}/text_error.log."
        )

    # Volume に残してある途中経過があれば復元（preprocess後に上書きしてresume用チェックポイントを活かす）
    def _restore_from_volume():
        def _truthy(v: str) -> bool:
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

        if not _truthy(RESUME_FROM_VOLUME):
            print(">>> RESUME_FROM_VOLUME is disabled. Skipping restore.")
            return

        vol_root = Path(DATA_DIR) / "trained_models" / model_name
        dst_models = Path(REPO_PATH) / "Data" / model_name / "models"
        dst_assets = Path(REPO_PATH) / "model_assets" / model_name

        def _copytree_if_exists(src: Path, dst: Path):
            if src.exists() and src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)

        def _copy_assets_from_root_if_present(src_root: Path, dst: Path):
            # 旧挙動/手動コピーで /data/trained_models/<model>/ 直下に
            # *.safetensors や config.json が置かれているケースに対応
            if not (src_root.exists() and src_root.is_dir()):
                return
            dst.mkdir(parents=True, exist_ok=True)
            copied = 0
            for p in src_root.iterdir():
                if not p.is_file():
                    continue
                if p.name in {"config.json", "style_vectors.npy"} or p.suffix in {".safetensors", ".npy"}:
                    shutil.copy2(p, dst / p.name)
                    copied += 1
            if copied:
                print(f">>> Restored {copied} model asset files from Volume root")

        # チェックポイントの場所は運用で揺れがちなので複数候補を探す
        ckpt_candidates = [
            vol_root / "checkpoints",
            vol_root / "models",
        ]
        restored_ckpt = False
        for vol_ckpt in ckpt_candidates:
            if vol_ckpt.exists() and vol_ckpt.is_dir():
                pth_files = sorted(vol_ckpt.glob("*.pth"))
                safetensor_files = sorted(vol_ckpt.glob("*.safetensors"))
                if pth_files or safetensor_files:
                    print(">>> Restoring checkpoints from Volume (resume if available)...")
                    if pth_files:
                        print(f">>> Found {len(pth_files)} .pth in {vol_ckpt} (latest: {pth_files[-1].name})")
                    if safetensor_files:
                        print(f">>> Found {len(safetensor_files)} .safetensors in {vol_ckpt}")
                _copytree_if_exists(vol_ckpt, dst_models)
                restored_ckpt = True
                break

        if not restored_ckpt:
            print(">>> No checkpoints found in Volume. Starting fresh (pretrained init only).")

        # model_assets は通常 /model_assets 配下に保存するが、直下にあるケースも救済
        vol_assets = vol_root / "model_assets"
        if vol_assets.exists():
            _copytree_if_exists(vol_assets, dst_assets)
        _copy_assets_from_root_if_present(vol_root, dst_assets)

    _restore_from_volume()

    print(">>> Preprocessing Done. Starting Training...")
    
    # Debug: Check files in Data directory and read log if config is missing
    config_file = f"Data/{model_name}/config.json"
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
        "--model", model_name, # train_ms_jp_extra.pyの引数は --model
        "--config", f"Data/{model_name}/config.json",
    ]
    
    def _sync_to_volume(reason: str):
        # 途中終了に備えて、学習途中の成果物もVolumeへ退避する
        # - checkpoints: Data/<model>/models
        # - infer assets: model_assets/<model>
        # - logs/config: Data/<model> の主要ファイル
        src_models = Path(REPO_PATH) / "Data" / model_name / "models"
        src_assets = Path(REPO_PATH) / "model_assets" / model_name
        src_data_root = Path(REPO_PATH) / "Data" / model_name

        dst_root = Path(DATA_DIR) / "trained_models" / model_name
        dst_ckpt = dst_root / "checkpoints"
        dst_assets = dst_root / "model_assets"
        dst_root.mkdir(parents=True, exist_ok=True)

        def _copytree_if_exists(src: Path, dst: Path):
            if src.exists() and src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)

        def _copyfile_if_exists(src: Path, dst: Path):
            if src.exists() and src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        _copytree_if_exists(src_models, dst_ckpt)
        _copytree_if_exists(src_assets, dst_assets)
        for name in ["config.json", "train.list", "val.list", "text_error.log", "esd.list"]:
            _copyfile_if_exists(src_data_root / name, dst_root / name)

        # できればコミットして、別ターミナルからも見えるようにする
        try:
            vol.commit()
            print(f">>> Snapshot committed to Volume ({reason})")
        except Exception as e:
            print(f">>> Snapshot copy done but commit failed/ignored: {e}")

    # epochs/batch_size を反映させるために config.json を編集
    config_json_path = Path(REPO_PATH) / "Data" / model_name / "config.json"
    with open(config_json_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config_data["train"]["batch_size"] = TRAIN_BATCH_SIZE
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # train_ms_jp_extra.py がカレントディレクトリにモデルディレクトリがあることを期待している可能性があるため、シンボリックリンクを作成
    if not os.path.exists(model_name):
        os.symlink(f"Data/{model_name}", model_name)

    # 10エポックごと等で区切って実行し、その都度Volumeへ退避する
    # train_ms_jp_extra.py は checkpoints があれば自動で resume するため、分割実行が可能
    total_epochs = int(epochs)
    step_epochs = int(SNAPSHOT_EVERY_EPOCHS)
    if step_epochs <= 0:
        step_epochs = total_epochs

    print(">>> Starting Training Loop...")
    current_target = 0
    while current_target < total_epochs:
        current_target = min(total_epochs, current_target + step_epochs)

        with open(config_json_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config_data["train"]["epochs"] = current_target
        config_data["train"]["batch_size"] = TRAIN_BATCH_SIZE
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        print(f">>> Training until epoch={current_target} (snapshot interval={step_epochs})")
        try:
            # リアルタイムで出力を表示
            subprocess.run(cmd_train, check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(">>> Training Failed!")
            print(f"Training exited with status code: {e.returncode}")
            _sync_to_volume(reason=f"failed_at_target_epoch_{current_target}")
            raise e

        _sync_to_volume(reason=f"after_epoch_{current_target}")
    
    # 4. モデルの保存
    # 学習済みモデル (model_assets/{MODEL_NAME}/*.safetensors) をVolumeに退避
    output_model_dir = f"{REPO_PATH}/model_assets/{model_name}"
    backup_dir = f"{DATA_DIR}/trained_models/{model_name}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f">>> Copying trained models to {backup_dir}...")
    # 最新のsafetensors等をコピー
    subprocess.run(f"cp -r {output_model_dir}/* {backup_dir}/", shell=True)
    print(">>> Training Complete!")

@app.local_entrypoint()
def main():
    # dataset.zipを渡して学習開始
    # MODEL_NAME を指定すると Volume 内の保存先や学習名を切り替えられます
    model_name = _validate_model_name(DEFAULT_MODEL_NAME)
    train_pipeline.remote("dataset.zip", model_name=model_name, epochs=TRAIN_EPOCHS)
