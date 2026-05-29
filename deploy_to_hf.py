"""
HuggingFace deployment — clean version.
1. Upload ONNX files to model repo (5GB LFS limit, not Space)
2. Clean wrongly uploaded model files from Space
3. Re-upload code to Space with correct ignore patterns
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN      = os.environ["HF_TOKEN"]
API_KEY    = os.environ["API_KEY"]
MODEL_REPO = "AHFIDAILabs/immuniwatch-lora-classifier"
SPACE_REPO = "AHFIDAILabs/immuniwatch-ml-service"

from huggingface_hub import HfApi

api = HfApi(token=TOKEN)

# ------------------------------------------------------------------
# Step 1 — Upload ONNX files to MODEL repo (5 GB LFS limit)
# ------------------------------------------------------------------
print("\n=== Step 1: Uploading ONNX files to model repo ===")
onnx_files = [
    ("models/onnx/immuniwatch_classifier.onnx",      "immuniwatch_classifier.onnx"),
    ("models/onnx/immuniwatch_classifier.onnx.data", "immuniwatch_classifier.onnx.data"),
    ("models/onnx/model_config.json",                "model_config.json"),
    ("models/onnx/thresholds.json",                  "thresholds.json"),
]

for local, remote in onnx_files:
    p = Path(local)
    if not p.exists():
        print(f"  [SKIP] Not found: {local}")
        continue
    size_mb = p.stat().st_size / (1024 * 1024)
    print(f"  Uploading {remote} ({size_mb:.1f} MB) ...")
    sys.stdout.flush()
    try:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=MODEL_REPO,
            repo_type="model",
            token=TOKEN,
        )
        print(f"  [DONE] {remote}")
    except Exception as e:
        print(f"  [ERROR] {remote}: {e}")
    sys.stdout.flush()

# ------------------------------------------------------------------
# Step 2 — Delete wrongly uploaded model files from Space
# ------------------------------------------------------------------
print("\n=== Step 2: Cleaning model files from Space ===")
to_delete = [
    "models/onnx/immuniwatch_classifier.onnx",
    "models/onnx/model_config.json",
    "models/onnx/thresholds.json",
    "models/misinfo/production/v1/tokenizer.json",
    "models/knowledge_base/chroma.sqlite3",
]
for path in to_delete:
    try:
        api.delete_file(
            path_in_repo=path,
            repo_id=SPACE_REPO,
            repo_type="space",
            token=TOKEN,
        )
        print(f"  [DELETED] {path}")
    except Exception as e:
        print(f"  [SKIP] {path}: {e}")

# ------------------------------------------------------------------
# Step 3 — Re-upload code to Space with correct ignore patterns
# ------------------------------------------------------------------
print("\n=== Step 3: Uploading code to Space ===")
api.upload_folder(
    folder_path=".",
    repo_id=SPACE_REPO,
    repo_type="space",
    token=TOKEN,
    ignore_patterns=[
        "models/**", "venv/**", "__pycache__/**",
        "**/*.pyc", ".env", "data/**",
        "notebooks/**", "**/*.ipynb",
        ".pytest_cache/**", ".git/**",
        "temp_test.py", "docker-compose.yml",
        "deploy_to_hf.py",
        "**/*.onnx", "**/*.onnx.data",
        "**/*.safetensors", "**/*.bin",
        "**/*.pt", "**/*.pkl",
    ],
)
print("  [DONE] Code uploaded")

# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("DEPLOYMENT COMPLETE")
print("=" * 55)
print(f"Space : https://huggingface.co/spaces/{SPACE_REPO}")
print(f"URL   : https://ahfidailabs-immuniwatch-ml-service.hf.space")
print(f"Health: https://ahfidailabs-immuniwatch-ml-service.hf.space/health")
print("=" * 55)
print("Docker build starts automatically (~5 min).")
print("First startup downloads 1GB ONNX from model repo (~3-5 min).")
print("After that the service is fast and ready.")
print("=" * 55)
