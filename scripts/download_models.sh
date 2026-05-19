#!/usr/bin/env bash
# Download Model A (SD2) and Model B (Marigold) to local checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MARIGOLD_ROOT="${ROOT}"
CKPT="${ROOT}/checkpoints"
mkdir -p "${CKPT}"
cd "${ROOT}"

echo "==> Model A: stabilityai/stable-diffusion-2 (UNet + VAE for weight-delta / hooks)"
python - <<'PY'
from huggingface_hub import snapshot_download
import os

root = os.environ.get("MARIGOLD_ROOT", ".")
ckpt = os.path.join(root, "checkpoints", "model_A_sd2")
os.makedirs(ckpt, exist_ok=True)
snapshot_download(
    repo_id="stabilityai/stable-diffusion-2",
    local_dir=ckpt,
    allow_patterns=["unet/*", "vae/*", "scheduler/*", "model_index.json"],
    local_dir_use_symlinks=False,
)
print("Model A saved to", ckpt)
PY

echo "==> Model B: prs-eth/marigold-depth-v1-1"
python - <<'PY'
from huggingface_hub import snapshot_download
import os

root = os.environ.get("MARIGOLD_ROOT", ".")
ckpt = os.path.join(root, "checkpoints", "model_B_marigold")
os.makedirs(ckpt, exist_ok=True)
snapshot_download(
    repo_id="prs-eth/marigold-depth-v1-1",
    local_dir=ckpt,
    local_dir_use_symlinks=False,
)
print("Model B saved to", ckpt)
PY

echo "Done. Models under ${CKPT}/model_A_sd2 and ${CKPT}/model_B_marigold"
