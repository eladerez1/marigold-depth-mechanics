#!/usr/bin/env bash
# DEPRECATED — use ./scripts/acr_submit.sh (see docs/ACR.md).
# This script remains as a thin wrapper for backward compatibility.
set -euo pipefail

PROJ="/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics"
if command -v acr >/dev/null 2>&1; then
  echo "Redirecting to ACR (acr_submit.sh probing)..."
  exec "${PROJ}/scripts/acr_submit.sh" probing \
    --gpus 1 \
    --max-images "${MAX_IMAGES:-1000}" \
    --models "${MODELS:-B,D,A}"
fi
echo "WARNING: acr not installed — falling back to direct ssh-style run." >&2
echo "Install: ${PROJ}/scripts/acr_install.sh && acr init" >&2

PROJ="/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics"
GPU="${GPU:-3}"
PORT=8765
MAX_IMAGES="${MAX_IMAGES:-1000}"
MODELS="${MODELS:-B,D,A}"
LOG="${PROJ}/results/pipeline.log"
VIEWER_LOG="${PROJ}/results/viewer.log"

source ~/miniconda/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="/raid/homes/elad.e/.cache/huggingface"
export PYTHONPATH="${PROJ}:${PROJ}/third_party/Marigold:${PYTHONPATH:-}"
cd "${PROJ}"

mkdir -p results/figures results/exp01 results/exp02 results/exp03 results/exp04

# Viewer (idempotent)
if ! pgrep -f "viewer_server.py --port ${PORT}" >/dev/null 2>&1; then
  nohup conda run -n sd_visualizer python scripts/viewer_server.py --port "${PORT}" \
    > "${VIEWER_LOG}" 2>&1 &
  echo "Viewer started on port ${PORT} (log: ${VIEWER_LOG})"
else
  echo "Viewer already running on port ${PORT}"
fi

# Archive previous probing CSVs/figures before overwrite
ARCHIVE="${PROJ}/results/archive/probing_$(date +%Y%m%d_%H%M%S)"
if [[ -f "${PROJ}/results/exp02/probing_matrix.csv" ]]; then
  mkdir -p "${ARCHIVE}"
  cp -a "${PROJ}/results/exp02" "${PROJ}/results/exp03" "${PROJ}/results/exp04" "${ARCHIVE}/" 2>/dev/null || true
  cp -a "${PROJ}/results/figures"/fig{2,3,4}*.png "${ARCHIVE}/" 2>/dev/null || true
  echo "Archived prior probing results to ${ARCHIVE}"
fi

echo "Starting probing on GPU ${GPU} — ${MAX_IMAGES} NYU — models ${MODELS} — log: ${LOG}"
conda run --no-capture-output -n sd_visualizer \
  env PYTHONPATH="${PYTHONPATH}" \
  python scripts/run_gpu_pipeline.py \
  --gpu 0 \
  --max_images "${MAX_IMAGES}" \
  --denoise_steps 10 \
  --models "${MODELS}" \
  --skip-download \
  --probing-only \
  2>&1 | tee "${LOG}"

echo "Done. Open on laptop: ssh -L ${PORT}:localhost:${PORT} dgx04  →  http://localhost:${PORT}"
