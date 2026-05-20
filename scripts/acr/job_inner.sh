#!/usr/bin/env bash
# Runs inside ACR container (/workspace = cloned git repo).
set -euo pipefail

PROJ="${MARIGOLD_PROJ:-/workspace}"
ISILON_ROOT="${MARIGOLD_ISILON_ROOT:-/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics}"
export HF_HOME="${HF_HOME:-/isilon/Automotive/RnD/elad.e/.cache/huggingface}"
export PYTHONPATH="${PROJ}/third_party/Marigold:${PROJ}:${PYTHONPATH:-}"

cd "${PROJ}"

# Checkpoints, prior results, and NYU paths live on isilon (not in git).
_link_isilon() {
  local name
  for name in checkpoints data; do
    if [[ -d "${ISILON_ROOT}/${name}" ]]; then
      rm -rf "${PROJ}/${name}" 2>/dev/null || true
      ln -sfn "${ISILON_ROOT}/${name}" "${PROJ}/${name}"
    fi
  done
  mkdir -p "${ISILON_ROOT}/results"
  rm -rf "${PROJ}/results" 2>/dev/null || true
  ln -sfn "${ISILON_ROOT}/results" "${PROJ}/results"
}
_link_isilon

LOG_DIR="${PROJ}/results"
mkdir -p "${LOG_DIR}/archive" "${LOG_DIR}/figures"

_run_python() {
  python3 "$@"
}

case "${ACR_JOB:-probing}" in
  probing)
    ARCHIVE="${LOG_DIR}/archive/probing_$(date +%Y%m%d_%H%M%S)"
    if [[ -f "${LOG_DIR}/exp02/probing_matrix.csv" ]]; then
      mkdir -p "${ARCHIVE}"
      cp -a "${LOG_DIR}/exp02" "${LOG_DIR}/exp03" "${LOG_DIR}/exp04" "${ARCHIVE}/" 2>/dev/null || true
      cp -a "${LOG_DIR}/figures"/fig{2,3,4}*.png "${ARCHIVE}/" 2>/dev/null || true
      echo "Archived prior probing → ${ARCHIVE}"
    fi
    LOG="${LOG_DIR}/spatial_probing_acr.log"
    _run_python scripts/run_gpu_pipeline.py \
      --gpu 0 \
      --max_images "${MAX_IMAGES:-200}" \
      --denoise_steps "${DENOISE_STEPS:-10}" \
      --models "${MODELS:-B,D,A}" \
      --skip-download \
      --probing-only \
      2>&1 | tee "${LOG}"
    ;;

  full)
    LOG="${LOG_DIR}/pipeline.log"
    _run_python scripts/run_gpu_pipeline.py \
      --gpu 0 \
      --max_images "${MAX_IMAGES:-1000}" \
      --denoise_steps "${DENOISE_STEPS:-10}" \
      --models "${MODELS:-B,D,A}" \
      2>&1 | tee "${LOG}"
    ;;

  export_denoise)
    LOG="${LOG_DIR}/denoise_export_acr.log"
    _run_python scripts/export_denoise_trajectory.py \
      --nyu_index "${NYU_INDEX:-0}" \
      --num_samples "${NUM_SAMPLES:-5}" \
      --n_steps "${DENOISE_STEPS:-10}" \
      --gpu 0 \
      2>&1 | tee "${LOG}"
    ;;

  train_model_c)
    LOG="${LOG_DIR}/train_model_c_acr.log"
    EXTRA_ARGS=(--no_wandb)
    if [[ -n "${MARIGOLD_BASE_DATA_DIR:-}" ]]; then
      EXTRA_ARGS+=(--data_root "${MARIGOLD_BASE_DATA_DIR}")
    fi
    _run_python src/models/train_single_step.py \
      --output_dir "${PROJ}/results/model_C" \
      --train_data "${TRAIN_DATA:-nyu}" \
      --steps "${TRAIN_STEPS:-30000}" \
      --batch_size 2 \
      --grad_accum 4 \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "${LOG}"
    ;;

  finish)
    LOG="${LOG_DIR}/finish_acr.log"
    {
      echo "=== train Model C ==="
      EXTRA_ARGS=(--no_wandb)
      if [[ -n "${MARIGOLD_BASE_DATA_DIR:-}" ]]; then
        EXTRA_ARGS+=(--data_root "${MARIGOLD_BASE_DATA_DIR}")
      fi
      _run_python src/models/train_single_step.py \
        --output_dir "${PROJ}/results/model_C" \
        --train_data "${TRAIN_DATA:-nyu}" \
        --steps "${TRAIN_STEPS:-30000}" \
        --batch_size 2 \
        --grad_accum 4 \
        "${EXTRA_ARGS[@]}"

      echo "=== probe Model C (append to exp02/03) ==="
      _run_python scripts/run_gpu_pipeline.py \
        --gpu 0 \
        --max_images "${MAX_IMAGES:-1000}" \
        --denoise_steps "${DENOISE_STEPS:-10}" \
        --models "C" \
        --skip-download \
        --probing-only \
        --append
    } 2>&1 | tee "${LOG}"
    ;;

  *)
    echo "Unknown ACR_JOB=${ACR_JOB} (probing | full | export_denoise | train_model_c | finish)" >&2
    exit 1
    ;;
esac

echo "ACR job ${ACR_JOB} finished."
