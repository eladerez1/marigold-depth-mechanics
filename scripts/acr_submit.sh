#!/usr/bin/env bash
# Submit Marigold GPU work via ACR (uv-algo-compute-orc).
#
# REQUIRED: code must be in git. Set MARIGOLD_ACR_REPO before submit.
#
# One-time: scripts/acr_install.sh && acr init  (see docs/ACR.md)
#
# Examples:
#   export MARIGOLD_ACR_REPO=git@github.com:UVeye/marigold-depth-mechanics.git
#   export MARIGOLD_ACR_BRANCH=main
#   ./scripts/acr_submit.sh export_denoise --num-samples 10
#   ./scripts/acr_submit.sh probing --max-images 200 --models B,D,A
#
set -euo pipefail

PROJ="/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics"
ISILON_RESULTS="${PROJ}/results"
DGX_USER="${MARIGOLD_DGX_USER:-elad.e}"

ACR_REPO="${MARIGOLD_ACR_REPO:-${ACR_REPO:-}}"
ACR_BRANCH="${MARIGOLD_ACR_BRANCH:-${ACR_BRANCH:-main}}"

NODE="${ACR_NODE:-}"
GPU_TYPE="${ACR_GPU_TYPE:-}"
GPUS="${ACR_GPUS:-1}"
JOB="probing"
MAX_IMAGES=200
MODELS="B,D,A"
DENOISE_STEPS=10
NYU_INDEX=0
NUM_SAMPLES=5
TRAIN_DATA="${MARIGOLD_TRAIN_DATA:-nyu}"
TRAIN_STEPS=30000
EXTRA_ACR_ARGS=()

usage() {
  sed -n '2,14p' "$0"
  echo ""
  echo "Usage: $0 <probing|full|export_denoise|train_model_c|finish|status|jobs|logs> [options]"
  echo "  Requires: MARIGOLD_ACR_REPO=git@github.com:UVeye/<repo>.git"
  echo "  --branch BRANCH   default: main (or set MARIGOLD_ACR_BRANCH)"
  echo "  --node NODE       optional pin (never dgx04)"
  echo "  --gpu-type TYPE   optional: v100, a6000, t4"
  echo "  --gpus N          default: 1"
  echo "  --max-images N    probing/full"
  echo "  --models LIST     e.g. B,D,A"
  echo "  --denoise-steps N"
  echo "  --num-samples N   export_denoise"
  echo "  --nyu-index N     export_denoise"
  echo "  --train-data STR  train_model_c: nyu | hypersim,vkitti"
  echo "  --steps N         train_model_c optimization steps"
  echo "  --job-id ID       for logs (acr logs -f)"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi
JOB_CMD="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node) NODE="$2"; shift 2 ;;
    --branch) ACR_BRANCH="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --denoise-steps) DENOISE_STEPS="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --nyu-index) NYU_INDEX="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --steps) TRAIN_STEPS="$2"; shift 2 ;;
    --job-id) JOB_ID="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA_ACR_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${NODE}" == "dgx04" ]]; then
  echo "ERROR: dgx04 is not allowed for this project." >&2
  exit 1
fi

if ! command -v acr >/dev/null 2>&1; then
  echo "ERROR: acr CLI not found. Run: ${PROJ}/scripts/acr_install.sh" >&2
  exit 1
fi

case "${JOB_CMD}" in
  status) exec acr status ;;
  jobs)   exec acr jobs "${EXTRA_ACR_ARGS[@]}" ;;
  quota)  exec acr quota ;;
  logs)
    [[ -n "${JOB_ID:-}" ]] || { echo "Pass --job-id"; exit 1; }
    exec acr logs "${JOB_ID}" -f
    ;;
  probing|full|export_denoise|train_model_c|finish)
    JOB="${JOB_CMD}"
    ;;
  *)
    echo "Unknown command: ${JOB_CMD}" >&2
    usage
    ;;
esac

if [[ -z "${ACR_REPO}" ]]; then
  echo "ERROR: MARIGOLD_ACR_REPO is required (ACR blocks bare 'acr run' without --repo)." >&2
  echo "  export MARIGOLD_ACR_REPO=git@github.com:UVeye/<your-repo>.git" >&2
  echo "  Push this project from isilon first — see docs/ACR.md" >&2
  exit 1
fi

REMOTE_CMD="bash /workspace/scripts/acr/job_inner.sh"

if [[ -n "${NODE}" ]]; then
  echo "Submitting ACR job: ${JOB} on ${NODE} (${GPUS} GPU) repo=${ACR_REPO}@${ACR_BRANCH}"
else
  echo "Submitting ACR job: ${JOB} — scheduler picks node (${GPUS} GPU) repo=${ACR_REPO}@${ACR_BRANCH}"
fi
echo "  max_images=${MAX_IMAGES} models=${MODELS} denoise_steps=${DENOISE_STEPS}"
[[ "${JOB}" == "train_model_c" || "${JOB}" == "finish" ]] && echo "  train_data=${TRAIN_DATA} steps=${TRAIN_STEPS}"
[[ -n "${GPU_TYPE}" ]] && echo "  gpu_type=${GPU_TYPE}"

ACR_ARGS=(
  run "${REMOTE_CMD}"
  --gpus "${GPUS}"
  --repo "${ACR_REPO}"
  --branch "${ACR_BRANCH}"
  --output-dir "${ISILON_RESULTS}"
  -e "ACR_JOB=${JOB}"
  -e "MARIGOLD_PROJ=/workspace"
  -e "MARIGOLD_ISILON_ROOT=${PROJ}"
  -e "MARIGOLD_DGX_USER=${DGX_USER}"
  -e "MAX_IMAGES=${MAX_IMAGES}"
  -e "MODELS=${MODELS}"
  -e "DENOISE_STEPS=${DENOISE_STEPS}"
  -e "NYU_INDEX=${NYU_INDEX}"
  -e "NUM_SAMPLES=${NUM_SAMPLES}"
  -e "TRAIN_DATA=${TRAIN_DATA}"
  -e "TRAIN_STEPS=${TRAIN_STEPS}"
)

[[ -n "${NODE}" ]] && ACR_ARGS+=(--node "${NODE}")
[[ -n "${GPU_TYPE}" ]] && ACR_ARGS+=(--gpu-type "${GPU_TYPE}")

acr "${ACR_ARGS[@]}" "${EXTRA_ACR_ARGS[@]}"

echo ""
echo "Monitor:  acr jobs    acr logs <job_id> -f    acr quota"
