#!/usr/bin/env bash
# End-to-end runner — stops at first incomplete step. GPU steps require confirmation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export MARIGOLD_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python scripts/check_status.py

if [[ ! -d checkpoints/model_A_sd2 ]]; then
  echo "Run: bash scripts/download_models.sh"
  exit 1
fi

if [[ ! -d checkpoints/model_C ]]; then
  echo "Model C not trained. Before running (~18-24h GPU):"
  echo "  python src/models/train_single_step.py --output_dir checkpoints/model_C"
  exit 1
fi

python experiments/exp01_weight_delta/run.py \
  --model_a checkpoints/model_A_sd2 \
  --model_b checkpoints/model_B_marigold \
  --output results/exp01/

python experiments/exp02_layer_probing/run.py \
  --models A,B,C,D --dataset nyuv2 \
  --tasks ordinal,depth,normals,boundary,planar \
  --output results/exp02/

python experiments/exp03_timestep_probing/run.py \
  --model B \
  --timesteps 1000,900,800,700,600,500,400,300,200,100,50,10,1 \
  --tasks ordinal,depth,normals,boundary,planar \
  --dataset nyuv2 --output results/exp03/

python experiments/exp04_multistep_vs_single/run.py \
  --models B,C,D --layer_comparison all --output results/exp04/

python src/visualization/plot_all.py --results_dir results/ --output results/figures/

echo "Pipeline complete."
