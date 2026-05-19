# CLAUDE.md — Marigold Mechanistic Interpretability Experiment

## Project Goal

This is a research project aimed at understanding **what fine-tuning actually does to Stable Diffusion's internal representations** when it is adapted for monocular depth estimation (as in Marigold).

The motivating mystery: Marigold fine-tunes SD2 for depth using a full diffusion objective over 1000 timesteps. A follow-up paper ("Fine-Tuning is Easier than You Think", arXiv:2409.11355) shows that a single-step deterministic fine-tune of the same backbone achieves comparable or better depth metrics. Neither paper explains *why*. We open the black box.

**This is not a paper about building a better depth estimator.**
It is a mechanistic interpretability paper: we probe internal U-Net features to understand what each layer and each timestep encodes about scene geometry, before and after fine-tuning.

**GPU execution:** All cluster GPU jobs must be submitted via **ACR** (`./scripts/acr_submit.sh`). See `docs/ACR.md`. Do not `ssh` + `nohup` to DGX unless the user confirms ACR is unavailable.

---

## Research Questions

**Q1 — Which layers encode geometry, and were they already geometry-aware before fine-tuning?**
Compare SD2 (vanilla) vs Marigold (fine-tuned): which layers gained geometric information, and which had it already?

**Q2 — Do denoising timesteps specialize in different depth properties?**
Does t≈1000 encode depth ordering (what's in front)? Does t≈100 encode boundary sharpness? Does t≈1 encode metric precision?

**Q3 — Multi-step vs single-step: same output, different internals?**
Does the training objective (diffusion loss vs regression loss) leave a structural difference in the learned representations, even when final depth metrics are similar?

---

## Model Variants (Baselines)

Train or download these four model variants:

| ID | Name | Description |
|---|---|---|
| A | SD2-vanilla | Stable Diffusion 2 base, no depth fine-tuning |
| B | Marigold-v1.1 | Official Marigold multi-step depth model (download from HuggingFace: `prs-eth/marigold-depth-v1-1`) |
| C | Single-step-regression | SD2 fine-tuned end-to-end with L1/SiLog loss, single forward pass at inference — train this from scratch |
| D | Marigold-1NFE | Marigold-v1.1 run with exactly 1 denoising step at inference (no retraining needed, just change inference steps=1) |

For Model C, use the same training data as Marigold (Hypersim + Virtual KITTI synthetic datasets).

---

## Datasets

**Training (for Model C only):**
- Hypersim: https://github.com/apple/ml-hypersim
- Virtual KITTI 2: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

**Evaluation (for probing — all models):**
- NYUv2 (indoor): standard split, 654 test images with dense GT depth
- KITTI Eigen split (outdoor): 697 test images
- ETH3D (mixed): for zero-shot generalization testing

Download scripts should be placed in `scripts/download_data.sh`.

---

## Project Structure

```
project/
├── CLAUDE.md                        # This file
├── README.md                        # Auto-generated experiment log
├── environment.yml                  # Conda environment
├── scripts/
│   ├── download_data.sh             # Dataset download
│   ├── download_models.sh           # HuggingFace model download
│   └── run_all.sh                   # End-to-end experiment runner
├── src/
│   ├── models/
│   │   ├── load_models.py           # Load all 4 model variants
│   │   └── train_single_step.py     # Train Model C from scratch
│   ├── extraction/
│   │   ├── feature_extractor.py     # Hook into U-Net layers, extract features
│   │   └── timestep_sampler.py      # Sample specific timesteps for extraction
│   ├── probing/
│   │   ├── probe_trainer.py         # Train linear probes on frozen features
│   │   ├── probe_tasks.py           # Define probing tasks (ordinal, normals, etc.)
│   │   └── probe_evaluator.py       # Evaluate probe accuracy
│   ├── analysis/
│   │   ├── weight_delta.py          # Compute per-layer weight change norms
│   │   └── compare_models.py        # Cross-model feature comparison
│   └── visualization/
│       ├── plot_heatmaps.py         # Layer × property probing heatmaps
│       ├── plot_timestep_curves.py  # Probing accuracy vs timestep
│       └── plot_weight_deltas.py    # Per-layer weight change visualization
├── experiments/
│   ├── exp01_weight_delta/          # Q1: which layers changed during fine-tuning
│   ├── exp02_layer_probing/         # Q1: layer-wise geometric probing
│   ├── exp03_timestep_probing/      # Q2: timestep specialization
│   └── exp04_multistep_vs_single/  # Q3: compare internals of B vs C vs D
└── results/
    ├── figures/                     # All paper figures
    └── tables/                      # Quantitative results as CSV
```

---

## Feature Extraction Protocol

### U-Net Hook Points

The SD2 U-Net has encoder, bottleneck, and decoder blocks at 4 resolutions (64×64, 32×32, 16×16, 8×8). Hook into the output of every ResNet block and every cross-attention block. This gives approximately 20–24 extraction points per forward pass.

Name hooks systematically: `encoder.block{i}.res{j}`, `encoder.block{i}.attn{j}`, `bottleneck.res`, `decoder.block{i}.res{j}`, etc.

### Timesteps to Sample

For multi-step models (B, D): extract features at t ∈ {1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10, 1}
For single-step models (A, C): single forward pass only (t=1 equivalent)

For Model A (vanilla SD): run the forward pass with a noisy depth-like input at each timestep to get activations. Use a random depth map encoded through the VAE as the noisy input.

### Feature Downsampling

All extracted feature maps should be bilinearly downsampled to a canonical 64×64 spatial resolution before probing. This normalizes across resolution blocks.

---

## Probing Tasks

Train a separate linear probe for each (layer, timestep, task) combination. Each probe is a single linear layer: `nn.Linear(feature_dim, output_dim)`.

| Task | Output | Loss | Metric |
|---|---|---|---|
| Ordinal depth ranking | Binary (A closer than B?) | BCE | Accuracy |
| Absolute depth | Per-pixel scalar | SiLog | AbsRel |
| Surface normals | Per-pixel 3D unit vector | Cosine | Mean angle error |
| Depth boundary | Binary edge map | BCE | F1 |
| Planar regions | Binary (planar vs. not) | BCE | IoU |

**Probe training protocol:**
- Use 80% of NYUv2 train set for probe training, 20% for validation
- Frozen backbone — only the probe trains
- Adam optimizer, lr=1e-3, 20 epochs max, early stopping
- Batch size 64 (feature vectors, not images)
- Report test accuracy on NYUv2 test set and KITTI

**Important:** probes should be small and fast. If a probe takes more than 5 minutes to train, something is wrong.

---

## Weight Delta Analysis

For Q1, compute per-layer weight change between SD2-vanilla (Model A) and Marigold (Model B). See `src/analysis/weight_delta.py`. Aggregate by block for visualization.

---

## Training Model C (Single-Step Regression)

Base: `stabilityai/stable-diffusion-2` U-Net. SiLog loss on depth latent, single forward at t=1. AdamW lr=3e-5, warmup 500 steps, batch 16 × grad accum 4, 30k steps, Hypersim 70% + VKitti 30%, bf16. Checkpoints every 5k → `checkpoints/model_C/`. wandb project `marigold-internals`, run `model-C-single-step`.

---

## Experiment Execution Order

### Step 0 — Setup
```bash
bash scripts/download_models.sh
bash scripts/download_data.sh
```

### Step 1 — Train Model C
```bash
python src/models/train_single_step.py --output_dir checkpoints/model_C --train_data hypersim,vkitti --steps 30000
```

### Step 2 — Exp 01
```bash
python experiments/exp01_weight_delta/run.py --model_a checkpoints/model_A_sd2 --model_b checkpoints/model_B_marigold --output results/exp01/
```

### Step 3 — Exp 02
```bash
python experiments/exp02_layer_probing/run.py --models A,B,C,D --dataset nyuv2 --tasks ordinal,depth,normals,boundary,planar --output results/exp02/
```

### Step 4 — Exp 03
```bash
python experiments/exp03_timestep_probing/run.py --model B --timesteps 1000,900,...,1 --tasks ordinal,depth,normals,boundary,planar --dataset nyuv2 --output results/exp03/
```

### Step 5 — Exp 04
```bash
python experiments/exp04_multistep_vs_single/run.py --models B,C,D --layer_comparison all --output results/exp04/
```

### Step 6 — Figures
```bash
python src/visualization/plot_all.py --results_dir results/ --output results/figures/
```

Also: `scripts/run_all.sh`, `scripts/check_status.py`.

---

## What Claude Code Should Do When You Start It

1. Read this file completely before doing anything else
2. Check which steps are already done (look for existing checkpoints and results CSVs)
3. Start from the first incomplete step
4. Ask the user before starting any training run longer than 1 hour — show estimated time first
5. After each experiment completes, update `README.md` with: what was run, key numerical results, any anomalies observed
6. If a step fails, diagnose the error, attempt one fix, and if still failing — write a clear description of the failure to `README.md` and ask the user before proceeding
7. Never delete existing results or checkpoints without asking
8. After all experiments complete, print a summary of the key findings across all 3 research questions

---

## Coding Conventions

- Python 3.10+, PyTorch 2.x, CUDA 12.4
- `diffusers` for model loading; `accelerate` for DGX
- `seed = 42`; wandb project `marigold-internals`
- Feature caches: HDF5 under `data/features/`
- Results → CSV first, figures from CSV
- `tqdm` + `rich` for logging

---

## References

- Marigold: arXiv:2312.02145, arXiv:2505.09358
- Fine-Tuning is Easier than You Think: arXiv:2409.11355
- Revelio: arXiv:2411.16725
- SD2: `stabilityai/stable-diffusion-2`
