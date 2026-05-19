# Marigold Depth Mechanics — Experiment Log

**Project root:** `/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics`

Mechanistic interpretability study: what fine-tuning does to SD2 internal representations for depth (Marigold). See `CLAUDE.md` for the full protocol.

## GPU jobs (ACR — required)

All DGX GPU runs go through **ACR** ([uv-algo-compute-orc](https://github.com/UVeye/uv-algo-compute-orc)). See **`docs/ACR.md`**.

```bash
./scripts/acr_install.sh    # once
acr init                      # once (VPN on)
# Push repo to GitHub, then:
export MARIGOLD_ACR_REPO=git@github.com:UVeye/<repo>.git
./scripts/acr_submit.sh export_denoise --num-samples 10
./scripts/acr_submit.sh status
```

Do not use `ssh` + `nohup` on DGX unless ACR is down.

## Quick start (no GPU)

```bash
cd /isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics
conda env create -f environment.yml && conda activate marigold-internals
bash scripts/download_models.sh
bash scripts/download_data.sh
python scripts/check_status.py
python scripts/smoke_test_cpu.py
```

## Status (2026-05-17)

| Step | Description | Status |
|------|-------------|--------|
| 0 | Marigold model on disk | Done (DGX04) |
| 1 | Train Model C | Not run |
| 2 | Exp02 layer probing (B, 40 NYUv2 imgs) | Done |
| 3 | Exp03 timestep probing | Done |
| 4 | Exp04 CKA (single pair) | Done |
| exp01 | SD2 vs Marigold weight delta | Done (from `stable-diffusion-2.tar`) |
| Viewer | `scripts/viewer_server.py` port 8765 | Running on dgx04 |

### View results in browser (laptop)

```bash
ssh -L 8765:localhost:8765 dgx04
```

Open **http://localhost:8765**

## Layout

```
marigold_depth_mechanics/
├── CLAUDE.md
├── README.md
├── environment.yml
├── scripts/
├── src/
│   ├── models/
│   ├── extraction/
│   ├── probing/
│   ├── analysis/
│   ├── visualization/
│   └── data/
├── experiments/
│   ├── exp01_weight_delta/
│   ├── exp02_layer_probing/
│   ├── exp03_timestep_probing/
│   └── exp04_multistep_vs_single/
├── checkpoints/
├── data/
└── results/
```

## Key results

*(none yet)*
