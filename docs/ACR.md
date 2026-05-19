# GPU jobs via ACR

All GPU work runs through **[ACR](https://github.com/UVeye/uv-algo-compute-orc)**. The cluster **requires** `--repo` (git URL): do not submit `bash /isilon/.../job_inner.sh` directly.

**Never use dgx04** for this project.

## One-time setup

```bash
./scripts/acr_install.sh
acr init
```

During `acr init`, mount at least `/isilon:/isilon` and `/raid/homes.elad.e:/raid/homes/elad.e`.

## Push code to GitHub (required)

The project on isilon is not a git repo until you initialize it:

```bash
cd /isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics
git init
git add .
git commit -m "Marigold depth mechanics — ACR onboarding"
# Create empty repo on GitHub (UVeye org), then:
git remote add origin git@github.com:UVeye/<repo-name>.git
git push -u origin main
```

Set the repo URL for submits:

```bash
export MARIGOLD_ACR_REPO=git@github.com:UVeye/<repo-name>.git
export MARIGOLD_ACR_BRANCH=main
```

Heavy assets stay on isilon (not in git): `checkpoints/`, `data/`, `results/`. The job symlinks them into `/workspace` at runtime (see `.acr.yaml`).

## Submit jobs

```bash
export MARIGOLD_ACR_REPO=git@github.com:UVeye/<repo-name>.git

# Scheduler picks node + GPU
./scripts/acr_submit.sh export_denoise --num-samples 10
./scripts/acr_submit.sh probing --max-images 200 --models B,D,A

# Optional pin
./scripts/acr_submit.sh probing --node dgx01 --gpu-type v100

./scripts/acr_submit.sh status
./scripts/acr_submit.sh jobs
./scripts/acr_submit.sh logs --job-id <id>
```

ACR clones the repo to `/workspace`, builds from `Dockerfile` + `.acr.yaml`, and runs `scripts/acr/job_inner.sh`.

## What changed (why jobs were cancelled)

| Old (wrong) | New (required) |
|-------------|----------------|
| `acr run "bash /isilon/.../job_inner.sh"` | `acr run "bash scripts/acr/job_inner.sh" --repo <git-url> --branch main` |
| Default nvcr pytorch + pip on isilon | Repo `Dockerfile` (clean CUDA + torch 2.3) |
| No `.acr.yaml` | `.acr.yaml` at repo root |

## Dashboard

- http://10.1.1.136:8082 — token: `cat ~/.config/acr/token`
