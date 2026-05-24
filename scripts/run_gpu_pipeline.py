#!/usr/bin/env python3
"""
Run experiments on GPU: download models, exp01, mini layer/timestep probing on NYU pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MARIGOLD_ROOT = ROOT / "third_party" / "Marigold"


def _load_module_from_path(module_name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ckpt_paths = _load_module_from_path(
    "marigold_checkpoint_paths", ROOT / "src" / "models" / "checkpoint_paths.py"
)
model_c_dir = _ckpt_paths.model_c_dir
model_c_ready = _ckpt_paths.model_c_ready

# Project root first for probing code; Marigold for pipeline.
sys.path.insert(0, str(MARIGOLD_ROOT))
sys.path.insert(0, str(ROOT))

from src.analysis.compare_models import linear_cka
from src.analysis.weight_delta import aggregate_by_block, compute_weight_delta
from src.extraction.feature_extractor import (
    FeatureExtractor,
    list_resnet_hook_layers,
    subsample_layers,
)
from src.models.load_models import load_unet_pair_for_delta
from src.probing.spatial_labels import make_spatial_probe_labels
from src.probing.spatial_probe import train_spatial_probes_for_model

NYU_RGB_ROOT = Path(
    "/isilon/Automotive/RnD/elad.e/Dev/research/sparse_confidence/datasets/nyu_raw/colmap_input"
)
NYU_DEPTH_ROOT = Path(
    "/isilon/Automotive/RnD/elad.e/Dev/research/sparse_confidence/datasets/nyu_raw/gt_depth"
)


def set_status(stage: str, **kwargs) -> None:
    path = ROOT / "results" / "run_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"stage": stage, **kwargs}
    if path.exists():
        data = {**json.loads(path.read_text()), **data}
    path.write_text(json.dumps(data, indent=2))


def _hf_home() -> str:
    import os

    return os.environ.get("HF_HOME", "/raid/homes/elad.e/.cache/huggingface")


def download_models() -> None:
    import os
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = _hf_home()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    ckpt_b = ROOT / "checkpoints" / "model_B_marigold"
    if (ckpt_b / "unet" / "config.json").exists():
        return
    set_status("downloading_models")
    # Prefer safetensors (smaller); skip legacy .bin to avoid xethub timeouts on DGX.
    if ckpt_b.is_symlink() or ckpt_b.exists():
        import shutil

        if ckpt_b.is_symlink():
            ckpt_b.unlink()
        elif ckpt_b.is_dir():
            shutil.rmtree(ckpt_b, ignore_errors=True)
    snapshot_download(
        repo_id="prs-eth/marigold-depth-v1-1",
        local_dir=str(ckpt_b),
        allow_patterns=[
            "model_index.json",
            "**/*.json",
            "**/*.txt",
            "**/*.safetensors",
        ],
    )
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="prs-eth/marigold-depth-v1-1",
        filename="model_index.json",
        local_dir=str(ckpt_b),
    )


def run_exp01() -> None:
    ckpt_a = ROOT / "checkpoints" / "model_A_sd2"
    ckpt_b = ROOT / "checkpoints" / "model_B_marigold"
    if not (ckpt_a / "unet" / "config.json").exists():
        set_status("exp01_skipped", reason="SD2 base not on disk (HF stabilityai/* 404 from DGX)")
        return
    set_status("exp01_weight_delta")
    out = ROOT / "results" / "exp01"
    out.mkdir(parents=True, exist_ok=True)
    unet_a, unet_b = load_unet_pair_for_delta(ckpt_a, ckpt_b)
    deltas = compute_weight_delta(unet_a, unet_b)
    block = aggregate_by_block(deltas)
    csv_path = out / "weight_delta_by_layer.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "relative_l2_delta"])
        for k, v in sorted(block.items()):
            w.writerow([k, f"{v:.6e}"])
    from src.visualization.plot_weight_deltas import plot_weight_deltas

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_weight_deltas(csv_path, fig_dir / "fig1_weight_delta.png")
    del unet_a, unet_b
    torch.cuda.empty_cache()


def collect_nyu_pairs(max_images: int) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for seq_dir in sorted(NYU_DEPTH_ROOT.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        rgb_dir = NYU_RGB_ROOT / seq / "images"
        if not rgb_dir.exists():
            continue
        for depth_path in sorted(seq_dir.glob("*.npy")):
            stem = depth_path.stem
            rgb_path = rgb_dir / f"{stem}.jpg"
            if rgb_path.exists():
                pairs.append((rgb_path, depth_path))
            if len(pairs) >= max_images:
                return pairs
    return pairs


def load_rgb_tensor(path: Path, size: int = 768) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def load_depth_tensor(path: Path, size: int = 64) -> torch.Tensor:
    d = np.load(path).astype(np.float32)
    if d.ndim == 3:
        d = d.squeeze()
    t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze()  # [H,W]


def _load_marigold_pipe(device: str):
    from src.models.marigold_pipe_loader import load_marigold_depth_pipeline

    return load_marigold_depth_pipeline(device, ROOT / "checkpoints" / "model_B_marigold")


def _preload_spatial_labels(pairs: list) -> dict[str, list]:
    labels_acc: dict = defaultdict(list)
    for _rgb_path, depth_path in pairs:
        labels = make_spatial_probe_labels(load_depth_tensor(depth_path))
        for task in ("ordinal", "depth", "boundary"):
            labels_acc[task].append(labels[task])
    return labels_acc


@torch.no_grad()
def _extract_marigold_one_layer(
    pipe,
    pairs: list,
    device: str,
    n_steps: int,
    layer_name: str,
    desc: str,
) -> dict:
    """feats[timestep] -> list of [C,H,W] per image for a single layer."""
    feats: dict = defaultdict(list)
    extractor = FeatureExtractor(pipe.unet, layers={layer_name})
    extractor.register_hooks()
    pipe.encode_empty_text()
    rng = torch.Generator(device=device)

    for rgb_path, _depth_path in tqdm(pairs, desc=desc, leave=False):
        rgb = load_rgb_tensor(rgb_path).to(device, dtype=pipe.dtype)
        rgb_latent = pipe.encode_rgb(rgb)
        target_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=pipe.dtype, generator=rng
        )
        batch_embed = pipe.empty_text_embed.repeat(rgb_latent.shape[0], 1, 1).to(device)
        pipe.scheduler.set_timesteps(n_steps, device=device)

        for t in pipe.scheduler.timesteps:
            extractor.clear_cache()
            unet_in = torch.cat([rgb_latent, target_latent], dim=1)
            noise_pred = pipe.unet(unet_in, t, encoder_hidden_states=batch_embed).sample
            tensor = extractor.get_latest().get(layer_name)
            if tensor is not None:
                feats[int(t.item())].append(tensor.squeeze(0).cpu())
            target_latent = pipe.scheduler.step(
                noise_pred, t, target_latent, generator=rng
            ).prev_sample

    extractor.remove_hooks()
    return {layer_name: feats}


def _probe_marigold_streaming(
    pipe,
    pairs: list,
    device: str,
    n_steps: int,
    model_id: str,
    n_images: int,
    target_layers: list[str],
) -> tuple[list, list]:
    labels = _preload_spatial_labels(pairs)
    exp02_rows: list = []
    exp03_rows: list = []
    for layer in tqdm(target_layers, desc=f"layers-{model_id}"):
        feats = _extract_marigold_one_layer(
            pipe, pairs, device, n_steps, layer, desc=f"extract-{model_id}"
        )
        r2, r3 = train_spatial_probes_for_model(
            model_id, feats, labels, n_images, device, layer_subsample=999
        )
        exp02_rows.extend(r2)
        exp03_rows.extend(r3)
        del feats
        torch.cuda.empty_cache()
    return exp02_rows, exp03_rows


def _load_existing_probe_rows(
    csv_path: Path, skip_models: set[str]
) -> tuple[list, list]:
    if not csv_path.exists():
        return [], []
    rows02, rows03 = [], []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") in skip_models:
                continue
            rows02.append(
                [
                    row["model"],
                    row["layer"],
                    row["task"],
                    row["metric_value"],
                    row.get("best_timestep", ""),
                ]
            )
    ts_path = csv_path.parent.parent / "exp03" / "timestep_curves.csv"
    if ts_path.exists():
        with ts_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") in skip_models:
                    continue
                rows03.append(
                    [
                        row["model"],
                        row["timestep"],
                        row["task"],
                        row["metric_value"],
                        row.get("best_layer", ""),
                    ]
                )
    return rows02, rows03


def run_probing(
    device: str,
    max_images: int,
    n_steps: int,
    models: str = "B,D,A",
    append: bool = False,
) -> None:
    from diffusers import UNet2DConditionModel

    from src.extraction.sd2_probing_forward import extract_sd2_one_layer

    model_list = [m.strip().upper() for m in models.split(",") if m.strip()]
    set_status("probing", max_images=max_images, n_steps=n_steps, models=model_list)

    pairs = collect_nyu_pairs(max_images)
    if not pairs:
        raise RuntimeError(f"No NYU pairs under {NYU_RGB_ROOT}")
    n_images = len(pairs)

    if "C" in model_list and not model_c_ready(ROOT / "checkpoints"):
        print("Model C not trained — skipping C (see src/models/train_single_step.py)")
        model_list = [m for m in model_list if m != "C"]

    pipe = _load_marigold_pipe(device)
    pipe.encode_empty_text()
    max_layers = 12 if n_images <= 200 else (6 if n_images <= 500 else 4)
    target_layers = subsample_layers(list_resnet_hook_layers(pipe.unet), max_layers=max_layers)
    print(f"Probing {n_images} images, {len(target_layers)} layers, models={model_list}", flush=True)

    all_feats: dict[str, dict] = {}
    out02 = ROOT / "results" / "exp02"
    out03 = ROOT / "results" / "exp03"
    skip = set(model_list) if append else set()
    exp02_rows, exp03_rows = (
        _load_existing_probe_rows(out02 / "probing_matrix.csv", skip)
        if append
        else ([], [])
    )

    for mid in model_list:
        if mid in ("B", "D"):
            steps = n_steps if mid == "B" else 1
            rows2, rows3 = _probe_marigold_streaming(
                pipe, pairs, device, steps, mid, n_images, target_layers
            )
            exp02_rows.extend(rows2)
            exp03_rows.extend(rows3)
            all_feats[mid] = {}
        elif mid == "A":
            ckpt_a = ROOT / "checkpoints" / "model_A_sd2"
            if not (ckpt_a / "unet" / "config.json").exists():
                print("Model A UNet missing — skip A")
                continue
            sd2_unet = UNet2DConditionModel.from_pretrained(
                ckpt_a, subfolder="unet", torch_dtype=torch.float16
            ).to(device)
            labels = _preload_spatial_labels(pairs)
            a_layers = subsample_layers(list_resnet_hook_layers(sd2_unet))
            for layer in tqdm(a_layers, desc="layers-A"):
                feats = extract_sd2_one_layer(
                    sd2_unet,
                    pipe,
                    pairs,
                    load_rgb_tensor,
                    load_depth_tensor,
                    layer,
                    n_steps,
                    device,
                    n_images,
                )
                r2, r3 = train_spatial_probes_for_model(
                    "A", feats, labels, n_images, device, layer_subsample=999
                )
                exp02_rows.extend(r2)
                exp03_rows.extend(r3)
                del feats
            del sd2_unet
            torch.cuda.empty_cache()
        elif mid == "C":
            pipe_c = _load_marigold_pipe(device)
            pipe_c.unet = UNet2DConditionModel.from_pretrained(
                model_c_dir(ROOT / "checkpoints"),
                subfolder="unet",
                torch_dtype=torch.float16,
            ).to(device)
            rows2, rows3 = _probe_marigold_streaming(
                pipe_c, pairs, device, 1, "C", n_images, target_layers
            )
            exp02_rows.extend(rows2)
            exp03_rows.extend(rows3)
            all_feats["C"] = {}

    out02.mkdir(parents=True, exist_ok=True)
    out03.mkdir(parents=True, exist_ok=True)

    with (out02 / "probing_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "task", "metric_value", "best_timestep"])
        w.writerows(exp02_rows)

    with (out03 / "timestep_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "timestep", "task", "metric_value", "best_layer"])
        w.writerows(exp03_rows)

    out04 = ROOT / "results" / "exp04"
    out04.mkdir(parents=True, exist_ok=True)
    with (out04 / "cka_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "layer", "cka"])
        feat_models = sorted(all_feats.keys())
        for i, ma in enumerate(feat_models):
            for mb in feat_models[i + 1 :]:
                common = sorted(set(all_feats[ma]) & set(all_feats[mb]))
                if len(common) > 12:
                    common = common[:: max(1, len(common) // 12)]
                for layer in common:
                    ts_a = sorted(all_feats[ma][layer].keys())
                    ts_b = sorted(all_feats[mb][layer].keys())
                    if not ts_a or not ts_b:
                        continue
                    fa = all_feats[ma][layer][ts_a[len(ts_a) // 2]]
                    fb = all_feats[mb][layer][ts_b[len(ts_b) // 2]]
                    xa = torch.stack([f.mean(dim=(1, 2)) for f in fa], dim=0)
                    xb = torch.stack([f.mean(dim=(1, 2)) for f in fb], dim=0)
                    if xa.shape == xb.shape:
                        w.writerow([ma, mb, layer, f"{linear_cka(xa, xb):.4f}"])

    import sys as _sys
    from src.visualization.plot_all import main as plot_all_main

    _argv = _sys.argv
    _sys.argv = ["plot_all", "--results_dir", str(ROOT / "results"), "--output", str(ROOT / "results" / "figures")]
    try:
        plot_all_main()
    finally:
        _sys.argv = _argv

    del pipe
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_images", type=int, default=40)
    p.add_argument("--denoise_steps", type=int, default=10)
    p.add_argument("--skip-download", "--skip_download", action="store_true", dest="skip_download")
    p.add_argument("--models", type=str, default="B,D,A", help="Comma-separated: A,B,C,D")
    p.add_argument(
        "--probing-only",
        "--probing_only",
        action="store_true",
        dest="probing_only",
        help="Skip model download and Exp01; only extract features and train probes.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Keep existing probe rows for models not in --models.",
    )
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    set_status("starting", device=device, max_images=args.max_images)
    if not args.skip_download:
        download_models()
    if not args.probing_only:
        run_exp01()
    run_probing(
        device,
        args.max_images,
        args.denoise_steps,
        models=args.models,
        append=args.append,
    )
    set_status("complete", device=device, max_images=args.max_images)


if __name__ == "__main__":
    main()
