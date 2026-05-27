"""
Feature geometry visualization: PCA and UMAP of depth-predictive layer activations.

For each model we take the best depth-predictive layer (from exp05), extract per-pixel
feature vectors, project to 2D with PCA and UMAP, and scatter-plot coloured by depth.
This reveals whether depth information lives on a linear subspace or a curved manifold.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA


def _best_mlp_layer_per_model(exp05_csv: Path) -> dict[str, str]:
    """Return {model: layer} with highest MLP depth R² from exp05."""
    best: dict[str, tuple[float, str]] = {}
    with exp05_csv.open() as f:
        for r in csv.DictReader(f):
            if r["task"] != "depth":
                continue
            v = float(r["metric_value"])
            if r["model"] not in best or v > best[r["model"]][0]:
                best[r["model"]] = (v, r["layer"])
    return {m: layer for m, (_, layer) in best.items()}


def _sample_pixels(
    feat: torch.Tensor,
    depth: torch.Tensor,
    n: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """feat [C,H,W], depth [H,W] → sampled (X [n,C], d [n]) arrays."""
    c, h, w = feat.shape
    feat_flat = feat.float().permute(1, 2, 0).reshape(-1, c).numpy()
    d_flat = depth.reshape(-1).numpy()
    mask = np.isfinite(d_flat) & (d_flat > 0)
    feat_flat = feat_flat[mask]
    d_flat = d_flat[mask]
    if len(feat_flat) > n:
        idx = rng.sample(range(len(feat_flat)), n)
        feat_flat = feat_flat[idx]
        d_flat = d_flat[idx]
    return feat_flat, d_flat


def run_feature_geometry(
    device: str,
    root: Path,
    max_images: int = 200,
    pixels_per_image: int = 256,
    n_pca_components: int = 50,
    seed: int = 42,
) -> None:
    """Extract features, run PCA (+ UMAP if available), save fig6."""
    import sys
    sys.path.insert(0, str(root / "third_party" / "Marigold"))
    sys.path.insert(0, str(root))

    from scripts.run_gpu_pipeline import (
        collect_nyu_pairs,
        load_rgb_tensor,
        load_depth_tensor,
        _load_marigold_pipe,
        _extract_marigold_one_layer,
    )
    from src.models.checkpoint_paths import model_c_dir
    from src.extraction.feature_extractor import FeatureExtractor
    from diffusers import UNet2DConditionModel

    exp05_csv = root / "results" / "exp05" / "mlp_probing.csv"
    if not exp05_csv.exists():
        print("exp05/mlp_probing.csv not found — run MLP probing first")
        return

    best_layers = _best_mlp_layer_per_model(exp05_csv)
    print("Best depth layers:", best_layers)

    pairs = collect_nyu_pairs(max_images)
    n_images = len(pairs)
    rng = random.Random(seed)

    # Try to import UMAP; fall back gracefully
    try:
        from umap import UMAP
        use_umap = True
    except ImportError:
        print("umap-learn not installed — PCA only (pip install umap-learn)")
        use_umap = False

    model_configs = {
        "A": ("sd2", None),
        "B": ("marigold", None),
        "C": ("marigold_c", None),
        "D": ("marigold_1step", None),
    }

    all_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    pipe = _load_marigold_pipe(device)
    pipe.encode_empty_text()
    gen_rng = torch.Generator(device=device)

    for mid in ["A", "B", "C", "D"]:
        layer = best_layers.get(mid)
        if layer is None:
            print(f"No best layer for model {mid} — skip")
            continue
        print(f"\nModel {mid}: extracting layer {layer}", flush=True)

        if mid == "A":
            ckpt_a = root / "checkpoints" / "model_A_sd2"
            if not (ckpt_a / "unet" / "config.json").exists():
                print("  Model A UNet missing — skip")
                continue
            unet = UNet2DConditionModel.from_pretrained(
                ckpt_a, subfolder="unet", torch_dtype=torch.float16
            ).to(device)
            working_pipe = pipe
            working_pipe.unet = unet
            n_steps = 10
        elif mid == "B":
            working_pipe = _load_marigold_pipe(device)
            working_pipe.encode_empty_text()
            n_steps = 10
        elif mid == "C":
            working_pipe = _load_marigold_pipe(device)
            working_pipe.unet = UNet2DConditionModel.from_pretrained(
                model_c_dir(root / "checkpoints"),
                subfolder="unet",
                torch_dtype=torch.float16,
            ).to(device)
            working_pipe.encode_empty_text()
            n_steps = 1
        elif mid == "D":
            working_pipe = _load_marigold_pipe(device)
            working_pipe.encode_empty_text()
            n_steps = 1

        feats_all = []
        depths_all = []

        layer_data = _extract_marigold_one_layer(
            working_pipe, pairs, device, n_steps, layer,
            desc=f"extract-{mid}",
        )
        layer_feats = layer_data.get(layer, {})
        ts = sorted(layer_feats.keys())
        if not ts:
            print(f"  No features extracted for layer {layer}")
            continue
        rep_t = ts[len(ts) // 2]
        feat_list = layer_feats[rep_t]  # list of [C,H,W] tensors, one per image

        for i, (feat_tensor, (_, depth_path)) in enumerate(zip(feat_list, pairs)):
            depth = load_depth_tensor(depth_path, size=feat_tensor.shape[1])
            x, d = _sample_pixels(feat_tensor, depth, pixels_per_image, rng)
            feats_all.append(x)
            depths_all.append(d)

        X = np.concatenate(feats_all, axis=0)
        D = np.concatenate(depths_all, axis=0)
        print(f"  {X.shape[0]} pixel samples, {X.shape[1]} dims")
        all_data[mid] = (X, D)

        # Free GPU memory
        del working_pipe
        torch.cuda.empty_cache()

    if not all_data:
        print("No data collected")
        return

    # --- Plot ---
    n_models = len(all_data)
    n_cols = 2 if use_umap else 1
    fig, axes = plt.subplots(
        n_models, n_cols + 1,
        figsize=(5 * (n_cols + 1), 4 * n_models),
        squeeze=False,
    )
    fig.suptitle("Feature geometry at best depth-predictive layer", fontsize=14)

    cmap = plt.cm.plasma

    for row, (mid, (X, D)) in enumerate(sorted(all_data.items())):
        layer_short = best_layers[mid].replace("res::", "").replace("_blocks", "").replace(".resnets", "")

        # PCA
        pca = PCA(n_components=min(n_pca_components, X.shape[1]))
        X_pca = pca.fit_transform(X)
        var2 = pca.explained_variance_ratio_[:2].sum()

        ax = axes[row][0]
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=D, cmap=cmap, s=1, alpha=0.4, rasterized=True)
        plt.colorbar(sc, ax=ax, label="depth")
        ax.set_title(f"Model {mid} PCA  [{layer_short}]\nPC1+PC2 var={var2:.1%}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Explained variance bar
        ax2 = axes[row][1]
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumvar) + 1), cumvar, marker=".")
        ax2.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90%")
        ax2.set_xlabel("PCs")
        ax2.set_ylabel("Cumulative var")
        ax2.set_title(f"Model {mid} scree")
        ax2.legend(fontsize=8)

        if use_umap:
            reducer = UMAP(n_components=2, random_state=seed, n_jobs=4)
            X_umap = reducer.fit_transform(X_pca[:, :20])
            ax3 = axes[row][2]
            sc3 = ax3.scatter(X_umap[:, 0], X_umap[:, 1], c=D, cmap=cmap, s=1, alpha=0.4, rasterized=True)
            plt.colorbar(sc3, ax=ax3, label="depth")
            ax3.set_title(f"Model {mid} UMAP (on 20 PCs)")
            ax3.set_xlabel("UMAP-1")
            ax3.set_ylabel("UMAP-2")

    plt.tight_layout()
    out = root / "results" / "figures" / "fig6_feature_geometry.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close()

    # Also save a depth-correlation-by-PC plot
    fig2, axes2 = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 4))
    if len(all_data) == 1:
        axes2 = [axes2]
    fig2.suptitle("Pearson |r| with depth per principal component", fontsize=13)
    for ax, (mid, (X, D)) in zip(axes2, sorted(all_data.items())):
        pca2 = PCA(n_components=min(30, X.shape[1]))
        X_pca2 = pca2.fit_transform(X)
        correlations = [
            abs(float(np.corrcoef(X_pca2[:, i], D)[0, 1]))
            for i in range(X_pca2.shape[1])
        ]
        ax.bar(range(1, len(correlations) + 1), correlations)
        ax.set_xlabel("PC index")
        ax.set_ylabel("|r| with depth")
        ax.set_title(f"Model {mid}")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    out2 = root / "results" / "figures" / "fig6b_depth_pc_correlation.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close()
