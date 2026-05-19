"""PCA RGB maps from U-Net feature tensors (Plug-and-Play diffusion style)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _percentile_bounds(flat: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(flat, lo, axis=0), np.percentile(flat, hi, axis=0)


def fit_pca_on_trajectory(feat_maps: np.ndarray) -> tuple[PCA, np.ndarray, np.ndarray]:
    """
    Args:
        feat_maps: [T, C, H, W] float32
    Returns:
        fitted PCA, low_bounds [3], high_bounds [3] (global across timesteps)
    """
    t, c, h, w = feat_maps.shape
    x = feat_maps.transpose(0, 2, 3, 1).reshape(-1, c).astype(np.float64)
    pca = PCA(n_components=3, random_state=0)
    pca.fit(x)
    proj = pca.transform(x).reshape(t, h, w, 3)
    lo, hi = _percentile_bounds(proj.reshape(-1, 3))
    return pca, lo, hi


def feature_map_to_pca_rgb(
    feat: np.ndarray,
    pca: PCA,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """
    Args:
        feat: [C, H, W]
    Returns:
        uint8 RGB [H, W, 3]
    """
    c, h, w = feat.shape
    x = feat.transpose(1, 2, 0).reshape(-1, c).astype(np.float64)
    rgb = pca.transform(x).reshape(h, w, 3)
    span = np.maximum(hi - lo, 1e-6)
    rgb = (rgb - lo) / span
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)
