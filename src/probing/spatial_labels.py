"""Per-pixel probe targets at 64×64 (aligned with feature extractor resolution)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PROBE_SIZE = 64


def make_spatial_probe_labels(depth: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Build spatial targets from GT depth [H, W] (typically 64×64).

    Returns:
        depth: [H, W] z-scored log-depth (invalid → nan)
        boundary: [H, W] binary edge map
        ordinal: [H, W-1] horizontal closer-farther (invalid pairs → nan)
        valid: [H, W] bool mask
    """
    if depth.dim() != 2:
        raise ValueError(f"depth must be [H,W], got {depth.shape}")

    valid = torch.isfinite(depth) & (depth > 1e-3)
    d = depth.clone()
    d[~valid] = float("nan")

    log_d = torch.log(depth.clamp(min=1e-3))
    if valid.any():
        ref = log_d[valid]
        mu = ref.median()
        std = ref.std().clamp(min=1e-4)
        depth_y = (log_d - mu) / std
    else:
        depth_y = log_d * 0.0
    depth_y[~valid] = float("nan")

    # Sobel-like gradients for boundaries (per-image adaptive threshold).
    d_fill = depth.clone()
    d_fill[~valid] = depth[valid].median() if valid.any() else 0.0
    gx = F.pad((d_fill[:, 1:] - d_fill[:, :-1]).abs(), (0, 1), value=0)
    gy = F.pad((d_fill[1:, :] - d_fill[:-1, :]).abs(), (0, 0, 0, 1), value=0)
    grad = torch.sqrt(gx**2 + gy**2)
    if valid.any():
        thr = torch.quantile(grad[valid], 0.75)
    else:
        thr = grad.mean()
    boundary = (grad > thr).float()
    boundary[~valid] = float("nan")

    ord_h = (d_fill[:, :-1] < d_fill[:, 1:]).float()
    pair_valid = valid[:, :-1] & valid[:, 1:]
    ord_h[~pair_valid] = float("nan")

    return {
        "depth": depth_y,
        "boundary": boundary,
        "ordinal": ord_h,
        "valid": valid.float(),
    }
