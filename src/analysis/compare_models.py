"""CKA and cross-model feature similarity."""

from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Centered Kernel Alignment between two feature matrices [N, D]."""
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    hsic_xy = (x.T @ y).norm() ** 2
    hsic_xx = (x.T @ x).norm() ** 2
    hsic_yy = (y.T @ y).norm() ** 2
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-8)).item()
