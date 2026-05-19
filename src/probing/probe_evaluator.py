"""Evaluate trained linear probes on test features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def evaluate_probe(
    probe: nn.Linear,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    loss_type: str,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or next(probe.parameters()).device
    x_test = x_test.float().to(device)
    y_test = y_test.float().to(device)
    probe.eval()
    pred = probe(x_test)
    out: dict[str, float] = {"test_loss": _loss(pred, y_test, loss_type).item()}
    if loss_type == "bce":
        bin_pred = (torch.sigmoid(pred) > 0.5).float()
        out["test_acc"] = (bin_pred == y_test).float().mean().item()
    return out


def _loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return F.mse_loss(pred, target)
