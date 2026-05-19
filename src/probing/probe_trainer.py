"""Train small linear probes on frozen features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _silog_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_diff = torch.log(pred.clamp(min=1e-3)) - torch.log(target.clamp(min=1e-3))
    return torch.sqrt((log_diff**2).mean() - 0.85 * (log_diff.mean() ** 2))


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    output_dim: int = 1,
    loss_type: str = "bce",
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 20,
    patience: int = 5,
    device: torch.device | None = None,
) -> tuple[nn.Linear, dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = nn.Linear(x_train.shape[1], output_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(x_train.float(), y_train.float()),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val = x_val.float().to(device)
    y_val = y_val.float().to(device)

    best_val = float("inf")
    best_state = None
    stale = 0

    for _ in range(max_epochs):
        probe.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = probe(xb)
            loss = _compute_loss(pred, yb, loss_type)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_loss = _compute_loss(probe(x_val), y_val, loss_type).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        probe.load_state_dict(best_state)
    metrics = {"val_loss": best_val}
    if loss_type == "bce":
        with torch.no_grad():
            pred = (torch.sigmoid(probe(x_val)) > 0.5).float()
            metrics["val_acc"] = (pred == y_val).float().mean().item()
    return probe, metrics


def _compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    if loss_type == "silog":
        return _silog_loss(pred.squeeze(-1), target.squeeze(-1))
    raise ValueError(loss_type)
