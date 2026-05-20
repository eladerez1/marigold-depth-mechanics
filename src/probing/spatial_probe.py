"""Train/evaluate per-pixel linear probes on frozen U-Net feature maps."""

from __future__ import annotations

import random
from typing import Sequence

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.probing.probe_tasks import TASK_SPECS


def _align_feat_label(
    feat: torch.Tensor, label: torch.Tensor, task: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """feat [C,H,W], label spatial → flatten to [N,C], [N,1], mask [N]."""
    c, h, w = feat.shape
    feat = feat.float()
    if task == "ordinal":
        lh, lw = label.shape
        assert lh == h and lw == w - 1
        x = feat[:, :, :lw].permute(1, 2, 0).reshape(-1, c)
        y = label.reshape(-1, 1)
    else:
        assert label.shape == (h, w)
        x = feat.permute(1, 2, 0).reshape(-1, c)
        y = label.reshape(-1, 1)
    mask = torch.isfinite(y.squeeze(1))
    return x[mask], y[mask], mask


def _sample_pixels(
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] <= n:
        return x, y
    idx = torch.tensor(rng.sample(range(x.shape[0]), n), dtype=torch.long)
    return x[idx], y[idx]


def _compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "silog":
        log_diff = torch.log(pred.clamp(min=1e-3)) - torch.log(target.clamp(min=1e-3))
        return torch.sqrt((log_diff**2).mean() - 0.85 * (log_diff.mean() ** 2))
    raise ValueError(loss_type)


@torch.no_grad()
def _eval_metric(
    probe: nn.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_type: str,
    device: torch.device,
) -> dict[str, float]:
    probe.eval()
    x = x.float().to(device)
    y = y.float().to(device)
    pred = probe(x)
    if loss_type == "bce":
        acc = ((torch.sigmoid(pred) > 0.5).float() == y).float().mean().item()
        return {"val_metric": acc, "val_acc": acc}
    if loss_type == "mse":
        pred_d = pred.squeeze(-1)
        tgt = y.squeeze(-1)
        absrel = (torch.abs(pred_d - tgt) / tgt.abs().clamp(min=1e-3)).mean().item()
        score = 1.0 / (1.0 + absrel)
        return {"val_metric": score, "val_absrel": absrel}
    return {"val_metric": 0.0}


def train_spatial_probe(
    feat_maps: Sequence[torch.Tensor],
    label_maps: Sequence[torch.Tensor],
    task_name: str,
    train_idx: list[int],
    val_idx: list[int],
    *,
    max_pixels_per_image: int = 4096,
    max_epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 1e-3,
    patience: int = 6,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, float]:
    spec = TASK_SPECS[task_name]
    # Z-scored log-depth targets: MSE is appropriate (SiLog needs metric depth).
    loss_type = "mse" if task_name == "depth" else spec.loss
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    train_x, train_y = [], []
    for i in train_idx:
        xi, yi, _ = _align_feat_label(feat_maps[i], label_maps[i], task_name)
        xi, yi = _sample_pixels(xi, yi, max_pixels_per_image, rng)
        train_x.append(xi)
        train_y.append(yi)
    val_x, val_y = [], []
    val_cap = min(len(val_idx), max(64, 200_000 // max(max_pixels_per_image, 1)))
    for i in val_idx[:val_cap]:
        xi, yi, _ = _align_feat_label(feat_maps[i], label_maps[i], task_name)
        xi, yi = _sample_pixels(xi, yi, max_pixels_per_image, rng)
        val_x.append(xi)
        val_y.append(yi)

    x_train = torch.cat(train_x, dim=0)
    y_train = torch.cat(train_y, dim=0)
    x_val = torch.cat(val_x, dim=0)
    y_val = torch.cat(val_y, dim=0)
    if x_train.shape[0] < 64 or x_val.shape[0] < 32:
        return {"val_metric": 0.0, "skipped": 1.0}

    probe = nn.Linear(x_train.shape[1], spec.output_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train.float(), y_train.float()),
        batch_size=batch_size,
        shuffle=True,
    )

    best = float("inf")
    best_state = None
    stale = 0
    for _ in range(max_epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = _compute_loss(probe(xb), yb, loss_type)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_loss = _compute_loss(probe(x_val.to(device)), y_val.to(device), loss_type).item()
        if val_loss < best:
            best = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        probe.load_state_dict(best_state)
    metrics = _eval_metric(probe, x_val, y_val, loss_type, device)
    metrics["val_loss"] = best
    return metrics


def train_spatial_probes_for_model(
    model_id: str,
    feats: dict,
    labels_acc: dict[str, list[torch.Tensor]],
    n_images: int,
    device: str,
    *,
    layer_subsample: int = 12,
    tasks: tuple[str, ...] = ("ordinal", "depth", "boundary"),
) -> tuple[list, list]:
    """
    feats[layer][timestep] -> list of [C,H,W] per image.

    Exp02 row: best metric over timesteps per (layer, task).
    Exp03 row: best metric over layers per (timestep, task).
    """
    layer_names = sorted(feats.keys())
    if len(layer_names) > layer_subsample:
        layer_names = layer_names[:: max(1, len(layer_names) // layer_subsample)]

    split = max(1, int(n_images * 0.8))
    train_idx = list(range(split))
    val_idx = list(range(split, n_images))
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    exp02_rows: list = []
    exp03_rows: list = []
    dev = torch.device(device)
    # Scale down pixel budget so 1000-image runs finish in reasonable time.
    max_pixels = min(4096, max(512, 500_000 // max(n_images, 1)))

    for task_name in tasks:
        label_maps = labels_acc[task_name]
        if len(label_maps) != n_images:
            continue

        # layer -> task -> t -> metric
        layer_task_t: dict[str, dict[int, float]] = {layer: {} for layer in layer_names}
        t_layer: dict[int, dict[str, float]] = {}

        t_items = sorted(feats[layer_names[0]].items(), key=lambda x: x[0])
        if len(t_items) > 5 and n_images > 300:
            t_items = [t_items[0], t_items[len(t_items) // 2], t_items[-1]]
            print(
                f"spatial-{model_id}-{task_name}: subsampled {len(t_items)} timesteps "
                f"(n_images={n_images})",
                flush=True,
            )

        for layer in tqdm(layer_names, desc=f"spatial-{model_id}-{task_name}", leave=False):
            for t_int, feat_list in feats[layer].items():
                if len(feat_list) != n_images:
                    continue
                if n_images > 300 and t_int not in {ti for ti, _ in t_items}:
                    continue
                metrics = train_spatial_probe(
                    feat_list,
                    label_maps,
                    task_name,
                    train_idx,
                    val_idx,
                    max_pixels_per_image=max_pixels,
                    device=dev,
                    seed=hash((model_id, layer, t_int, task_name)) % (2**31),
                )
                if metrics.get("skipped"):
                    continue
                score = metrics["val_metric"]
                layer_task_t[layer][t_int] = score
                t_layer.setdefault(t_int, {})[layer] = score

        for layer in layer_names:
            if not layer_task_t[layer]:
                continue
            best_t = max(layer_task_t[layer], key=layer_task_t[layer].get)
            best_score = layer_task_t[layer][best_t]
            exp02_rows.append([model_id, layer, task_name, f"{best_score:.4f}", int(best_t)])

        for t_int, lscores in t_layer.items():
            if not lscores:
                continue
            best_layer = max(lscores, key=lscores.get)
            best_score = lscores[best_layer]
            exp03_rows.append([model_id, t_int, task_name, f"{best_score:.4f}", best_layer])

    return exp02_rows, exp03_rows
