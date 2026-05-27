#!/usr/bin/env python3
"""
Fine-tune Marigold on 3DRealCar RGB-D dataset.

3DRealCar provides dense, metric depth maps captured with iPhone 14 ARKit
(ToF LiDAR), giving per-pixel depth at ~1-3cm accuracy.

Starting point: pre-trained Marigold (prs-eth/marigold-depth-v1-1).
Loss: scale-shift invariant MSE on log-depth (MiDaS style).
     Since depth is dense and metric we could also use plain L1,
     but SSI is more robust to scale drift across diverse car distances.

Usage (ACR):
  ACR_JOB=train_3drealcar bash /workspace/scripts/acr/job_inner.sh

Usage (local):
  python scripts/train_marigold_3drealcar.py --data_root /data/3DRealCar --steps 5000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MARIGOLD_ROOT = ROOT / "third_party" / "Marigold"
sys.path.insert(0, str(MARIGOLD_ROOT))

from marigold import MarigoldDepthPipeline  # noqa: E402
from diffusers import DDIMScheduler        # noqa: E402

from src.dataset.threedreal_car_dataset import ThreeDRealCarDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_TRAIN_TIMESTEP = 999


def scale_shift_invariant_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    loss_total = torch.tensor(0.0, device=pred.device)
    n_valid = 0
    for b in range(pred.shape[0]):
        m = mask[b, 0]
        if m.sum() < 10:
            continue
        p = pred[b, 0][m].clamp(min=eps)
        t = target[b, 0][m].clamp(min=eps)
        diff = torch.log(p) - torch.log(t)
        s = diff.mean()
        loss_total = loss_total + ((diff - s) ** 2).mean()
        n_valid += 1
    return loss_total / max(n_valid, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None,
                   help="Root of 3DRealCar dataset (default: /workspace/data/3drealcar)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--base_ckpt", type=str, default="prs-eth/marigold-depth-v1-1")
    p.add_argument("--lighting", type=str, default="standard",
                   help="Filter lighting condition: standard | reflective | dark | all")
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="Lower LR for 3DRealCar — dense metric depth, careful fine-tune")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--image_size", type=str, default="480,640")
    p.add_argument("--max_depth_m", type=float, default=10.0)
    p.add_argument("--min_valid_frac", type=float, default=0.10)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return max(0.0, 1.0 - progress)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # --- Dataset ---
    H, W = map(int, args.image_size.split(","))
    data_root = Path(args.data_root) if args.data_root else Path("/workspace/data/3drealcar")
    lighting = None if args.lighting == "all" else args.lighting

    dataset = ThreeDRealCarDataset(
        root=data_root,
        image_size=(H, W),
        lighting=lighting,
        max_depth_m=args.max_depth_m,
        min_valid_frac=args.min_valid_frac,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {data_root}")
    log.info("Dataset: %d samples (lighting=%s)", len(dataset), lighting)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Model ---
    hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    log.info("Loading Marigold from %s", args.base_ckpt)
    pipe = MarigoldDepthPipeline.from_pretrained(
        args.base_ckpt, torch_dtype=torch.float32
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.train()
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    if args.freeze_encoder:
        frozen = ("down_blocks", "mid_block", "conv_in")
        n_f = n_t = 0
        for name, param in pipe.unet.named_parameters():
            if any(name.startswith(f) for f in frozen):
                param.requires_grad_(False); n_f += param.numel()
            else:
                n_t += param.numel()
        log.info("Encoder frozen: %dM frozen, %dM trainable", n_f // 1e6, n_t // 1e6)

    # --- Optimizer ---
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    opt = AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = LambdaLR(opt, lr_lambda=lambda s: _lr_lambda(s, args.warmup_steps, args.steps))

    pipe.encode_empty_text()
    empty_embed = pipe.empty_text_embed.to(device)

    # --- Output dir ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif "ISILON_PATH" in os.environ:
        out_dir = Path(os.environ["ISILON_PATH"]) / "results" / "model_3drealcar"
    else:
        out_dir = ROOT / "results" / "model_3drealcar"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints: %s", out_dir)

    # --- Training loop ---
    global_step = 0
    running_loss = 0.0
    opt.zero_grad()

    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break

            rgb = batch["rgb"].to(device)
            depth_gt = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            B = rgb.shape[0]

            rgb_norm = rgb * 2.0 - 1.0
            with torch.no_grad():
                rgb_lat = pipe.vae.encode(rgb_norm).latent_dist.sample() * pipe.vae.config.scaling_factor

            depth_norm = torch.zeros_like(depth_gt)
            for b in range(B):
                m = mask[b, 0]
                if m.sum() > 0:
                    d_min = depth_gt[b, 0][m].min()
                    d_max = depth_gt[b, 0][m].max()
                    rng = (d_max - d_min).clamp(min=1e-4)
                    depth_norm[b] = (depth_gt[b] - d_min) / rng * 2.0 - 1.0

            depth_norm_3ch = depth_norm.repeat(1, 3, 1, 1)
            with torch.no_grad():
                depth_lat = pipe.vae.encode(depth_norm_3ch).latent_dist.sample() * pipe.vae.config.scaling_factor

            t = torch.tensor([_TRAIN_TIMESTEP] * B, device=device)
            noise = torch.randn_like(depth_lat)
            noisy_depth_lat = pipe.scheduler.add_noise(depth_lat, noise, t)
            unet_in = torch.cat([rgb_lat, noisy_depth_lat], dim=1)

            text_embed = empty_embed.expand(B, -1, -1)
            pred_noise = pipe.unet(unet_in, t, encoder_hidden_states=text_embed).sample

            pred_depth_lat = pipe.scheduler.step(
                pred_noise, _TRAIN_TIMESTEP, noisy_depth_lat
            ).pred_original_sample
            pred_depth_img = pipe.vae.decode(
                pred_depth_lat / pipe.vae.config.scaling_factor
            ).sample
            pred_depth = pred_depth_img[:, :1]

            if mask.shape[-2:] != pred_depth.shape[-2:]:
                mask_r = F.interpolate(mask.float(), size=pred_depth.shape[-2:], mode="nearest").bool()
            else:
                mask_r = mask

            loss = scale_shift_invariant_loss(pred_depth, depth_gt, mask_r)
            (loss / args.grad_accum).backward()
            running_loss += loss.item()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

            global_step += 1

            if global_step % args.log_every == 0:
                avg = running_loss / args.log_every
                running_loss = 0.0
                log.info("step=%d/%d  loss=%.4f  lr=%.2e  valid_frac=%.2f",
                         global_step, args.steps, avg, sched.get_last_lr()[0],
                         batch["valid_frac"].mean().item())

            if global_step % args.save_every == 0 or global_step == args.steps:
                ckpt = out_dir / f"checkpoint-{global_step}"
                pipe.unet.save_pretrained(ckpt / "unet")
                log.info("Saved: %s", ckpt)

    log.info("Done. Final step %d", global_step)


if __name__ == "__main__":
    train(parse_args())
