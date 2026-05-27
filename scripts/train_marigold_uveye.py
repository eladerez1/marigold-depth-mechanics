#!/usr/bin/env python3
"""
Fine-tune Marigold on UVeye inspection-lane images with Pi3-derived depth pseudo-labels.

Starting point: pre-trained Marigold B (prs-eth/marigold-depth-v1-1).
Depth supervision: sparse depth maps projected from Pi3 per-frame car PLY files,
using known camera intrinsics and world-to-camera poses.

Loss: scale-shift invariant MSE on log-depth over valid (Pi3-observed) pixels only.
This is appropriate for sparse pseudo-GT since we don't trust absolute depth scale
but do trust relative depth structure from Pi3's 3D reconstruction.

Usage (ACR):
  ACR_JOB=train_uveye bash /workspace/scripts/acr/job_inner.sh

Usage (local):
  python scripts/train_marigold_uveye.py --steps 2000 --batch_size 1
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

from src.dataset.uveye_pi3_dataset import UVeyePi3Dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_TRAIN_TIMESTEP = 999


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def scale_shift_invariant_loss(
    pred: torch.Tensor,  # [B, 1, H, W]
    target: torch.Tensor,  # [B, 1, H, W]
    mask: torch.Tensor,  # [B, 1, H, W] bool
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Scale-and-shift invariant log-depth loss (Ranftl et al. MiDaS style).
    Only evaluated at valid (mask=True) pixels.
    """
    loss_total = torch.tensor(0.0, device=pred.device)
    n_valid = 0
    for b in range(pred.shape[0]):
        m = mask[b, 0]
        if m.sum() < 10:
            continue
        p = pred[b, 0][m].clamp(min=eps)
        t = target[b, 0][m].clamp(min=eps)

        log_p = torch.log(p)
        log_t = torch.log(t)
        diff = log_p - log_t

        # Least-squares scale/shift alignment
        s = diff.mean()
        loss = ((diff - s) ** 2).mean()
        loss_total = loss_total + loss
        n_valid += 1

    return loss_total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Marigold on UVeye Pi3 depth")
    p.add_argument("--sessions_root", type=str, default=None,
                   help="Root dir of Pi3 benchmark sessions (default: hardcoded isilon path)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save checkpoints")
    p.add_argument("--base_ckpt", type=str, default="prs-eth/marigold-depth-v1-1",
                   help="Starting Marigold checkpoint (HF hub or local path)")
    p.add_argument("--steps", type=int, default=3_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Lower LR than Model C (we're fine-tuning, not training from scratch)")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--image_size", type=str, default="480,640",
                   help="H,W to resize images to")
    p.add_argument("--min_valid_frac", type=float, default=0.02,
                   help="Min fraction of valid depth pixels to include a sample")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (for debugging)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Only train decoder (up_blocks + conv_out)")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def _setup_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    elif "ISILON_PATH" in os.environ:
        out = Path(os.environ["ISILON_PATH"]) / "results" / "model_uveye"
    else:
        out = ROOT / "results" / "model_uveye"
    out.mkdir(parents=True, exist_ok=True)
    return out


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
    sessions_root = Path(args.sessions_root) if args.sessions_root else None
    dataset = UVeyePi3Dataset(
        sessions_root=sessions_root,
        image_size=(H, W),
        min_valid_frac=args.min_valid_frac,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No samples found — check sessions_root path")
    log.info("Dataset: %d samples", len(dataset))

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
    log.info("Loading Marigold from: %s (HF_HOME=%s)", args.base_ckpt, hf_cache)
    pipe = MarigoldDepthPipeline.from_pretrained(
        args.base_ckpt,
        torch_dtype=torch.float32,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.train()
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    if args.freeze_encoder:
        frozen_prefixes = ("down_blocks", "mid_block", "conv_in")
        n_frozen = n_train = 0
        for name, param in pipe.unet.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad_(False)
                n_frozen += param.numel()
            else:
                n_train += param.numel()
        log.info("Encoder frozen: %dM frozen, %dM trainable", n_frozen // 1_000_000, n_train // 1_000_000)

    # --- Optimizer ---
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    opt = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = LambdaLR(opt, lr_lambda=lambda s: _lr_lambda(s, args.warmup_steps, args.steps))

    # --- Empty text embedding (Marigold uses empty prompt) ---
    pipe.encode_empty_text()
    empty_embed = pipe.empty_text_embed.to(device)  # [1, 77, 1024]

    # --- Output dir ---
    out_dir = _setup_output_dir(args)
    log.info("Saving checkpoints to: %s", out_dir)

    # --- Training loop ---
    global_step = 0
    running_loss = 0.0
    opt.zero_grad()

    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break

            rgb = batch["rgb"].to(device)       # [B, 3, H, W]
            depth_gt = batch["depth"].to(device) # [B, 1, H, W]
            mask = batch["mask"].to(device)      # [B, 1, H, W]

            B = rgb.shape[0]

            # Encode RGB to latent (Marigold's VAE expects image in [-1, 1])
            rgb_norm = rgb * 2.0 - 1.0
            with torch.no_grad():
                rgb_lat = pipe.vae.encode(rgb_norm).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Normalise depth GT to [-1, 1] for latent encoding
            # Use per-sample min-max over valid pixels (affine-invariant supervision)
            depth_norm = torch.zeros_like(depth_gt)
            for b in range(B):
                m = mask[b, 0]
                if m.sum() > 0:
                    d_min = depth_gt[b, 0][m].min()
                    d_max = depth_gt[b, 0][m].max()
                    rng = (d_max - d_min).clamp(min=1e-4)
                    depth_norm[b] = (depth_gt[b] - d_min) / rng * 2.0 - 1.0

            # Encode depth GT to latent
            depth_norm_3ch = depth_norm.repeat(1, 3, 1, 1)
            with torch.no_grad():
                depth_lat = pipe.vae.encode(depth_norm_3ch).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Marigold 8-channel UNet input: [rgb_lat | depth_lat] (noise-free at t=999 for regression)
            t = torch.tensor([_TRAIN_TIMESTEP] * B, device=device)
            noise = torch.randn_like(depth_lat)
            noisy_depth_lat = pipe.scheduler.add_noise(depth_lat, noise, t)
            unet_in = torch.cat([rgb_lat, noisy_depth_lat], dim=1)

            text_embed = empty_embed.expand(B, -1, -1)
            pred_noise = pipe.unet(unet_in, t, encoder_hidden_states=text_embed).sample

            # Decode predicted noise to depth latent prediction
            pred_depth_lat = pipe.scheduler.step(
                pred_noise, _TRAIN_TIMESTEP, noisy_depth_lat
            ).pred_original_sample

            # Decode to depth image
            pred_depth_img = pipe.vae.decode(
                pred_depth_lat / pipe.vae.config.scaling_factor
            ).sample  # [B, 3, H, W]
            pred_depth = pred_depth_img[:, :1]  # take first channel

            # Resize mask to latent-decoded resolution if needed
            if mask.shape[-2:] != pred_depth.shape[-2:]:
                mask_resized = F.interpolate(mask.float(), size=pred_depth.shape[-2:], mode="nearest").bool()
            else:
                mask_resized = mask

            # Unnormalise pred_depth to metric scale for SSI loss
            # (SSI loss handles scale/shift invariance, so absolute values don't matter)
            loss = scale_shift_invariant_loss(pred_depth, depth_gt, mask_resized)
            (loss / args.grad_accum).backward()
            running_loss += loss.item()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()

            global_step += 1

            if global_step % args.log_every == 0:
                avg = running_loss / args.log_every
                running_loss = 0.0
                lr = scheduler.get_last_lr()[0]
                log.info("step=%d/%d  loss=%.4f  lr=%.2e  valid_frac=%.2f",
                         global_step, args.steps, avg, lr,
                         batch["valid_frac"].mean().item())

            if global_step % args.save_every == 0 or global_step == args.steps:
                ckpt = out_dir / f"checkpoint-{global_step}"
                pipe.unet.save_pretrained(ckpt / "unet")
                log.info("Saved checkpoint: %s", ckpt)

    log.info("Training complete. Final checkpoint at step %d", global_step)


if __name__ == "__main__":
    train(parse_args())
