#!/usr/bin/env python3
"""
Train Model C: single-step depth regression on SD2 U-Net (Marigold-style 8-ch input).

Uses Hypersim + Virtual KITTI when MARIGOLD_BASE_DATA_DIR is set; otherwise NYU proxy
(--train_data nyu) for interim runs on existing isilon pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# Fixed timestep for single-step regression training.
# Using t=999 tells the UNet "full noise → clean depth" in one shot.
_TRAIN_TIMESTEP = 999

ROOT = Path(__file__).resolve().parents[2]
MARIGOLD_ROOT = ROOT / "third_party" / "Marigold"


def _load_module_from_path(module_name: str, path: Path):
    """Load a project file without importing project `src` (shadowed by Marigold)."""
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

_nyu = _load_module_from_path(
    "nyu_marigold_train", ROOT / "src" / "data" / "nyu_marigold_train.py"
)

# Marigold's `src` must win over the project package of the same name.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MARIGOLD_ROOT))

from marigold import MarigoldDepthPipeline  # noqa: E402
from src.dataset import DatasetMode, get_dataset  # noqa: E402
from src.dataset.mixed_sampler import MixedBatchSampler  # noqa: E402
from src.util.depth_transform import get_depth_normalizer  # noqa: E402
from src.util.loss import SILogMSELoss  # noqa: E402

NYUMarigoldTrainDataset = _nyu.NYUMarigoldTrainDataset
collect_nyu_train_pairs = _nyu.collect_nyu_train_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model C (single-step regression)")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: results/model_C in ACR (isilon ro), else checkpoints/model_C",
    )
    p.add_argument(
        "--train_data",
        type=str,
        default="hypersim,vkitti",
        help="hypersim,vkitti | nyu (proxy on isilon NYU pairs)",
    )
    p.add_argument("--data_root", type=str, default=None, help="MARIGOLD_BASE_DATA_DIR")
    p.add_argument("--base_ckpt", type=str, default=None, help="SD2 / model_A root")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_nyu_images", type=int, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="marigold-internals")
    p.add_argument("--wandb_run", type=str, default="model-C-single-step")
    return p.parse_args()


def resolve_data_root(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for key in ("MARIGOLD_BASE_DATA_DIR", "BASE_DATA_DIR"):
        if os.environ.get(key):
            p = Path(os.environ[key])
            if p.exists():
                return p
    for candidate in (
        ROOT / "data" / "marigold",
        Path("/isilon/Automotive/RnD/shared/marigold_data"),
    ):
        if candidate.exists():
            return candidate
    return None


def _pipeline_skeleton_usable(path: Path) -> bool:
    """model_A_sd2 is UNet-only; need tokenizer + scheduler for from_pretrained."""
    return (
        (path / "tokenizer" / "tokenizer_config.json").exists()
        and (path / "scheduler" / "scheduler_config.json").exists()
        and (path / "vae" / "config.json").exists()
    )


def adapt_unet_8ch(pipe: MarigoldDepthPipeline) -> None:
    if pipe.unet.config.in_channels == 8:
        return
    weight = pipe.unet.conv_in.weight.clone()
    bias = pipe.unet.conv_in.bias.clone()
    weight = weight.repeat((1, 2, 1, 1)) * 0.5
    new_in = Conv2d(
        8,
        pipe.unet.conv_in.out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    )
    new_in.weight = Parameter(weight)
    new_in.bias = Parameter(bias)
    pipe.unet.conv_in = new_in
    pipe.unet.config["in_channels"] = 8


def build_pipeline(device: torch.device) -> MarigoldDepthPipeline:
    from diffusers import UNet2DConditionModel

    loader_mod = _load_module_from_path(
        "marigold_pipe_loader", ROOT / "src" / "models" / "marigold_pipe_loader.py"
    )
    ckpt_b = ROOT / "checkpoints" / "model_B_marigold"
    if not _pipeline_skeleton_usable(ckpt_b):
        logging.warning(
            "Local Marigold checkpoint incomplete at %s; using HF hub loader", ckpt_b
        )
    pipe = loader_mod.load_marigold_depth_pipeline(str(device), ckpt_b)

    unet_a = ROOT / "checkpoints" / "model_A_sd2" / "unet"
    if (unet_a / "config.json").exists():
        logging.info("Replacing UNet with SD2 vanilla from %s", unet_a.parent)
        pipe.unet = UNet2DConditionModel.from_pretrained(
            unet_a.parent, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
    adapt_unet_8ch(pipe)
    pipe.encode_empty_text()
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.to(device)
    return pipe


def build_train_loader(
    train_data: str,
    data_root: Path | None,
    cfg,
    batch_size: int,
    seed: int,
    max_nyu_images: int | None,
) -> DataLoader:
    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    gen = torch.Generator().manual_seed(seed)

    if train_data.strip().lower() == "nyu":
        pairs = collect_nyu_train_pairs(max_nyu_images)
        if not pairs:
            raise SystemExit(
                "No NYU train pairs found under sparse_confidence/nyu_raw. "
                "Use --train_data hypersim,vkitti with MARIGOLD_BASE_DATA_DIR."
            )
        ds = NYUMarigoldTrainDataset(
            pairs,
            depth_transform,
            resize_to_hw=(480, 640),
            lr_flip_p=cfg.augmentation.lr_flip_p,
        )
        logging.info("NYU proxy train set: %d pairs", len(ds))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            generator=gen,
            pin_memory=True,
        )

    if data_root is None:
        raise SystemExit(
            "Hypersim/VKitti training requires MARIGOLD_BASE_DATA_DIR or --data_root.\n"
            "Preprocess per third_party/Marigold/script/depth/dataset_preprocess/hypersim/README.md\n"
            "Or use --train_data nyu for an interim run on existing NYU pairs."
        )

    cfg_ds = OmegaConf.load(ROOT / "config" / "train_model_c.yaml")
    cfg_data = cfg_ds.dataset
    for i, ds_cfg in enumerate(cfg_data.train.dataset_list):
        rel = ds_cfg.filenames
        cfg_data.train.dataset_list[i].filenames = str(MARIGOLD_ROOT / rel)

    train_dataset = get_dataset(
        cfg_data.train,
        base_data_dir=str(data_root),
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    dataset_ls = train_dataset
    concat = ConcatDataset(dataset_ls)
    sampler = MixedBatchSampler(
        src_dataset_ls=dataset_ls,
        batch_size=batch_size,
        drop_last=True,
        prob=cfg_data.train.prob_ls,
        shuffle=True,
        generator=gen,
    )
    return DataLoader(concat, batch_sampler=sampler, num_workers=4, pin_memory=True)


def save_checkpoint(pipe: MarigoldDepthPipeline, out_dir: Path, step: int) -> None:
    ckpt = out_dir / f"checkpoint-{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_pretrained(ckpt / "unet")
    pipe.vae.save_pretrained(ckpt / "vae")
    pipe.scheduler.save_pretrained(ckpt / "scheduler")
    # Latest layout for load_models.py (copy — symlinks break on some ACR mounts)
    import shutil

    for sub in ("unet", "vae", "scheduler"):
        dst = out_dir / sub
        src = ckpt / sub
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        shutil.copytree(src, dst)


def warmup_lambda(step: int, warmup: int) -> float:
    if step < warmup:
        return float(step) / float(max(1, warmup))
    return 1.0


def train_step(
    pipe: MarigoldDepthPipeline,
    batch: dict,
    device: torch.device,
    empty_text: torch.Tensor,
) -> torch.Tensor:
    """Single-step regression: predict depth latent directly from rgb latent.

    We skip the DDIM scheduler.step entirely. At t=_TRAIN_TIMESTEP the UNet
    receives [rgb_latent | noise] and is supervised to output the GT depth
    latent with MSE loss. This avoids the 0/0 in the zero-SNR DDIM formula
    (which caused NaN) and the expensive per-step VAE decode (which caused
    ~12 s/step on V100).
    """
    rgb = batch["rgb_norm"].to(device)
    depth_norm = batch["depth_raw_norm"].to(device)

    with torch.no_grad():
        rgb_latent = pipe.encode_rgb(rgb)
        gt_latent = pipe.encode_rgb(depth_norm.repeat(1, 3, 1, 1))

    B = rgb_latent.shape[0]
    noise_latent = torch.randn_like(gt_latent)
    t = torch.full((B,), _TRAIN_TIMESTEP, device=device, dtype=torch.long)
    text_embed = empty_text.repeat(B, 1, 1)

    unet_in = torch.cat([rgb_latent, noise_latent], dim=1)
    pred_latent = pipe.unet(unet_in, t, encoder_hidden_states=text_embed).sample
    return torch.nn.functional.mse_loss(pred_latent, gt_latent.to(pred_latent.dtype))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit(
            "Model C training requires CUDA. Submit via ./scripts/acr_submit.sh train_model_c"
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir) if args.output_dir else model_c_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Model C output dir: %s", out_dir)

    cfg = OmegaConf.load(ROOT / "config" / "train_model_c.yaml")
    data_root = resolve_data_root(args.data_root)
    train_modes = [s.strip().lower() for s in args.train_data.split(",")]
    if "nyu" in train_modes or train_modes == ["nyu"]:
        train_key = "nyu"
    else:
        train_key = "hypersim,vkitti"

    pipe = build_pipeline(device)
    loader = build_train_loader(
        train_key,
        data_root,
        cfg,
        args.batch_size,
        args.seed,
        args.max_nyu_images,
    )

    scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    pipe.scheduler = scheduler

    empty_text = pipe.empty_text_embed.detach().to(device)

    optimizer = AdamW(pipe.unet.parameters(), lr=args.lr)
    lr_sched = LambdaLR(
        optimizer, lr_lambda=lambda s: warmup_lambda(s, args.warmup_steps)
    )

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=vars(args),
            )
        except Exception as e:
            logging.warning("wandb disabled: %s", e)

    status_path = ROOT / "results" / "train_model_c_status.json"
    effective = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=args.steps, desc="Model C")
    while effective < args.steps:
        for batch in loader:
            # Use fp16 (not bf16): V100 has native fp16 TensorCores but no bf16.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = train_step(pipe, batch, device, empty_text)
            if not torch.isfinite(loss):
                logging.warning("step %d: non-finite loss %.5f — skipping", effective, loss.item())
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue
            (loss / args.grad_accum).backward()
            accum += 1
            if accum < args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            effective += 1
            pbar.update(1)

            if effective % args.log_every == 0:
                logging.info("step %d loss=%.5f lr=%.2e", effective, loss.item(), lr_sched.get_last_lr()[0])
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    json.dumps(
                        {
                            "stage": "training_model_c",
                            "step": effective,
                            "loss": loss.item(),
                            "train_data": train_key,
                        },
                        indent=2,
                    )
                )
                if wandb_run:
                    wandb_run.log({"loss": loss.item(), "step": effective})

            if effective % args.save_every == 0:
                save_checkpoint(pipe, out_dir, effective)

            if effective >= args.steps:
                break

    pbar.close()
    save_checkpoint(pipe, out_dir, effective)
    (out_dir / "training_done.json").write_text(
        json.dumps({"steps": effective, "train_data": train_key}, indent=2)
    )
    logging.info("Model C training finished → %s", out_dir)


if __name__ == "__main__":
    main()
