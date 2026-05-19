"""SD2 vanilla U-Net forward for probing (4-ch depth latent, Marigold VAE for encoding)."""

from __future__ import annotations

import torch
from tqdm import tqdm

from src.extraction.feature_extractor import FeatureExtractor
from src.probing.spatial_labels import make_spatial_probe_labels


def encode_depth_latent(pipe, depth_64: torch.Tensor) -> torch.Tensor:
    """Map [H,W] depth to VAE latent via 3-channel pseudo-RGB."""
    d = depth_64.unsqueeze(0).unsqueeze(0).float()
    d3 = d.repeat(1, 3, 1, 1)
    mx = d3.max().clamp(min=1e-6)
    d3 = (d3 / mx) * 2.0 - 1.0
    d3 = d3.to(device=pipe.device, dtype=pipe.dtype)
    return pipe.encode_rgb(d3)


@torch.no_grad()
def extract_sd2_one_layer(
    sd2_unet,
    pipe,
    pairs,
    load_rgb_tensor,
    load_depth_tensor,
    layer_name: str,
    n_steps: int,
    device: str,
    max_images: int,
) -> dict:
    """Return {layer_name: {timestep: list[[C,H,W]]}}."""
    from collections import defaultdict

    feats: dict = defaultdict(list)
    extractor = FeatureExtractor(sd2_unet, layers={layer_name})
    extractor.register_hooks()
    pipe.encode_empty_text()
    embed = pipe.empty_text_embed
    rng = torch.Generator(device=device)

    for rgb_path, depth_path in tqdm(
        pairs[:max_images], desc=f"extract-A-{layer_name.split('.')[-1]}", leave=False
    ):
        _ = load_rgb_tensor(rgb_path)
        depth_gt = load_depth_tensor(depth_path)
        depth_latent = encode_depth_latent(pipe, depth_gt)
        target = torch.randn(
            depth_latent.shape, device=device, dtype=pipe.dtype, generator=rng
        )
        batch_embed = embed.repeat(target.shape[0], 1, 1).to(device)
        pipe.scheduler.set_timesteps(n_steps, device=device)

        for t in pipe.scheduler.timesteps:
            extractor.clear_cache()
            noise_pred = sd2_unet(target, t, encoder_hidden_states=batch_embed).sample
            tensor = extractor.get_latest().get(layer_name)
            if tensor is not None:
                feats[int(t.item())].append(tensor.squeeze(0).cpu())
            target = pipe.scheduler.step(
                noise_pred, t, target, generator=rng
            ).prev_sample

    extractor.remove_hooks()
    return {layer_name: feats}


@torch.no_grad()
def extract_sd2_features(
    sd2_unet,
    pipe,
    pairs,
    load_rgb_tensor,
    load_depth_tensor,
    n_steps: int,
    device: str,
    max_images: int,
) -> tuple[dict, dict]:
    """Legacy full extract — prefer layer streaming from run_gpu_pipeline."""
    from collections import defaultdict

    from src.extraction.feature_extractor import list_resnet_hook_layers, subsample_layers

    labels_acc: dict = defaultdict(list)
    for _, depth_path in pairs[:max_images]:
        lab = make_spatial_probe_labels(load_depth_tensor(depth_path))
        for task in ("ordinal", "depth", "boundary"):
            labels_acc[task].append(lab[task])

    all_feats: dict = {}
    for layer in subsample_layers(list_resnet_hook_layers(sd2_unet)):
        all_feats.update(
            extract_sd2_one_layer(
                sd2_unet,
                pipe,
                pairs,
                load_rgb_tensor,
                load_depth_tensor,
                layer,
                n_steps,
                device,
                max_images,
            )
        )
    return all_feats, labels_acc
