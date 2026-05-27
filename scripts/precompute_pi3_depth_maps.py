#!/usr/bin/env python3
"""
Precompute UVeye Pi3 depth maps from per-frame car PLY files.

Reads each (session, run, cam, frame) triple, projects the car point cloud
into the camera, and saves the resulting depth map as a float32 NPY file.
Also saves a small JSON index listing all valid (rgb_path, depth_path) pairs.

Run once before training — avoids re-reading PLY files at training time.

Usage:
    python scripts/precompute_pi3_depth_maps.py \
        --sessions_root /isilon/Automotive/RnD/elad.e/uv-3d/sessions/pi3_benchmark \
        --output_dir    /isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics/data/uveye_pi3_depth \
        --min_valid_frac 0.005 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.uveye_pi3_dataset import (  # noqa: E402
    _discover_samples,
    _parse_calibration,
    _load_ply_xyz,
    _project_to_depth,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _process_sample(args_tuple) -> dict | None:
    """Worker: project one sample and save depth NPY. Returns index entry or None."""
    s, output_dir, min_valid_frac = args_tuple
    try:
        calib_map = _parse_calibration(s["calib_path"])
        if s["cam"] not in calib_map:
            return None
        c = calib_map[s["cam"]]
        xyz = _load_ply_xyz(s["ply_path"])
        depth = _project_to_depth(xyz, s["pose_w2c"], c["K"], c["dist"], c["W"], c["H"])

        n_valid = (depth > 0).sum()
        valid_frac = n_valid / (c["W"] * c["H"])
        if valid_frac < min_valid_frac:
            return None

        # Save depth as float32 NPY
        rel = f"{s['session']}/{s['run']}/{s['cam']}/frame_{s['frame']:04d}.npy"
        out_path = Path(output_dir) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), depth)

        return {
            "session": s["session"],
            "cam": s["cam"],
            "frame": s["frame"],
            "rgb": str(s["img_path"]),
            "depth": str(out_path),
            "valid_frac": float(valid_frac),
            "W": c["W"],
            "H": c["H"],
        }
    except Exception as e:
        log.warning("Failed %s/%s/frame%d: %s", s["session"], s["cam"], s["frame"], e)
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_valid_frac", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    sessions_root = Path(args.sessions_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Discovering samples from %s ...", sessions_root)
    samples = _discover_samples(sessions_root)
    log.info("Found %d samples. Projecting with %d workers ...", len(samples), args.workers)

    work = [(s, args.output_dir, args.min_valid_frac) for s in samples]
    index: list[dict] = []
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_sample, w): i for i, w in enumerate(work)}
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                index.append(result)
            else:
                n_skipped += 1
            done = len(index) + n_skipped
            if done % 100 == 0:
                log.info("Progress: %d/%d  valid=%d  skipped=%d",
                         done, len(samples), len(index), n_skipped)

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    log.info("Done. %d valid depth maps saved to %s", len(index), output_dir)
    log.info("Index written to %s", index_path)


if __name__ == "__main__":
    main()
