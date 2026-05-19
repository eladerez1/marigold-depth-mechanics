#!/usr/bin/env python3
"""Exp 03: probing accuracy vs denoising timestep (Q2, Model B)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.extraction.timestep_sampler import parse_timesteps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="B")
    p.add_argument("--timesteps", type=str, default="1000,900,800,700,600,500,400,300,200,100,50,10,1")
    p.add_argument("--tasks", type=str, default="ordinal,depth,normals,boundary,planar")
    p.add_argument("--dataset", type=str, default="nyuv2")
    p.add_argument("--output", type=str, default="results/exp03")
    p.add_argument("--feature_cache", type=str, default="data/features")
    args = p.parse_args()

    timesteps = parse_timesteps(args.timesteps)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "timestep_curves.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "task", "metric_value"])
    print(
        f"Exp03 scaffold: extract features at {len(timesteps)} timesteps, "
        f"then train probes. Wrote empty {csv_path}"
    )


if __name__ == "__main__":
    main()
