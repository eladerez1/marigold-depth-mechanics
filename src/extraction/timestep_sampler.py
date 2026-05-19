"""Timestep schedules for multi-step feature extraction."""

from __future__ import annotations

DEFAULT_TIMESTEPS = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10, 1]


def parse_timesteps(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def timesteps_for_model(model_id: str, custom: list[int] | None = None) -> list[int]:
    if model_id in ("A", "C"):
        return [1]
    return custom or DEFAULT_TIMESTEPS
