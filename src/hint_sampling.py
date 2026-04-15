"""Mode-driven dispatch over src.hint_generators.

mode ∈ {'solo', 'double_target', 'double_all'} picks the eligible generators
and their weights. Returns a uint8 (H, W) hint mask.
"""

from __future__ import annotations

import numpy as np

from src.hint_generators import (
    box_hint,
    chroma_key_gated_hint,
    chroma_key_hint,
    trimap_hint,
    zero_hint,
)


HINT_WEIGHTS: dict[str, dict[str, float]] = {
    "solo":          {"trimap": 1.0, "box": 1.0, "chroma_key": 1.0, "zero": 0.3},
    "double_target": {"trimap": 1.0, "box": 1.0, "chroma_key_gated": 1.5},
    "double_all":    {"trimap": 1.0, "box": 1.0, "chroma_key": 1.0, "zero": 0.3},
}


def sample_hint(
    mode: str,
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_alpha: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Draw a hint-generator by mode weights and invoke it.

    rgb: (H,W,3) uint8. alpha: (H,W) uint8 — the supervising alpha.
    target_alpha: (H,W) uint8 target alpha, required for chroma_key_gated.
    Returns (hint_uint8, picked_generator_name).
    """
    weights = HINT_WEIGHTS[mode]
    names = list(weights.keys())
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()
    pick = str(rng.choice(names, p=probs))

    if pick == "trimap":
        return trimap_hint(alpha, rng), pick
    if pick == "box":
        return box_hint(alpha, rng), pick
    if pick == "chroma_key":
        return chroma_key_hint(rgb, rng), pick
    if pick == "chroma_key_gated":
        if target_alpha is None:
            raise ValueError("chroma_key_gated requires target_alpha")
        return chroma_key_gated_hint(rgb, target_alpha, rng), pick
    if pick == "zero":
        return zero_hint(alpha.shape), pick
    raise ValueError(f"Unknown hint pick: {pick}")
