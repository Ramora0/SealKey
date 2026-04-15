"""Mode-driven dispatch over src.hint_generators.

mode ∈ {'solo', 'double_target', 'double_all'} picks the eligible generators
and their weights. Returns a uint8 (H, W) hint mask.
"""

from __future__ import annotations

import numpy as np

from src.hint_generators import (
    box_hint,
    chroma_key_hint,
    trimap_hint,
    zero_hint,
)


HINT_WEIGHTS: dict[str, dict[str, float]] = {
    "solo":          {"trimap": 1.0, "box": 1.0, "chroma_key": 1.0, "zero": 0.3},
    "double_target": {"trimap": 1.0, "chroma_key_clean": 1.5},
    "double_all":    {"trimap": 1.0, "box": 1.0, "chroma_key": 1.0, "zero": 0.3},
}


def sample_hint(
    mode: str,
    rgb: np.ndarray,
    clean_target_rgb: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Draw a hint-generator by mode weights and invoke it.

    rgb: (H,W,3) uint8 input (may contain distractor).
    clean_target_rgb: (H,W,3) uint8 target-only composite over green (no
        distractor). Used by chroma_key_clean so the key is perfect by
        construction. For solo/double_all this can be the same as `rgb`.
    alpha: (H,W) uint8 supervising alpha.
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
    if pick == "chroma_key_clean":
        return chroma_key_hint(clean_target_rgb, rng), pick
    if pick == "zero":
        return zero_hint(alpha.shape), pick
    raise ValueError(f"Unknown hint pick: {pick}")
