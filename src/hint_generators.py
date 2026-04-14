"""Runtime hint generators for the SealKey `hint` input channel.

Each function returns a uint8 single-channel mask (H, W) suitable for
feeding to the model's 4th input channel. Called by the dataloader per
sample — no I/O, no caching, no stored artifacts.

Sources:
- trimap_hint:     erode/dilate GT alpha into a 3-level (0, 128, 255) mask.
- box_hint:        filled rectangle over the subject bbox with random looseness.
- chroma_key_hint: HSV threshold keying out green/blue screen colors.
- zero_hint:       all zeros — teaches the "no-hint" fallback regime.

All RNG-consuming functions are deterministic given `rng`.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Trimap
# ---------------------------------------------------------------------------

def trimap_hint(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erode + dilate GT alpha into a 3-level trimap.

    Output: 0 where definitely background, 255 where definitely foreground,
    128 in the uncertainty band between eroded and dilated masks.
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D, got shape {alpha.shape}")

    binary = (alpha >= 128).astype(np.uint8) * 255
    erode_k  = int(rng.integers(5, 25))
    dilate_k = int(rng.integers(5, 25))

    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    fg = cv2.erode(binary, ek)
    bg_not = cv2.dilate(binary, dk)

    out = np.full_like(binary, 0)
    out[bg_not > 0] = 128
    out[fg > 0] = 255
    return out


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def box_hint(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Filled rectangle over the subject bbox with random per-side looseness.

    Looseness is 0–15% of the box's own dimension per side. If the alpha is
    empty, returns all zeros.
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D, got shape {alpha.shape}")
    h, w = alpha.shape
    out = np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(alpha >= 128)
    if ys.size == 0:
        return out

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh = max(1, y1 - y0)
    bw = max(1, x1 - x0)

    y0 = max(0, y0 - int(rng.uniform(0, 0.15) * bh))
    y1 = min(h - 1, y1 + int(rng.uniform(0, 0.15) * bh))
    x0 = max(0, x0 - int(rng.uniform(0, 0.15) * bw))
    x1 = min(w - 1, x1 + int(rng.uniform(0, 0.15) * bw))

    out[y0:y1 + 1, x0:x1 + 1] = 255
    return out


# ---------------------------------------------------------------------------
# Chroma key
# ---------------------------------------------------------------------------

# Matches the base-green/base-blue ranges in green_screen._random_base_green.
# Wider than the generator's ranges so the keyer catches spill-tinted edges.
_GREEN_H = (35, 85)
_BLUE_H = (95, 130)


def chroma_key_hint(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """HSV threshold keying out green OR blue screen backgrounds.

    Returns 255 where the pixel looks like subject, 0 where it looks like
    key color. Saturation/value thresholds are randomized per call so the
    dataloader sees a distribution of keyer "tightness" — this intentionally
    includes configs that kill blonde hair (see data/problems.md), which is
    part of what Step-2 distractor training teaches the model to overcome.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape {rgb.shape}")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    s_min = int(rng.integers(60, 140))
    v_min = int(rng.integers(40, 100))

    green_mask = (h >= _GREEN_H[0]) & (h <= _GREEN_H[1]) & (s >= s_min) & (v >= v_min)
    blue_mask  = (h >= _BLUE_H[0])  & (h <= _BLUE_H[1])  & (s >= s_min) & (v >= v_min)
    key_mask = green_mask | blue_mask

    out = np.where(key_mask, 0, 255).astype(np.uint8)

    # Clean up speckle — close small holes in subject, open stray key-color pixels.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


# ---------------------------------------------------------------------------
# Gated chroma key — for multi-subject frames
# ---------------------------------------------------------------------------

def chroma_key_gated_hint(
    rgb: np.ndarray,
    target_alpha: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Chroma-key constrained to the target's bounding box.

    Simulates the real user workflow on multi-subject green-screen footage:
    draw a rough garbage matte around the target (excluding distractors),
    then apply a chroma keyer inside it. Returns 255 where the keyer says
    "subject" *and* we're inside the target's bbox with looseness.
    """
    gate = box_hint(target_alpha, rng)
    key = chroma_key_hint(rgb, rng)
    return np.minimum(gate, key)


# ---------------------------------------------------------------------------
# Zero
# ---------------------------------------------------------------------------

def zero_hint(shape: tuple[int, int]) -> np.ndarray:
    """All-zeros hint. Semantically: "key everything on the green screen."

    In single-subject frames, equivalent to a normal matte task. In Step-2
    multi-subject frames, the training target becomes the union of all
    subject alphas.
    """
    return np.zeros(shape, dtype=np.uint8)
