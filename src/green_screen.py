"""Procedural green screen background generator with realistic imperfections.

Generates varied chroma-key backgrounds spanning clean studio screens to
beat-up, poorly-lit ones.  Each image is a numpy array (H, W, 3) in BGR
(OpenCV convention) that can be composited behind a transparent subject.

Usage:
    python -m src.green_screen --output greenscreens/ --count 200 --size 1024
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def _perlin_grid(shape: tuple[int, int], scale: float, seed: int | None = None) -> np.ndarray:
    """Fast approximate Perlin-ish noise via interpolated random grids."""
    rng = np.random.default_rng(seed)
    h, w = shape
    gh = max(2, int(h / scale))
    gw = max(2, int(w / scale))
    grid = rng.standard_normal((gh, gw)).astype(np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def _fbm(shape: tuple[int, int], octaves: int = 4, seed: int | None = None) -> np.ndarray:
    """Fractal Brownian Motion — layered noise at multiple scales."""
    rng = np.random.default_rng(seed)
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    scale = max(shape) / 3.0
    for _ in range(octaves):
        result += amplitude * _perlin_grid(shape, scale, seed=rng.integers(0, 2**31))
        amplitude *= 0.5
        scale *= 0.5
        scale = max(scale, 2.0)
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-6:
        result = (result - lo) / (hi - lo)
    return result


# ---------------------------------------------------------------------------
# Base green generation
# ---------------------------------------------------------------------------

def _random_base_green(rng: np.random.Generator) -> tuple[int, int, int]:
    """Return a random (H, S, V) base green in OpenCV ranges (H:0-179, S/V:0-255)."""
    h = rng.integers(35, 80)          # hue range covering greens
    s = rng.integers(150, 255)        # fairly saturated
    v = rng.integers(130, 240)        # medium to bright
    return int(h), int(s), int(v)


def _make_base(h: int, w: int, base_hsv: tuple[int, int, int]) -> np.ndarray:
    """Solid HSV base, returned as float32 HSV image (H:0-179, S:0-255, V:0-255)."""
    img = np.full((h, w, 3), base_hsv, dtype=np.float32)
    return img


# ---------------------------------------------------------------------------
# Individual defect layers
# ---------------------------------------------------------------------------

def _apply_color_inconsistency(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow hue/saturation drift across the screen — aged fabric, mixed batches."""
    h, w = img.shape[:2]
    hue_drift = _fbm((h, w), octaves=2, seed=rng.integers(0, 2**31))
    sat_drift = _fbm((h, w), octaves=2, seed=rng.integers(0, 2**31))

    strength_h = rng.uniform(2, 10)
    strength_s = rng.uniform(5, 25)

    img[:, :, 0] += (hue_drift - 0.5) * strength_h
    img[:, :, 1] += (sat_drift - 0.5) * strength_s
    return img


def _apply_uneven_lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hotspots, falloff, and gradients on the value channel."""
    h, w = img.shape[:2]

    # Global gradient direction
    angle = rng.uniform(0, 2 * math.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys = ys / h - 0.5
    xs = xs / w - 0.5
    gradient = xs * math.cos(angle) + ys * math.sin(angle)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    grad_strength = rng.uniform(15, 60)
    img[:, :, 2] += (gradient - 0.5) * grad_strength

    # 1-3 hotspots
    n_spots = rng.integers(1, 4)
    for _ in range(n_spots):
        cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
        radius = rng.uniform(0.15, 0.4) * max(h, w)
        dist = np.sqrt((ys * h - cy + h / 2) ** 2 + (xs * w - cx + w / 2) ** 2)
        spot = np.clip(1.0 - dist / radius, 0, 1) ** 2
        img[:, :, 2] += spot * rng.uniform(15, 50)

    # Edge falloff (vignette)
    if rng.random() < 0.6:
        dist_center = np.sqrt(ys ** 2 + xs ** 2)
        vignette = np.clip(dist_center / 0.7, 0, 1) ** 2
        img[:, :, 2] -= vignette * rng.uniform(10, 40)

    return img


def _apply_wrinkles(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fabric folds — dark valleys, bright ridges via directional noise."""
    h, w = img.shape[:2]
    # Stretch noise in one direction for fold-like appearance
    angle = rng.uniform(0, math.pi)
    scale_x = rng.uniform(20, 80)
    scale_y = rng.uniform(100, 400)

    noise = _perlin_grid((h, w), scale=scale_x, seed=rng.integers(0, 2**31))
    # Add a second layer at different angle
    noise2 = _perlin_grid((h, w), scale=scale_y, seed=rng.integers(0, 2**31))
    wrinkle = 0.7 * noise + 0.3 * noise2

    lo, hi = wrinkle.min(), wrinkle.max()
    if hi - lo > 1e-6:
        wrinkle = (wrinkle - lo) / (hi - lo)

    strength = rng.uniform(10, 45)
    img[:, :, 2] += (wrinkle - 0.5) * strength
    # Slight saturation change in fold valleys
    img[:, :, 1] += (wrinkle - 0.5) * strength * 0.3
    return img


def _apply_subject_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dark shadow blob roughly where a subject would stand."""
    h, w = img.shape[:2]
    # Shadow center biased toward bottom-center
    cy = rng.uniform(0.4, 0.85) * h
    cx = rng.uniform(0.3, 0.7) * w
    # Elliptical shadow
    ry = rng.uniform(0.15, 0.4) * h
    rx = rng.uniform(0.1, 0.3) * w

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
    shadow = np.clip(1.0 - dist, 0, 1) ** 1.5

    darkness = rng.uniform(25, 70)
    img[:, :, 2] -= shadow * darkness
    # Shadows also slightly desaturate
    img[:, :, 1] -= shadow * darkness * 0.2
    return img


def _apply_seams(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Visible panel joins — horizontal or vertical lines of different brightness/hue."""
    h, w = img.shape[:2]
    n_seams = rng.integers(1, 4)
    for _ in range(n_seams):
        horizontal = rng.random() < 0.5
        pos = rng.uniform(0.15, 0.85)
        thickness = rng.integers(2, 8)
        hue_shift = rng.uniform(-4, 4)
        val_shift = rng.uniform(-20, 10)

        if horizontal:
            y = int(pos * h)
            y0 = max(0, y - thickness)
            y1 = min(h, y + thickness)
            img[y0:y1, :, 0] += hue_shift
            img[y0:y1, :, 2] += val_shift
        else:
            x = int(pos * w)
            x0 = max(0, x - thickness)
            x1 = min(w, x + thickness)
            img[:, x0:x1, 0] += hue_shift
            img[:, x0:x1, 2] += val_shift
    return img


def _apply_curvature_band(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shadow/brightness band where floor cove meets wall — typically lower third."""
    h, w = img.shape[:2]
    band_y = rng.uniform(0.55, 0.8) * h
    band_width = rng.uniform(0.02, 0.08) * h

    ys = np.arange(h, dtype=np.float32)
    profile = np.exp(-0.5 * ((ys - band_y) / band_width) ** 2)
    profile = profile[:, np.newaxis]

    darkness = rng.uniform(15, 45)
    img[:, :, 2] -= profile * darkness
    return img


def _apply_overexposure(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blown-out regions — value clips high, saturation drops."""
    h, w = img.shape[:2]
    n_patches = rng.integers(1, 3)
    for _ in range(n_patches):
        cy = rng.uniform(0.1, 0.6) * h
        cx = rng.uniform(0.2, 0.8) * w
        ry = rng.uniform(0.1, 0.3) * h
        rx = rng.uniform(0.1, 0.3) * w

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
        mask = np.clip(1.0 - dist, 0, 1) ** 2

        blow_strength = rng.uniform(40, 80)
        img[:, :, 2] += mask * blow_strength
        # Overexposure washes out saturation
        img[:, :, 1] -= mask * blow_strength * rng.uniform(0.6, 1.2)
    return img


def _apply_dirt_scuffs(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small non-green patches — footprints, tape residue, dust, marks."""
    h, w = img.shape[:2]
    n_marks = rng.integers(3, 15)
    for _ in range(n_marks):
        cy = int(rng.uniform(0.05, 0.95) * h)
        cx = int(rng.uniform(0.05, 0.95) * w)
        size = int(rng.uniform(0.005, 0.03) * max(h, w))
        if size < 2:
            size = 2

        # Random blob via small noise patch
        blob = rng.random((size * 2, size * 2)).astype(np.float32)
        blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=size * 0.4)
        blob = (blob > blob.mean()).astype(np.float32)

        y0 = max(0, cy - size)
        x0 = max(0, cx - size)
        y1 = min(h, cy + size)
        x1 = min(w, cx + size)
        # Crop blob to fit
        by0 = size - (cy - y0)
        bx0 = size - (cx - x0)
        by1 = by0 + (y1 - y0)
        bx1 = bx0 + (x1 - x0)

        if by1 <= by0 or bx1 <= bx0:
            continue
        region = blob[by0:by1, bx0:bx1]

        # Darken, desaturate, slight hue shift
        img[y0:y1, x0:x1, 2] -= region * rng.uniform(15, 50)
        img[y0:y1, x0:x1, 1] -= region * rng.uniform(20, 60)
        img[y0:y1, x0:x1, 0] += region * rng.uniform(-5, 5)
    return img


def _apply_screen_edge(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hard termination — screen doesn't fill frame. Fill edge region with
    a neutral gray/brown 'studio wall' color."""
    h, w = img.shape[:2]
    # Pick which edges are exposed
    edges = []
    if rng.random() < 0.5:
        edges.append("bottom")
    if rng.random() < 0.3:
        edges.append("top")
    if rng.random() < 0.3:
        edges.append("left")
    if rng.random() < 0.3:
        edges.append("right")

    if not edges:
        edges.append("bottom")

    # Neutral wall/floor color in HSV
    wall_h = rng.integers(10, 25)   # brownish/grayish
    wall_s = rng.integers(10, 60)
    wall_v = rng.integers(60, 140)

    for edge in edges:
        depth = rng.uniform(0.03, 0.15)
        # Slightly jagged edge
        n_points = rng.integers(4, 10)
        if edge in ("bottom", "top"):
            xs_pts = np.linspace(0, w, n_points)
            if edge == "bottom":
                base = h * (1 - depth)
                ys_pts = base + rng.uniform(-h * 0.01, h * 0.01, size=n_points)
                # Interpolate to full width
                edge_line = np.interp(np.arange(w), xs_pts, ys_pts).astype(int)
                for x in range(w):
                    y_start = max(0, min(h - 1, edge_line[x]))
                    img[y_start:, x, 0] = wall_h
                    img[y_start:, x, 1] = wall_s
                    img[y_start:, x, 2] = wall_v
            else:
                base = h * depth
                ys_pts = base + rng.uniform(-h * 0.01, h * 0.01, size=n_points)
                edge_line = np.interp(np.arange(w), xs_pts, ys_pts).astype(int)
                for x in range(w):
                    y_end = max(0, min(h, edge_line[x]))
                    img[:y_end, x, 0] = wall_h
                    img[:y_end, x, 1] = wall_s
                    img[:y_end, x, 2] = wall_v
        else:
            ys_pts_arr = np.linspace(0, h, n_points)
            if edge == "right":
                base = w * (1 - depth)
                xs_pts_arr = base + rng.uniform(-w * 0.01, w * 0.01, size=n_points)
                edge_line = np.interp(np.arange(h), ys_pts_arr, xs_pts_arr).astype(int)
                for y in range(h):
                    x_start = max(0, min(w - 1, edge_line[y]))
                    img[y, x_start:, 0] = wall_h
                    img[y, x_start:, 1] = wall_s
                    img[y, x_start:, 2] = wall_v
            else:
                base = w * depth
                xs_pts_arr = base + rng.uniform(-w * 0.01, w * 0.01, size=n_points)
                edge_line = np.interp(np.arange(h), ys_pts_arr, xs_pts_arr).astype(int)
                for y in range(h):
                    x_end = max(0, min(w, edge_line[y]))
                    img[y, :x_end, 0] = wall_h
                    img[y, :x_end, 1] = wall_s
                    img[y, :x_end, 2] = wall_v
    return img


def _apply_fabric_texture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-frequency weave texture — muslin grain."""
    h, w = img.shape[:2]
    grain = rng.standard_normal((h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), sigmaX=0.8)
    strength = rng.uniform(2, 8)
    img[:, :, 2] += grain * strength
    return img


# ---------------------------------------------------------------------------
# Defect profiles
# ---------------------------------------------------------------------------

# Each profile is (defect_name, probability_of_applying)
CLEAN_PROFILE: list[tuple[str, float]] = [
    ("color_inconsistency", 0.3),
    ("uneven_lighting", 0.4),
    ("fabric_texture", 0.5),
    ("wrinkles", 0.2),
]

MODERATE_PROFILE: list[tuple[str, float]] = [
    ("color_inconsistency", 0.7),
    ("uneven_lighting", 0.8),
    ("fabric_texture", 0.7),
    ("wrinkles", 0.6),
    ("subject_shadow", 0.4),
    ("seams", 0.3),
    ("dirt_scuffs", 0.3),
    ("curvature_band", 0.3),
]

MESSY_PROFILE: list[tuple[str, float]] = [
    ("color_inconsistency", 0.9),
    ("uneven_lighting", 0.95),
    ("fabric_texture", 0.8),
    ("wrinkles", 0.9),
    ("subject_shadow", 0.7),
    ("seams", 0.6),
    ("curvature_band", 0.5),
    ("overexposure", 0.4),
    ("dirt_scuffs", 0.7),
    ("screen_edge", 0.5),
]

_DEFECT_FNS = {
    "color_inconsistency": _apply_color_inconsistency,
    "uneven_lighting": _apply_uneven_lighting,
    "wrinkles": _apply_wrinkles,
    "subject_shadow": _apply_subject_shadow,
    "seams": _apply_seams,
    "curvature_band": _apply_curvature_band,
    "overexposure": _apply_overexposure,
    "dirt_scuffs": _apply_dirt_scuffs,
    "screen_edge": _apply_screen_edge,
    "fabric_texture": _apply_fabric_texture,
}


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_green_screen(
    height: int = 1024,
    width: int = 1024,
    seed: int | None = None,
    profile: str | None = None,
) -> np.ndarray:
    """Generate a single green screen background image.

    Args:
        height: Image height.
        width: Image width.
        seed: Random seed for reproducibility.
        profile: One of "clean", "moderate", "messy", or None (random).

    Returns:
        BGR uint8 numpy array (H, W, 3).
    """
    rng = np.random.default_rng(seed)

    if profile is None:
        profile = rng.choice(["clean", "moderate", "messy"], p=[0.25, 0.45, 0.30])

    profiles = {
        "clean": CLEAN_PROFILE,
        "moderate": MODERATE_PROFILE,
        "messy": MESSY_PROFILE,
    }
    defect_list = profiles[profile]

    base_hsv = _random_base_green(rng)
    img = _make_base(height, width, base_hsv)

    # Apply defects based on profile probabilities
    for defect_name, prob in defect_list:
        if rng.random() < prob:
            img = _DEFECT_FNS[defect_name](img, rng)

    # Clamp and convert to BGR uint8
    img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)
    img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)
    bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural green screen backgrounds"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("greenscreens"), help="Output directory"
    )
    parser.add_argument(
        "--count", type=int, default=50, help="Number of images to generate"
    )
    parser.add_argument("--size", type=int, default=1024, help="Image size (square)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument(
        "--profile",
        choices=["clean", "moderate", "messy"],
        default=None,
        help="Defect profile (default: random mix)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31)

    for i in range(args.count):
        img = generate_green_screen(
            height=args.size,
            width=args.size,
            seed=base_seed + i,
            profile=args.profile,
        )
        path = args.output / f"gs_{i:04d}.png"
        cv2.imwrite(str(path), img)
        profile_label = args.profile or "mixed"
        print(f"[{i + 1}/{args.count}] Saved {path.name} ({profile_label})")

    print(f"\nDone — {args.count} green screens saved to {args.output}/")


if __name__ == "__main__":
    main()
