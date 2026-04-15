"""PyAV wrappers for SealKey video decoding.

Preprocess outputs per sample:
- `input.<ext>` (mp4/webm/mpg) — degraded 3-channel RGB.
- `<label>_rgb.mp4`   — clean GT RGB (h264 yuv444p CRF 12, visually lossless).
- `<label>_alpha.mkv` — clean GT alpha (FFV1 gray, lossless).

Labels: "gt" for solos; "gt_target" and "gt_all" for doubles.
All videos decoded start-to-finish into numpy arrays. No mid-video seeking.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def decode_rgb_video(path: Path, max_frames: int | None = None) -> np.ndarray:
    """Decode an arbitrary lossy RGB video to (T, H, W, 3) uint8 RGB."""
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            out.append(arr)
            if max_frames is not None and len(out) >= max_frames:
                break
    if not out:
        raise RuntimeError(f"Decoded zero frames from {path}")
    return np.stack(out, axis=0)


def decode_alpha_ffv1(path: Path, max_frames: int | None = None) -> np.ndarray:
    """Decode an FFV1 gray alpha video to (T, H, W) uint8."""
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            f = frame.reformat(format="gray")
            arr = f.to_ndarray()
            if arr.ndim == 3:
                arr = arr[..., 0]
            out.append(arr)
            if max_frames is not None and len(out) >= max_frames:
                break
    if not out:
        raise RuntimeError(f"Decoded zero frames from {path}")
    return np.stack(out, axis=0)


def decode_gt_pair(sample_dir: Path, label: str,
                   max_frames: int | None = None) -> np.ndarray:
    """Decode `{label}_rgb.mp4` + `{label}_alpha.mkv` and stack to (T,H,W,4) uint8.

    Trims to the shorter of the two streams (encode boundaries can drop a frame).
    """
    rgb = decode_rgb_video(sample_dir / f"{label}_rgb.mp4", max_frames=max_frames)
    alpha = decode_alpha_ffv1(sample_dir / f"{label}_alpha.mkv", max_frames=max_frames)
    n = min(rgb.shape[0], alpha.shape[0])
    if rgb.shape[1:3] != alpha.shape[1:3]:
        raise RuntimeError(
            f"{label}: rgb {rgb.shape[1:3]} vs alpha {alpha.shape[1:3]} under {sample_dir}"
        )
    return np.concatenate([rgb[:n], alpha[:n, ..., None]], axis=-1)


def find_input_file(sample_dir: Path) -> Path:
    """Locate `input.<ext>` within a sample directory."""
    for p in sample_dir.iterdir():
        if p.is_file() and p.stem == "input" and p.suffix.lower() in {".mp4", ".webm", ".mpg", ".mkv"}:
            return p
    raise FileNotFoundError(f"No input.* under {sample_dir}")
