"""PyAV wrappers for SealKey video decoding.

Preprocess outputs two shapes of video:
- `input.<ext>` (mp4/webm/mpg) — degraded 3-channel RGB.
- `gt*.mkv` — FFV1 yuva444p, 4-channel RGBA (clean pre-composite fg + alpha).

Both are decoded start-to-finish into numpy arrays. No mid-video seeking.
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


def _frame_to_rgba(frame: av.VideoFrame) -> np.ndarray:
    """Convert an FFV1/yuva444p VideoFrame to (H, W, 4) uint8 RGBA.

    PyAV's ndarray output for 4-channel formats varies by version. We reformat
    to rgba, then defensively handle both interleaved (H,W,4) and packed
    (H, 4W) returns.
    """
    f = frame.reformat(format="rgba")
    arr = f.to_ndarray()
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr
    if arr.ndim == 2 and arr.shape[1] == f.width * 4:
        return arr.reshape(f.height, f.width, 4)
    raise RuntimeError(
        f"Unexpected RGBA ndarray shape {arr.shape} for frame {f.width}x{f.height}"
    )


def decode_rgba_ffv1(path: Path, max_frames: int | None = None) -> np.ndarray:
    """Decode an FFV1 yuva444p RGBA video to (T, H, W, 4) uint8 (RGBA)."""
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            out.append(_frame_to_rgba(frame))
            if max_frames is not None and len(out) >= max_frames:
                break
    if not out:
        raise RuntimeError(f"Decoded zero frames from {path}")
    return np.stack(out, axis=0)


def find_input_file(sample_dir: Path) -> Path:
    """Locate `input.<ext>` within a sample directory."""
    for p in sample_dir.iterdir():
        if p.is_file() and p.stem == "input" and p.suffix.lower() in {".mp4", ".webm", ".mpg", ".mkv"}:
            return p
    raise FileNotFoundError(f"No input.* under {sample_dir}")
