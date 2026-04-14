"""Preview one randomized augmentation pass on a clip.

Runs preprocess_solo on an input RGBA-PNG clip, decodes the resulting
encoded input.<ext> + alpha.<ext> back to frames, generates a hint per
frame with a chosen (or random) generator, and writes a side-by-side
PNG sequence plus an optional stitched MP4 for inspection.

Layout per run:
    <out>/input.<ext>        — the codec-encoded composite (from preprocess)
    <out>/alpha.<ext>        — the codec-encoded GT alpha
    <out>/frames/%06d.png    — [input | alpha | hint] side-by-side RGB
    <out>/preview.mp4        — frames/ stitched into a playable video

Usage:
    python -m src.preview_augment --input path/to/rgba_clip \\
        --output previews/run1 --hint trimap
    python -m src.preview_augment --input ... --output ... --hint random
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.hint_generators import (
    box_hint,
    chroma_key_gated_hint,
    chroma_key_hint,
    trimap_hint,
    zero_hint,
)
from src.preprocess import _find_clip_dirs, _unzip_if_needed, preprocess_solo


HINT_CHOICES = ["trimap", "box", "chroma", "chroma_gated", "zero", "random"]

DEFAULT_SCRATCH = Path("/fs/scratch/PAS2836/lees_stuff/sealkey_wan_alpha")
DEFAULT_OUTPUT = Path("./preview_out")


def _pick_random_clip(scratch: Path, rng: random.Random) -> Path:
    """Pick a random .zip under `scratch`, unpack if needed, return a clip dir."""
    if not scratch.exists():
        raise SystemExit(f"Scratch dir not found: {scratch}")
    zips = sorted(scratch.rglob("*.zip"))
    if not zips:
        # Fall back to any clip dir already present under scratch.
        clips = _find_clip_dirs(scratch)
        if not clips:
            raise SystemExit(f"No .zip files or clip dirs under {scratch}")
        chosen = rng.choice(clips)
        print(f"[input] no zips — picked existing clip dir {chosen}")
        return chosen
    zip_path = rng.choice(zips)
    print(f"[input] picked zip {zip_path.name}")
    extracted = _unzip_if_needed(zip_path)
    clips = _find_clip_dirs(extracted)
    if not clips:
        raise SystemExit(f"Zip {zip_path} unpacked to {extracted} but contains no PNG clip dir")
    return clips[0]


def _decode_video(video_path: Path, out_dir: Path) -> list[Path]:
    """Decode a video file to %06d.png frames using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-start_number", "0",
            str(out_dir / "%06d.png"),
        ],
        check=True, capture_output=True,
    )
    return sorted(out_dir.glob("*.png"))


def _make_hint(kind: str, rgb: np.ndarray, alpha: np.ndarray,
               rng: np.random.Generator) -> tuple[str, np.ndarray]:
    """Return (effective_kind, hint). 'random' picks one per-frame."""
    if kind == "random":
        kind = str(rng.choice(["trimap", "box", "chroma", "chroma_gated", "zero"]))

    if kind == "trimap":
        return kind, trimap_hint(alpha, rng)
    if kind == "box":
        return kind, box_hint(alpha, rng)
    if kind == "chroma":
        return kind, chroma_key_hint(rgb, rng)
    if kind == "chroma_gated":
        return kind, chroma_key_gated_hint(rgb, alpha, rng)
    if kind == "zero":
        return kind, zero_hint(alpha.shape)
    raise ValueError(f"Unknown hint kind: {kind}")


def _label_frame(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 12 + 16, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _stack_side_by_side(input_rgb: np.ndarray, alpha: np.ndarray,
                        hint: np.ndarray, hint_label: str) -> np.ndarray:
    alpha_rgb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    hint_rgb = cv2.cvtColor(hint, cv2.COLOR_GRAY2BGR)
    panels = [
        _label_frame(input_rgb,  "input (degraded RGB)"),
        _label_frame(alpha_rgb,  "alpha (GT, co-degraded)"),
        _label_frame(hint_rgb,   f"hint ({hint_label})"),
    ]
    return np.hstack(panels)


def _stitch_preview(frames_dir: Path, out_dir: Path, fps: int) -> Path:
    """Stitch frames/%06d.png into a playable video using a locally-available
    encoder. Returns the written path (extension depends on encoder)."""
    from src.augment import CODECS  # probed at import time

    # Prefer a high-quality encoder for preview; mpeg2video is the HPC fallback.
    preference = ["libx264", "libx265", "libvpx-vp9", "mpeg2video"]
    codec = next(
        (c for enc in preference for c in CODECS if c["encoder"] == enc),
        CODECS[0],
    )
    out_path = out_dir / f"preview.{codec['container']}"

    enc = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", codec["encoder"],
        "-pix_fmt", "yuv420p",
    ]
    if codec["mode"] == "crf":
        enc += ["-crf", "18"]
        if codec["encoder"] == "libvpx-vp9":
            enc += ["-b:v", "0"]
    else:
        enc += ["-q:v", "3"]
    enc += [str(out_path)]

    r = subprocess.run(enc, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"preview stitch failed ({codec['encoder']}) exit={r.returncode}\n"
            f"--- stderr ---\n{r.stderr.decode(errors='replace')}"
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=None,
                        help="RGBA PNG clip directory (one subject). "
                             f"Default: pick a random .zip from {DEFAULT_SCRATCH}.")
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH,
                        help="Scratch dir to sample zips from when --input is omitted.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory for preview artifacts (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--hint", choices=HINT_CHOICES, default="random",
                        help="Hint generator to visualize.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true",
                        help="Reuse existing preprocess_solo output if present.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    clip_in = args.input if args.input is not None else _pick_random_clip(args.scratch, py_rng)
    print(f"[input] clip dir: {clip_in}")

    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Run one randomized augmentation via preprocess_solo.
    if not args.keep_existing:
        for old in list(args.output.glob("input.*")) + list(args.output.glob("alpha.*")):
            old.unlink()
    preprocess_solo(clip_in, args.output, seed=seed, fps=args.fps)

    input_videos = list(args.output.glob("input.*"))
    alpha_videos = list(args.output.glob("alpha.*"))
    if not input_videos or not alpha_videos:
        raise SystemExit(f"preprocess_solo did not produce input/alpha under {args.output}")
    input_video, alpha_video = input_videos[0], alpha_videos[0]
    print(f"[preprocess] seed={seed}  input={input_video.name}  alpha={alpha_video.name}")

    # 2. Decode both streams back to frames.
    decoded_root = args.output / "_decoded"
    if decoded_root.exists():
        shutil.rmtree(decoded_root)
    input_frames = _decode_video(input_video,  decoded_root / "input")
    alpha_frames = _decode_video(alpha_video,  decoded_root / "alpha")
    n = min(len(input_frames), len(alpha_frames))
    if n == 0:
        raise SystemExit("No frames decoded from preprocess output.")
    print(f"[decode] {n} frames")

    # 3. Generate hint + side-by-side per frame.
    frames_dir = args.output / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()

    kinds_used: set[str] = set()
    for i in range(n):
        rgb = cv2.imread(str(input_frames[i]), cv2.IMREAD_COLOR)
        alpha = cv2.imread(str(alpha_frames[i]), cv2.IMREAD_GRAYSCALE)
        if rgb is None or alpha is None:
            continue
        if rgb.shape[:2] != alpha.shape[:2]:
            alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        kind, hint = _make_hint(args.hint, rgb, alpha, rng)
        kinds_used.add(kind)
        panel = _stack_side_by_side(rgb, alpha, hint, kind)
        cv2.imwrite(str(frames_dir / f"{i:06d}.png"), panel)

    print(f"[hint] kinds used: {sorted(kinds_used)}")

    # 4. Stitch preview video (extension depends on what ffmpeg supports).
    preview_path = _stitch_preview(frames_dir, args.output, args.fps)
    print(f"[preview] wrote {preview_path}")


if __name__ == "__main__":
    main()
