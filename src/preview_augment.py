"""Preview one randomized augmentation pass on a clip.

Supports both preprocess modes:

    solo     — one subject on a generated green screen. GT stream: gt.mkv.
    doubles  — target + distractor on one shared green screen. GT streams:
               gt_target.mkv (target only) and gt_all.mkv (target ∪ visible
               distractor).

All GT streams are RGBA (FFV1 yuva444p) — we decode and split into rgb/alpha
panels per stream.

Layout per run (solo):
    <out>/input.<ext>         — codec-encoded composite
    <out>/gt.mkv              — clean GT RGBA (FFV1)
    <out>/frames/%06d.png     — [input | gt_rgb | gt_alpha | hint]
    <out>/preview.<ext>       — frames stitched to a playable video

Layout per run (doubles):
    <out>/input.<ext>
    <out>/gt_target.mkv, <out>/gt_all.mkv
    <out>/frames/%06d.png     — [input | target_rgb | target_alpha |
                                 all_rgb | all_alpha | hint]

Usage:
    python -m src.preview_augment --input path/to/rgba_clip \\
        --output previews/run1 --hint trimap
    python -m src.preview_augment --mode doubles --hint box
    python -m src.preview_augment --mode doubles \\
        --input path/to/target path/to/distractor
"""

from __future__ import annotations

import argparse
import json
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
from src.preprocess import (
    _find_clip_dirs,
    _unzip_if_needed,
    preprocess_double,
    preprocess_solo,
)


HINT_CHOICES = ["trimap", "box", "chroma", "chroma_gated", "zero", "random"]
HINT_KINDS = ["trimap", "box", "chroma", "chroma_gated", "zero"]

DEFAULT_SCRATCH = Path("/fs/scratch/PAS2836/lees_stuff/sealkey_wan_alpha")
DEFAULT_OUTPUT = Path("./preview_out")


# ---------------------------------------------------------------------------
# Clip selection
# ---------------------------------------------------------------------------

def _all_available_clips(scratch: Path) -> list[Path]:
    """Return a list of clip dirs available under scratch. Unpacks any zips
    it encounters along the way. Falls back to already-extracted clip dirs."""
    if not scratch.exists():
        raise SystemExit(f"Scratch dir not found: {scratch}")
    zips = sorted(scratch.rglob("*.zip"))
    clips: list[Path] = []
    for z in zips:
        extracted = _unzip_if_needed(z)
        clips.extend(_find_clip_dirs(extracted))
    if not clips:
        clips = _find_clip_dirs(scratch)
    if not clips:
        raise SystemExit(f"No .zip files or clip dirs under {scratch}")
    return clips


def _pick_random_clips(scratch: Path, rng: random.Random, n: int) -> list[Path]:
    """Pick `n` distinct clip dirs from scratch."""
    clips = _all_available_clips(scratch)
    if len(clips) < n:
        raise SystemExit(f"Need {n} clips, only found {len(clips)} under {scratch}")
    return rng.sample(clips, n)


# ---------------------------------------------------------------------------
# Hint + panel helpers
# ---------------------------------------------------------------------------

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
        kind = str(rng.choice(HINT_KINDS))

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


def _gray_to_bgr(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)


def _stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    """Resize panels to the first panel's size (defensively) and hstack."""
    th, tw = panels[0].shape[:2]
    norm = []
    for p in panels:
        if p.shape[:2] != (th, tw):
            p = cv2.resize(p, (tw, th), interpolation=cv2.INTER_LINEAR)
        norm.append(p)
    return np.hstack(norm)


def _stitch_preview(frames_dir: Path, out_dir: Path, fps: int) -> Path:
    """Stitch frames/%06d.png into a playable video using a locally-available
    encoder. Returns the written path (extension depends on encoder)."""
    from src.augment import CODECS  # probed at import time

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


# ---------------------------------------------------------------------------
# Mode dispatch: produce (input_video, [(gt_label, gt_mkv), ...])
# ---------------------------------------------------------------------------

def _run_solo(clip_in: Path, out_dir: Path, seed: int, fps: int,
              ) -> tuple[Path, list[tuple[str, Path]]]:
    preprocess_solo(clip_in, out_dir, seed=seed, fps=fps)
    inputs = list(out_dir.glob("input.*"))
    gt = out_dir / "gt.mkv"
    if not inputs or not gt.exists():
        raise SystemExit(f"preprocess_solo missing outputs under {out_dir}")
    return inputs[0], [("gt", gt)]


def _run_doubles(clip_t: Path, clip_d: Path, out_dir: Path, seed: int, fps: int,
                 ) -> tuple[Path, list[tuple[str, Path]]]:
    preprocess_double(clip_t, clip_d, out_dir, seed=seed, fps=fps)
    inputs = list(out_dir.glob("input.*"))
    target = out_dir / "gt_target.mkv"
    all_mkv = out_dir / "gt_all.mkv"
    if not inputs or not target.exists() or not all_mkv.exists():
        raise SystemExit(f"preprocess_double missing outputs under {out_dir}")
    return inputs[0], [("target", target), ("all", all_mkv)]


def _clean_output(out_dir: Path) -> None:
    """Clear previous preview artifacts so a stale run doesn't short-circuit
    the idempotency check in preprocess_{solo,double}."""
    for pat in ("input.*", "gt*.mkv", "manifest.json", "augment_settings.json",
                "preview.*"):
        for p in out_dir.glob(pat):
            p.unlink()
    for sub in ("_decoded", "frames"):
        d = out_dir / sub
        if d.exists():
            shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["solo", "doubles"], default="solo",
                        help="Which preprocess pipeline to preview.")
    parser.add_argument("--input", type=Path, nargs="+", default=None,
                        help="RGBA PNG clip directory. solo: 1 path. "
                             "doubles: 2 paths (target, distractor). "
                             f"Default: pick random from {DEFAULT_SCRATCH}.")
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH,
                        help="Scratch dir to sample clips from when --input is omitted.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--hint", choices=HINT_CHOICES, default="random",
                        help="Hint generator to visualize.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true",
                        help="Reuse existing preprocess output if present.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    n_clips_needed = 1 if args.mode == "solo" else 2
    if args.input is None:
        clips = _pick_random_clips(args.scratch, py_rng, n_clips_needed)
    else:
        if len(args.input) != n_clips_needed:
            raise SystemExit(
                f"--mode {args.mode} needs {n_clips_needed} --input path(s), "
                f"got {len(args.input)}"
            )
        clips = list(args.input)
    for i, c in enumerate(clips):
        print(f"[input] clip {i}: {c}")

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        _clean_output(args.output)

    # 1. Run one randomized augmentation pass.
    if args.mode == "solo":
        input_video, gt_streams = _run_solo(clips[0], args.output, seed, args.fps)
    else:
        input_video, gt_streams = _run_doubles(
            clips[0], clips[1], args.output, seed, args.fps,
        )
    print(f"[preprocess] mode={args.mode}  seed={seed}  "
          f"input={input_video.name}  gt=[{', '.join(p.name for _, p in gt_streams)}]")

    settings_path = args.output / "augment_settings.json"
    if settings_path.exists():
        print(f"[augment settings] (from {settings_path.name})")
        print(json.dumps(json.loads(settings_path.read_text()), indent=2))

    # 2. Decode input + each GT stream back to PNG frames. GT mkvs are RGBA
    # via FFV1 yuva444p, so ffmpeg emits 4-channel PNGs natively.
    decoded_root = args.output / "_decoded"
    if decoded_root.exists():
        shutil.rmtree(decoded_root)
    input_frames = _decode_video(input_video, decoded_root / "input")
    gt_frames_by_label: dict[str, list[Path]] = {
        label: _decode_video(mkv, decoded_root / label) for label, mkv in gt_streams
    }

    n = min(len(input_frames),
            *(len(fs) for fs in gt_frames_by_label.values()))
    if n == 0:
        raise SystemExit("No frames decoded from preprocess output.")
    print(f"[decode] {n} frames")

    # 3. Generate hint + side-by-side per frame. Lock hint kind + seed for
    # the whole clip so only inputs change frame-to-frame.
    frames_dir = args.output / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()

    hint_kind = args.hint
    if hint_kind == "random":
        hint_kind = str(rng.choice(HINT_KINDS))
    hint_seed = int(rng.integers(0, 2**31))
    # For doubles, we drive the hint from the "all" alpha (everything visible
    # in frame) — that matches the more general training hint. Fall back to
    # the sole GT stream for solo.
    hint_source_label = "all" if "all" in gt_frames_by_label else gt_streams[0][0]
    print(f"[hint] kind={hint_kind}  seed={hint_seed}  source={hint_source_label}")

    for i in range(n):
        rgb = cv2.imread(str(input_frames[i]), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        th, tw = rgb.shape[:2]

        panels: list[np.ndarray] = [_label_frame(rgb, "input (degraded)")]
        gt_rgba_by_label: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, frames in gt_frames_by_label.items():
            rgba = cv2.imread(str(frames[i]), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                raise SystemExit(
                    f"Expected 4-channel RGBA from {label}, got shape "
                    f"{None if rgba is None else rgba.shape}"
                )
            gt_rgb = rgba[..., :3]
            gt_a = rgba[..., 3]
            if gt_rgb.shape[:2] != (th, tw):
                gt_rgb = cv2.resize(gt_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
                gt_a = cv2.resize(gt_a, (tw, th), interpolation=cv2.INTER_LINEAR)
            gt_rgba_by_label[label] = (gt_rgb, gt_a)
            panels.append(_label_frame(gt_rgb, f"{label}_rgb"))
            panels.append(_label_frame(_gray_to_bgr(gt_a), f"{label}_alpha"))

        hint_alpha = gt_rgba_by_label[hint_source_label][1]
        frame_rng = np.random.default_rng(hint_seed)
        _, hint = _make_hint(hint_kind, rgb, hint_alpha, frame_rng)
        panels.append(_label_frame(_gray_to_bgr(hint), f"hint ({hint_kind})"))

        cv2.imwrite(str(frames_dir / f"{i:06d}.png"), _stack_panels(panels))

    # 4. Stitch preview video.
    preview_path = _stitch_preview(frames_dir, args.output, args.fps)
    print(f"[preview] wrote {preview_path}")


if __name__ == "__main__":
    main()
