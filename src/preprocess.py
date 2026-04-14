"""Preprocessing pipeline for SealKey training data.

Two modes, selected by subcommand:

    solos   — single-subject clips. One output dir per input clip.
    doubles — two-subject clips with target-on-top + distractor below on one
              shared green screen. Used to teach the model to respect the
              hint channel in the presence of distractors.

Hint-channel generation is done at runtime by the dataloader using
`src.hint_generators` — no hint files are stored here.

Output per solo clip:
    <output>/<clip>/input.<ext>    — degraded RGB composite (model input)
    <output>/<clip>/gt_rgb.mkv     — clean pre-composite foreground RGB (FFV1)
    <output>/<clip>/gt_alpha.mkv   — clean matte (FFV1)

Output per double clip:
    <output>/double_<i>/input.<ext>               — multi-subject composite
    <output>/double_<i>/gt_rgb_target.mkv         — clean target RGB (FFV1)
    <output>/double_<i>/gt_alpha_target.mkv       — clean target alpha (FFV1)
    <output>/double_<i>/gt_rgb_distractor.mkv     — clean distractor RGB (FFV1)
    <output>/double_<i>/gt_alpha_distractor.mkv   — distractor alpha masked by
                                                     target (FFV1)
    <output>/double_<i>/manifest.json             — source clips + placement

Degraded `input` extension varies by rolled codec (see augment.CODECS). GT
streams are always FFV1 in .mkv — lossless, 4:4:4, built into every ffmpeg.

Layout expected for --input (both modes): one or more paths. Each may be
    - A clip directory containing RGBA PNG frames, OR
    - A parent directory containing clip subdirectories, OR
    - A directory containing .zip files that each extract to a clip
      directory of PNG frames.

Zips are unpacked in-place into a sibling directory named by the zip's stem.
Unpacking is idempotent — existing populated target dirs are not touched.

All preprocessing is 100% resumable: each clip's output is checked before
work. A solo clip is skipped if `input.*` and `alpha.*` both exist; a
double is skipped if `input.*`, `alpha_target.*`, `alpha_distractor.*`, and
`manifest.json` all exist. Delete partial outputs to force a rerun.

Usage:
    python -m src.preprocess solos \\
        --input raw_clips/ /fs/scratch/.../sealkey_wan_alpha \\
        --output solos/ --workers 8
    python -m src.preprocess doubles \\
        --input raw_clips/ /fs/scratch/.../sealkey_wan_alpha \\
        --output doubles/ --count 5000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from src.augment import augment_multi, _smooth_trajectory
from src.green_screen import generate_green_screen


# ---------------------------------------------------------------------------
# Subject-only geometric trajectory (runs BEFORE compositing).
# BORDER_CONSTANT=0 on both RGB and alpha: newly-exposed margin has alpha=0,
# so the static green screen shows through after compositing.
# ---------------------------------------------------------------------------

def _subject_trajectory(clip_in: Path, out_dir: Path,
                        rng: np.random.Generator) -> list[Path]:
    paths = sorted(p for p in clip_in.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG frames in {clip_in}")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    h, w = first.shape[:2]

    n = len(paths)
    tx    = _smooth_trajectory(n, amp=rng.uniform(0.01, 0.025) * w,  rng=rng)
    ty    = _smooth_trajectory(n, amp=rng.uniform(0.01, 0.025) * h,  rng=rng)
    theta = _smooth_trajectory(n, amp=rng.uniform(0.5,  1.5),        rng=rng)
    scale = 1.0 + _smooth_trajectory(n, amp=rng.uniform(0.005, 0.015), rng=rng)

    out_paths: list[Path] = []
    for i, p in enumerate(paths):
        frame = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError(f"Expected RGBA PNG at {p}")
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(theta[i]), float(scale[i]))
        M[0, 2] += float(tx[i])
        M[1, 2] += float(ty[i])
        rgb = cv2.warpAffine(frame[..., :3], M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        a = cv2.warpAffine(frame[..., 3], M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dst = out_dir / p.name
        cv2.imwrite(str(dst), np.dstack([rgb, a]))
        out_paths.append(dst)
    return out_paths


# ---------------------------------------------------------------------------
# Solo: composite one subject over one green screen
# ---------------------------------------------------------------------------

def _composite_solo(clip_in: Path, comp_out: Path, fg_rgb_out: Path,
                    alpha_out: Path, rng: np.random.Generator) -> str:
    """Split each RGBA frame into three streams:
        comp_out    — RGB composited onto a generated green screen (model input)
        fg_rgb_out  — clean pre-composite foreground RGB (GT target)
        alpha_out   — grayscale alpha (GT target)

    Returns the green-screen profile used (for manifest/logging).
    """
    paths = sorted(p for p in clip_in.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG frames in {clip_in}")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None or first.ndim != 3 or first.shape[2] != 4:
        raise ValueError(f"Expected RGBA PNG, got {None if first is None else first.shape} for {paths[0]}")
    h, w = first.shape[:2]

    gs_profile = str(rng.choice(["clean", "moderate", "messy"], p=[0.25, 0.45, 0.30]))
    gs = generate_green_screen(height=h, width=w,
                               seed=int(rng.integers(0, 2**31)),
                               profile=gs_profile)
    gs_f = gs.astype(np.float32)

    for p in paths:
        frame = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.shape[2] != 4:
            continue
        rgb = frame[..., :3]
        alpha_u8 = frame[..., 3]
        a = (alpha_u8.astype(np.float32) / 255.0)[..., None]
        comp = rgb.astype(np.float32) * a + gs_f * (1.0 - a)
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        cv2.imwrite(str(comp_out / p.name), comp)
        cv2.imwrite(str(fg_rgb_out / p.name), rgb)
        cv2.imwrite(str(alpha_out / p.name), alpha_u8)
    return gs_profile


def _solo_done(clip_out: Path) -> bool:
    """True if this solo clip has already been fully preprocessed."""
    return (
        clip_out.is_dir()
        and any(clip_out.glob("input.*"))
        and (clip_out / "gt_rgb.mkv").exists()
        and (clip_out / "gt_alpha.mkv").exists()
    )


def preprocess_solo(
    clip_in: Path,
    clip_out: Path,
    seed: int,
    fps: int = 24,
) -> None:
    """Trajectory → composite → augment. Writes input.<ext> + gt_rgb.mkv + gt_alpha.mkv.

    Idempotent: if all three outputs already exist, returns immediately.
    """
    if _solo_done(clip_out):
        return
    clip_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        moved   = tmp / "moved";   moved.mkdir()
        comp    = tmp / "comp";    comp.mkdir()   # composited RGB (degraded)
        fg_rgb  = tmp / "fg_rgb";  fg_rgb.mkdir() # clean pre-composite RGB
        a_dir   = tmp / "alpha";   a_dir.mkdir()  # clean alpha

        _subject_trajectory(clip_in, moved, rng)
        _composite_solo(moved, comp, fg_rgb, a_dir, rng)
        augment_multi(
            input_rgb_dir=comp,
            gt_rgb_dirs=[fg_rgb],  gt_rgb_labels=["gt_rgb"],
            alpha_dirs=[a_dir],    alpha_labels=["gt_alpha"],
            output_dir=clip_out,
            seed=int(rng.integers(0, 2**31)),
            fps=fps,
            skip={"geometric"},
        )


# ---------------------------------------------------------------------------
# Doubles: two subjects, one shared green screen, target always on top
# ---------------------------------------------------------------------------

def _sample_placement(subject_hw: tuple[int, int],
                      canvas_hw: tuple[int, int],
                      scale: float,
                      rng: np.random.Generator,
                      max_off: float = 0.2) -> tuple[int, int, int, int]:
    """Return (new_w, new_h, tx, ty) for placing a scaled subject on canvas.

    Subject may hang off-canvas by up to `max_off` fraction of its own size.
    """
    h, w = subject_hw
    ch, cw = canvas_hw
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    tx_min = -int(max_off * new_w)
    tx_max = cw - new_w + int(max_off * new_w)
    ty_min = -int(max_off * new_h)
    ty_max = ch - new_h + int(max_off * new_h)
    if tx_max < tx_min:
        tx_min = tx_max = (cw - new_w) // 2
    if ty_max < ty_min:
        ty_min = ty_max = (ch - new_h) // 2
    tx = int(rng.integers(tx_min, tx_max + 1))
    ty = int(rng.integers(ty_min, ty_max + 1))
    return new_w, new_h, tx, ty


def _place_on_canvas(frame_rgba: np.ndarray, canvas_hw: tuple[int, int],
                     new_w: int, new_h: int, tx: int, ty: int) -> np.ndarray:
    """Scale an RGBA frame to (new_h, new_w) and place at (ty, tx) on a
    transparent canvas of `canvas_hw`. Returns an RGBA uint8 canvas."""
    ch, cw = canvas_hw
    scaled = cv2.resize(frame_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    x0 = max(0, tx); y0 = max(0, ty)
    x1 = min(cw, tx + new_w); y1 = min(ch, ty + new_h)
    if x1 > x0 and y1 > y0:
        sx0 = x0 - tx; sy0 = y0 - ty
        sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
        canvas[y0:y1, x0:x1] = scaled[sy0:sy1, sx0:sx1]
    return canvas


def _compose_double(target_paths: list[Path],
                    distractor_paths: list[Path],
                    gs_bg: np.ndarray,
                    canvas_hw: tuple[int, int],
                    target_place: tuple[int, int, int, int],
                    distractor_place: tuple[int, int, int, int],
                    out_comp_dir: Path,
                    out_target_rgb_dir: Path,
                    out_target_alpha_dir: Path,
                    out_distractor_rgb_dir: Path,
                    out_distractor_alpha_dir: Path) -> None:
    """Composite bg → distractor → target (writes degraded input), and also
    emit clean per-subject RGB + alpha on canvas-sized transparent planes.

    Target streams are full. Distractor alpha is masked by the target (so the
    two GT alphas are disjoint, matching what's actually visible in the
    composite); distractor RGB is left unmasked — the alpha handles occlusion
    in any compositing loss."""
    gs_f = gs_bg.astype(np.float32)
    n = min(len(target_paths), len(distractor_paths))

    for i in range(n):
        t_frame = cv2.imread(str(target_paths[i]), cv2.IMREAD_UNCHANGED)
        d_frame = cv2.imread(str(distractor_paths[i]), cv2.IMREAD_UNCHANGED)
        if t_frame is None or t_frame.shape[2] != 4:
            continue
        if d_frame is None or d_frame.shape[2] != 4:
            continue

        t_on_canvas = _place_on_canvas(t_frame, canvas_hw, *target_place)
        d_on_canvas = _place_on_canvas(d_frame, canvas_hw, *distractor_place)

        t_rgb = t_on_canvas[..., :3]
        t_a_u8 = t_on_canvas[..., 3]
        t_a = (t_a_u8.astype(np.float32) / 255.0)[..., None]
        d_rgb = d_on_canvas[..., :3]
        d_a_u8 = d_on_canvas[..., 3]
        d_a = (d_a_u8.astype(np.float32) / 255.0)[..., None]

        comp = gs_f * (1.0 - d_a) + d_rgb.astype(np.float32) * d_a
        comp = comp * (1.0 - t_a) + t_rgb.astype(np.float32) * t_a
        comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)

        distractor_alpha_masked = np.clip(
            d_a_u8.astype(np.float32) * (1.0 - t_a[..., 0]), 0, 255,
        ).astype(np.uint8)

        name = f"{i:06d}.png"
        cv2.imwrite(str(out_comp_dir / name), comp_u8)
        cv2.imwrite(str(out_target_rgb_dir / name), t_rgb)
        cv2.imwrite(str(out_target_alpha_dir / name), t_a_u8)
        cv2.imwrite(str(out_distractor_rgb_dir / name), d_rgb)
        cv2.imwrite(str(out_distractor_alpha_dir / name), distractor_alpha_masked)


def _double_done(clip_out: Path) -> bool:
    """True if this double has already been fully preprocessed."""
    return (
        clip_out.is_dir()
        and any(clip_out.glob("input.*"))
        and (clip_out / "gt_rgb_target.mkv").exists()
        and (clip_out / "gt_alpha_target.mkv").exists()
        and (clip_out / "gt_rgb_distractor.mkv").exists()
        and (clip_out / "gt_alpha_distractor.mkv").exists()
        and (clip_out / "manifest.json").exists()
    )


def preprocess_double(
    clip_target: Path,
    clip_distractor: Path,
    clip_out: Path,
    seed: int,
    fps: int = 24,
) -> None:
    """Build one double clip. Writes input + alpha_target + alpha_distractor
    + manifest.json under clip_out.

    Idempotent: if all outputs already exist, returns immediately.
    """
    if _double_done(clip_out):
        return
    clip_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    first_paths = sorted(p for p in clip_target.iterdir() if p.suffix.lower() == ".png")
    if not first_paths:
        raise ValueError(f"No PNG frames in {clip_target}")
    first = cv2.imread(str(first_paths[0]), cv2.IMREAD_UNCHANGED)
    canvas_hw = first.shape[:2]

    t_scale = float(rng.uniform(0.6, 1.0))
    d_scale = float(rng.uniform(0.3, 0.9))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t_moved = tmp / "t_moved"; t_moved.mkdir()
        d_moved = tmp / "d_moved"; d_moved.mkdir()
        comp    = tmp / "comp";    comp.mkdir()
        t_rgb   = tmp / "t_rgb";   t_rgb.mkdir()
        t_a     = tmp / "t_a";     t_a.mkdir()
        d_rgb   = tmp / "d_rgb";   d_rgb.mkdir()
        d_a     = tmp / "d_a";     d_a.mkdir()

        _subject_trajectory(clip_target,     t_moved, rng)
        _subject_trajectory(clip_distractor, d_moved, rng)

        t_paths = sorted(t_moved.glob("*.png"))
        d_paths = sorted(d_moved.glob("*.png"))
        t_first = cv2.imread(str(t_paths[0]), cv2.IMREAD_UNCHANGED)
        d_first = cv2.imread(str(d_paths[0]), cv2.IMREAD_UNCHANGED)
        t_place = _sample_placement(t_first.shape[:2], canvas_hw, t_scale, rng)
        d_place = _sample_placement(d_first.shape[:2], canvas_hw, d_scale, rng)

        gs_profile = str(rng.choice(["clean", "moderate", "messy"], p=[0.25, 0.45, 0.30]))
        gs = generate_green_screen(height=canvas_hw[0], width=canvas_hw[1],
                                   seed=int(rng.integers(0, 2**31)),
                                   profile=gs_profile)

        _compose_double(
            t_paths, d_paths, gs, canvas_hw,
            t_place, d_place,
            comp, t_rgb, t_a, d_rgb, d_a,
        )

        augment_multi(
            input_rgb_dir=comp,
            gt_rgb_dirs=[t_rgb, d_rgb],
            gt_rgb_labels=["gt_rgb_target", "gt_rgb_distractor"],
            alpha_dirs=[t_a, d_a],
            alpha_labels=["gt_alpha_target", "gt_alpha_distractor"],
            output_dir=clip_out,
            seed=int(rng.integers(0, 2**31)),
            fps=fps,
            skip={"geometric"},
        )

    manifest = {
        "target": {
            "source": clip_target.name,
            "scale": t_scale,
            "new_w": t_place[0], "new_h": t_place[1],
            "tx": t_place[2], "ty": t_place[3],
            "z_order": 1,
        },
        "distractor": {
            "source": clip_distractor.name,
            "scale": d_scale,
            "new_w": d_place[0], "new_h": d_place[1],
            "tx": d_place[2], "ty": d_place[3],
            "z_order": 0,
        },
        "canvas_hw": list(canvas_hw),
        "gs_profile": str(gs_profile),
        "seed": seed,
    }
    (clip_out / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Shared utilities — zip unpacking + clip discovery (both resumable)
# ---------------------------------------------------------------------------

def _unzip_if_needed(zip_path: Path) -> Path:
    """Extract `zip_path` into a sibling directory named by its stem.

    Returns the extraction directory. Idempotent: if the target dir already
    exists and is non-empty, the zip is not re-extracted.
    """
    target = zip_path.with_suffix("")
    if target.is_dir() and any(target.iterdir()):
        return target
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    except zipfile.BadZipFile as e:
        raise SystemExit(f"Bad zip: {zip_path}: {e}")
    return target


def _find_clip_dirs(root: Path) -> list[Path]:
    """Return clip directories under `root`.

    A "clip directory" is any directory that contains at least one .png
    directly inside it. Walks recursively, so a zip that extracted to
    `root/foo/some_nested_dir/frames/*.png` is still discovered.
    """
    if not root.exists():
        return []
    # Case: root itself IS a clip (PNGs live directly inside).
    if root.is_dir() and any(p.suffix.lower() == ".png" for p in root.iterdir() if p.is_file()):
        return [root]
    out: list[Path] = []
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        if any(p.suffix.lower() == ".png" for p in d.iterdir() if p.is_file()):
            out.append(d)
    return out


def _prepare_inputs(input_paths: list[Path]) -> list[Path]:
    """For each input path: unpack any .zip files into sibling dirs, then
    discover clip dirs. Returns a sorted deduplicated list across all paths.
    """
    clips: list[Path] = []
    seen: set[Path] = set()
    for root in input_paths:
        if not root.exists():
            print(f"WARN: input path does not exist, skipping: {root}")
            continue
        # Unzip any zips found under root (recursive — handles scratch dirs
        # where users drop zips at arbitrary depths).
        zips = sorted(root.rglob("*.zip")) if root.is_dir() else ([root] if root.suffix.lower() == ".zip" else [])
        for z in zips:
            extracted = _unzip_if_needed(z)
            for c in _find_clip_dirs(extracted):
                if c not in seen:
                    seen.add(c); clips.append(c)
        # Also pick up non-zip clip dirs already present under root.
        if root.is_dir():
            for c in _find_clip_dirs(root):
                if c not in seen:
                    seen.add(c); clips.append(c)
    return sorted(clips)


def _solo_worker(job: tuple[Path, Path, int, int]) -> tuple[str, bool]:
    clip_in, clip_out, seed, fps = job
    skipped = _solo_done(clip_out)
    preprocess_solo(clip_in, clip_out, seed=seed, fps=fps)
    return clip_in.name, skipped


def _double_worker(job: tuple[Path, Path, Path, int, int]) -> tuple[str, bool]:
    clip_t, clip_d, clip_out, seed, fps = job
    skipped = _double_done(clip_out)
    preprocess_double(clip_t, clip_d, clip_out, seed=seed, fps=fps)
    return clip_out.name, skipped


def _run_pool(worker, jobs, workers: int) -> None:
    def _label(name: str, skipped: bool) -> str:
        return f"{name} (skipped, already done)" if skipped else name

    if workers <= 1:
        for i, job in enumerate(jobs):
            name, skipped = worker(job)
            print(f"[{i + 1}/{len(jobs)}] {_label(name, skipped)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), start=1):
                name, skipped = fut.result()
                print(f"[{i}/{len(jobs)}] {_label(name, skipped)}")


# ---------------------------------------------------------------------------
# Entry point — subcommands: solos, doubles
# ---------------------------------------------------------------------------

def _main_solos(args) -> None:
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    clips = _prepare_inputs(args.input)
    if not clips:
        raise SystemExit(f"No clips found under {args.input}")
    print(f"Discovered {len(clips)} source clips.")
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = [
        (clip_in, args.output / clip_in.name, base_seed + i, args.fps)
        for i, clip_in in enumerate(clips)
    ]
    _run_pool(_solo_worker, jobs, args.workers)
    print(f"\nDone — {len(jobs)} solo clips written under {args.output}/")


def _main_doubles(args) -> None:
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    clips = _prepare_inputs(args.input)
    if len(clips) < 2:
        raise SystemExit(f"Need at least 2 source clips under {args.input}, found {len(clips)}")
    print(f"Discovered {len(clips)} source clips.")
    args.output.mkdir(parents=True, exist_ok=True)

    sampler = np.random.default_rng(base_seed)
    jobs = []
    for i in range(args.count):
        idx_t, idx_d = sampler.choice(len(clips), size=2, replace=False)
        jobs.append((
            clips[int(idx_t)],
            clips[int(idx_d)],
            args.output / f"double_{i:05d}",
            base_seed + i + 1,
            args.fps,
        ))
    _run_pool(_double_worker, jobs, args.workers)
    print(f"\nDone — {args.count} doubles written under {args.output}/")


def main():
    parser = argparse.ArgumentParser(
        description="SealKey preprocessing: composite subjects on green screen "
                    "backgrounds and augment with camera/codec degradation. "
                    "All steps are idempotent — rerunning skips completed "
                    "unzips and completed clips.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    for name in ("solos", "doubles"):
        sp = sub.add_parser(name)
        sp.add_argument("--input", type=Path, nargs="+", required=True,
                        help="One or more input paths. Each may be: a clip "
                             "directory of RGBA PNG frames, a directory of "
                             "such clip directories, or a directory "
                             "containing .zip files that each extract to a "
                             "clip. Zips are unpacked in-place (skipped if "
                             "already unpacked).")
        sp.add_argument("--output", type=Path, required=True,
                        help="Output root. Existing completed clips are skipped.")
        sp.add_argument("--fps", type=int, default=24)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--workers", type=int, default=1)
        if name == "doubles":
            sp.add_argument("--count", type=int, required=True,
                            help="Number of doubles to generate.")

    args = parser.parse_args()
    if args.mode == "solos":
        _main_solos(args)
    elif args.mode == "doubles":
        _main_doubles(args)


if __name__ == "__main__":
    main()
