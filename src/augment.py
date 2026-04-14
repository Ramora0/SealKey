"""Clip-level randomized augmentation for RGB + N-alpha frame sequences.

Input: an RGB PNG sequence + one or more parallel single-channel alpha PNG
sequences (matching filenames). Every stage applies identical random
parameters across the RGB stream and every alpha stream, so they emerge
spatially and temporally aligned at the same codec level.

Output:
    If codec stage runs (default): `output_dir/input.<ext>` (degraded RGB) and
    `output_dir/<label>.<ext>` for each alpha input, where `<ext>` is the
    rolled codec's native container (mp4 / mpg / webm).
    If `skip={"codec"}`: separate PNG sequences under `output_dir/rgb/` and
    `output_dir/<label>/`.

Pipeline (in order):
    1. Geometric trajectory — smooth per-frame affine shared across RGB+alphas.
    2. Motion blur — ffmpeg minterpolate + tmix per stream, identical params.
    3. Per-frame optical — DoF (RGB edge-guide uses the *union* of all alphas;
       each alpha uses its own), RGB-only sharpen/noise, banding on all.
    4. Codec — single (codec, CRF) roll applied to every stream.

The legacy RGBA wrapper `augment_clip(input_dir, output_dir, ...)` splits an
RGBA PNG sequence into RGB + single-alpha, calls augment_multi, and emits
`input.<ext>` + `alpha.<ext>` (or legacy RGBA PNGs when codec is skipped).

Usage:
    from src.augment import augment_clip, augment_multi
    augment_clip(Path("frames/"), Path("out/"), seed=0, fps=24)
    augment_multi(rgb_dir, [alpha_target_dir, alpha_distractor_dir],
                  out_dir, alpha_labels=["alpha_target", "alpha_distractor"],
                  seed=0, fps=24)
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Codec pool
# ---------------------------------------------------------------------------

CODECS = [
    {"name": "h264",  "encoder": "libx264",     "mode": "crf",    "range": (28, 42), "container": "mp4"},
    {"name": "h265",  "encoder": "libx265",     "mode": "crf",    "range": (32, 46), "container": "mp4"},
    {"name": "mpeg2", "encoder": "mpeg2video",  "mode": "qscale", "range": (10, 25), "container": "mpg"},
    {"name": "vp9",   "encoder": "libvpx-vp9",  "mode": "crf",    "range": (40, 55), "container": "webm"},
]


# ---------------------------------------------------------------------------
# Smooth trajectory (used by geometric stage + preprocess subject motion)
# ---------------------------------------------------------------------------

def _smooth_trajectory(n: int, amp: float, rng: np.random.Generator,
                       n_harmonics: int = 3) -> np.ndarray:
    """Zero-mean low-frequency signal of length n, bounded ~[-amp, +amp]."""
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    t = np.linspace(0, 1, n, dtype=np.float32)
    sig = np.zeros(n, dtype=np.float32)
    for _ in range(n_harmonics):
        freq = rng.uniform(0.4, 2.2)
        phase = rng.uniform(0, 2 * np.pi)
        weight = rng.uniform(0.5, 1.0)
        sig += weight * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    sig -= sig.mean()
    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = sig / peak * amp
    return sig


# ---------------------------------------------------------------------------
# RGBA ↔ (RGB, alpha) helpers
# ---------------------------------------------------------------------------

def _split_rgba(src_paths: list[Path], rgb_dir: Path, alpha_dir: Path) -> None:
    for p in src_paths:
        f = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(rgb_dir / p.name), f[..., :3])
        cv2.imwrite(str(alpha_dir / p.name), f[..., 3])


def _merge_rgba(rgb_dir: Path, alpha_dir: Path, out_dir: Path,
                names: list[str], target_hw: tuple[int, int]) -> None:
    th, tw = target_hw
    for name in names:
        rgb_p = rgb_dir / name
        a_p = alpha_dir / name
        if not rgb_p.exists() or not a_p.exists():
            continue
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        a = cv2.imread(str(a_p), cv2.IMREAD_GRAYSCALE)
        if rgb is None or a is None:
            continue
        rgb = rgb[:th, :tw]
        a = a[:th, :tw]
        if rgb.shape[:2] != (th, tw):
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        if a.shape[:2] != (th, tw):
            a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / name), np.dstack([rgb, a]))


# ---------------------------------------------------------------------------
# Stage 1: geometric — shared warp across RGB + all alphas
# ---------------------------------------------------------------------------

def _stage_geometric_multi(
    rgb_paths: list[Path],
    alpha_paths_list: list[list[Path]],
    rgb_out: Path,
    alpha_outs: list[Path],
    rng: np.random.Generator,
) -> None:
    n = len(rgb_paths)
    first = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]

    tx = _smooth_trajectory(n, amp=rng.uniform(0.01, 0.025) * w, rng=rng)
    ty = _smooth_trajectory(n, amp=rng.uniform(0.01, 0.025) * h, rng=rng)
    theta = _smooth_trajectory(n, amp=rng.uniform(0.5, 1.5), rng=rng)
    scale = 1.0 + _smooth_trajectory(n, amp=rng.uniform(0.005, 0.015), rng=rng)

    for i in range(n):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(theta[i]), float(scale[i]))
        M[0, 2] += float(tx[i])
        M[1, 2] += float(ty[i])

        rgb = cv2.imread(str(rgb_paths[i]), cv2.IMREAD_COLOR)
        rgb_w = cv2.warpAffine(rgb, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(rgb_out / rgb_paths[i].name), rgb_w)

        for paths, out in zip(alpha_paths_list, alpha_outs):
            a = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
            a_w = cv2.warpAffine(a, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cv2.imwrite(str(out / paths[i].name), a_w)


# ---------------------------------------------------------------------------
# Stage 2: motion blur (one ffmpeg per stream, identical params)
# ---------------------------------------------------------------------------

def _ffmpeg_motion_blur(in_dir: Path, out_dir: Path,
                        fps: int, factor: int, n_mix: int) -> None:
    target_fps = fps * factor
    vf = (
        f"minterpolate=fps={target_fps}:mi_mode=mci:me_mode=bidir:mc_mode=aobmc,"
        f"tmix=frames={n_mix},"
        f"fps={fps}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(in_dir / "%06d.png"),
            "-vf", vf,
            "-start_number", "0",
            str(out_dir / "%06d.png"),
        ],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# Stage 3: per-frame optical ops (DoF / sharpen / noise / banding)
# ---------------------------------------------------------------------------

def _alpha_edge_mask(alpha: np.ndarray, thickness: int) -> np.ndarray:
    edges = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    band = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), sigmaX=thickness)
    m = band.max()
    if m > 1e-6:
        band /= m
    return band


def _stage_per_frame_multi(
    rgb_paths: list[Path],
    alpha_paths_list: list[list[Path]],
    rgb_out: Path,
    alpha_outs: list[Path],
    rng: np.random.Generator,
) -> None:
    do_dof = rng.random() < 0.4
    dof_sigma = rng.uniform(1.5, 4.0)
    dof_thickness = int(rng.integers(6, 20))

    do_sharpen = rng.random() < 0.35
    sharpen_amount = rng.uniform(1.2, 2.5)
    sharpen_sigma = rng.uniform(0.8, 2.0)

    do_noise = rng.random() < 0.6
    noise_sigma = rng.uniform(3.0, 15.0)

    do_banding = rng.random() < 0.25
    banding_bits = int(rng.integers(3, 6))

    n = len(rgb_paths)
    for i in range(n):
        rgb = cv2.imread(str(rgb_paths[i]), cv2.IMREAD_COLOR)
        alphas = [cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for paths in alpha_paths_list]

        if do_dof:
            # RGB uses the union of all alphas as DoF edge guide; each alpha
            # uses its own edges. For the single-alpha wrapper case, union
            # reduces to the one alpha, so behavior matches the old pipeline.
            if alphas:
                union = np.maximum.reduce(alphas)
            else:
                union = np.zeros(rgb.shape[:2], dtype=np.uint8)
            union_band = _alpha_edge_mask(union, dof_thickness)
            br = cv2.GaussianBlur(rgb, (0, 0), sigmaX=dof_sigma)
            m = union_band[..., None]
            rgb = (rgb.astype(np.float32) * (1 - m) + br.astype(np.float32) * m).astype(np.uint8)

            new_alphas = []
            for a in alphas:
                band = _alpha_edge_mask(a, dof_thickness)
                ba = cv2.GaussianBlur(a, (0, 0), sigmaX=dof_sigma)
                a = (a.astype(np.float32) * (1 - band) + ba.astype(np.float32) * band).astype(np.uint8)
                new_alphas.append(a)
            alphas = new_alphas

        if do_sharpen:
            blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sharpen_sigma)
            sharp = rgb.astype(np.float32) * (1 + sharpen_amount) \
                    - blurred.astype(np.float32) * sharpen_amount
            rgb = np.clip(sharp, 0, 255).astype(np.uint8)

        if do_noise:
            noise = rng.normal(0, noise_sigma, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if do_banding:
            levels = 2 ** banding_bits
            step = 256 // levels
            rgb = ((rgb // step) * step).astype(np.uint8)
            alphas = [((a // step) * step).astype(np.uint8) for a in alphas]

        cv2.imwrite(str(rgb_out / rgb_paths[i].name), rgb)
        for a, paths, out in zip(alphas, alpha_paths_list, alpha_outs):
            cv2.imwrite(str(out / paths[i].name), a)


# ---------------------------------------------------------------------------
# Stage 4: codec encode
# ---------------------------------------------------------------------------

def _ffmpeg_codec_encode(in_dir: Path, out_path: Path, fps: int,
                         codec: dict, level: int) -> None:
    """Encode a PNG sequence to a single MP4 at the given codec/level.

    No decode — the MP4 is the codec pass's direct output. Downstream
    consumers (dataloader) decode at read time.
    """
    enc = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(in_dir / "%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", codec["encoder"],
        "-pix_fmt", "yuv420p",
    ]
    if codec["mode"] == "crf":
        enc += ["-crf", str(level)]
        if codec["encoder"] == "libvpx-vp9":
            enc += ["-b:v", "0"]
    else:
        enc += ["-q:v", str(level)]
    enc += [str(out_path)]
    r = subprocess.run(enc, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encode failed ({codec['encoder']} @ {codec['mode']}={level}) "
            f"exit={r.returncode}\n--- stderr ---\n{r.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def augment_multi(
    rgb_dir: Path,
    alpha_dirs: list[Path],
    output_dir: Path,
    alpha_labels: list[str],
    seed: int | None = None,
    fps: int = 24,
    skip: set[str] | None = None,
) -> None:
    """Augment one RGB stream + N parallel alpha streams together.

    Every stage applies identical random parameters to every stream so the
    result is a single codec pass at a single quality level, spatially and
    temporally aligned.

    Args:
        rgb_dir: directory of 3-channel PNG frames.
        alpha_dirs: list of directories, each with single-channel PNG frames
            whose filenames match rgb_dir's.
        output_dir: destination directory. Created if missing.
        alpha_labels: output filename stems for each alpha, in order matching
            alpha_dirs. Used as `<label>.<ext>` in the codec output.
        seed: RNG seed.
        fps: frame rate for ffmpeg stages.
        skip: optional stage names to skip: {"geometric", "motion_blur",
            "per_frame", "codec"}. Skipping codec writes the final PNG
            sequences into `output_dir/rgb/` and `output_dir/<label>/`.
    """
    if len(alpha_dirs) != len(alpha_labels):
        raise ValueError(f"alpha_dirs ({len(alpha_dirs)}) and alpha_labels ({len(alpha_labels)}) must match")

    skip = skip or set()
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() == ".png")
    if not rgb_paths:
        raise ValueError(f"No PNG frames in {rgb_dir}")

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp = Path(tmp_root)

        # Stage 0: sequential rename to %06d.png (ffmpeg needs this).
        cur_rgb = tmp / "00_rgb"; cur_rgb.mkdir()
        cur_alphas = [tmp / f"00_a{i}" for i in range(len(alpha_dirs))]
        for d in cur_alphas:
            d.mkdir()

        seq_names: list[str] = []
        for i, p in enumerate(rgb_paths):
            name = f"{i:06d}.png"
            seq_names.append(name)
            shutil.copy(p, cur_rgb / name)
            for j, alpha_dir in enumerate(alpha_dirs):
                a_src = alpha_dir / p.name
                if not a_src.exists():
                    raise ValueError(f"Missing alpha frame {p.name} in {alpha_dir}")
                shutil.copy(a_src, cur_alphas[j] / name)

        # Stage 1: geometric
        if "geometric" not in skip:
            nxt_rgb = tmp / "01_rgb"; nxt_rgb.mkdir()
            nxt_alphas = [tmp / f"01_a{i}" for i in range(len(alpha_dirs))]
            for d in nxt_alphas:
                d.mkdir()
            _stage_geometric_multi(
                sorted(cur_rgb.glob("*.png")),
                [sorted(d.glob("*.png")) for d in cur_alphas],
                nxt_rgb, nxt_alphas, rng,
            )
            cur_rgb, cur_alphas = nxt_rgb, nxt_alphas

        # Stage 2: motion blur
        if "motion_blur" not in skip and rng.random() < 0.7:
            factor = int(rng.integers(3, 6))
            n_mix = factor
            nxt_rgb = tmp / "02_rgb"; nxt_rgb.mkdir()
            _ffmpeg_motion_blur(cur_rgb, nxt_rgb, fps, factor, n_mix)
            nxt_alphas = []
            for i, d in enumerate(cur_alphas):
                nd = tmp / f"02_a{i}"; nd.mkdir()
                _ffmpeg_motion_blur(d, nd, fps, factor, n_mix)
                nxt_alphas.append(nd)
            cur_rgb, cur_alphas = nxt_rgb, nxt_alphas

        # Stage 3: per-frame optical
        if "per_frame" not in skip:
            nxt_rgb = tmp / "03_rgb"; nxt_rgb.mkdir()
            nxt_alphas = [tmp / f"03_a{i}" for i in range(len(cur_alphas))]
            for d in nxt_alphas:
                d.mkdir()
            _stage_per_frame_multi(
                sorted(cur_rgb.glob("*.png")),
                [sorted(d.glob("*.png")) for d in cur_alphas],
                nxt_rgb, nxt_alphas, rng,
            )
            cur_rgb, cur_alphas = nxt_rgb, nxt_alphas

        # Stage 4: codec (encode-only; MP4s are the output)
        if "codec" not in skip:
            codec = CODECS[int(rng.integers(len(CODECS)))]
            lo, hi = codec["range"]
            level = int(rng.integers(lo, hi + 1))
            ext = codec["container"]
            _ffmpeg_codec_encode(cur_rgb, output_dir / f"input.{ext}", fps, codec, level)
            for d, label in zip(cur_alphas, alpha_labels):
                _ffmpeg_codec_encode(d, output_dir / f"{label}.{ext}", fps, codec, level)
            return

        # Codec skipped: copy final PNG streams into per-stream subdirectories.
        rgb_out = output_dir / "rgb"; rgb_out.mkdir(exist_ok=True)
        for name, orig in zip(seq_names, rgb_paths):
            shutil.copy(cur_rgb / name, rgb_out / orig.name)
        for d, label in zip(cur_alphas, alpha_labels):
            lbl_out = output_dir / label; lbl_out.mkdir(exist_ok=True)
            for name, orig in zip(seq_names, rgb_paths):
                shutil.copy(d / name, lbl_out / orig.name)


def augment_clip(
    input_dir: Path,
    output_dir: Path,
    seed: int | None = None,
    fps: int = 24,
    skip: set[str] | None = None,
) -> None:
    """Augment an RGBA PNG sequence (single-alpha wrapper over augment_multi).

    When codec runs, outputs `input.<ext>` + `alpha.<ext>` to output_dir.
    When codec is skipped, outputs legacy-style RGBA PNGs to output_dir.
    """
    skip = skip or set()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG frames in {input_dir}")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None or first.ndim != 3 or first.shape[2] != 4:
        raise ValueError(f"Expected RGBA PNGs; {paths[0]} is {None if first is None else first.shape}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rgb_tmp = td / "rgb"; rgb_tmp.mkdir()
        alpha_tmp = td / "alpha"; alpha_tmp.mkdir()
        _split_rgba(paths, rgb_tmp, alpha_tmp)

        if "codec" in skip:
            # Legacy RGBA inspection output: run augment_multi with codec
            # skipped, then merge back to RGBA PNGs.
            stage_out = td / "stage_out"; stage_out.mkdir()
            augment_multi(rgb_tmp, [alpha_tmp], stage_out,
                          alpha_labels=["alpha"],
                          seed=seed, fps=fps, skip=skip)
            _merge_rgba(stage_out / "rgb", stage_out / "alpha",
                        output_dir, [p.name for p in paths], first.shape[:2])
        else:
            augment_multi(rgb_tmp, [alpha_tmp], output_dir,
                          alpha_labels=["alpha"],
                          seed=seed, fps=fps, skip=skip)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment a PNG-frame video clip.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input directory of RGBA PNG frames.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for augmented frames.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip", nargs="*", default=None,
                        choices=["geometric", "motion_blur", "per_frame", "codec"],
                        help="Stages to skip.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    augment_clip(args.input, args.output, seed=seed, fps=args.fps,
                 skip=set(args.skip) if args.skip else None)
    print(f"Augmented clip → {args.output}")


if __name__ == "__main__":
    main()
