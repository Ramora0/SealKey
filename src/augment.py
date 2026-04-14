"""Clip-level randomized augmentation with split-quality streams.

Pipeline processes one *degraded* RGB input alongside N *clean* GT RGB streams
and N clean alpha streams. Physical stages (geometric, motion_blur, DoF, noise)
apply to every stream with identical parameters — same affine, same ffmpeg
motion-blur roll, same DoF mask/sigma, same per-frame noise sample. Quality-
degrading stages (banding, lossy codec) apply only to the degraded input; GT
streams are encoded with FFV1 lossless.

The point: a model trained against the clean GT will learn to undo green-screen
composite, codec artifacts, and color banding, while preserving real scene
content (motion, DoF, sensor noise).

Outputs (codec stage default):
    <output_dir>/input.<ext>        — degraded RGB (rolled lossy codec)
    <output_dir>/<gt_rgb_label>.mkv — clean GT RGB (FFV1)
    <output_dir>/<alpha_label>.mkv  — clean GT alpha (FFV1)
    <output_dir>/augment_settings.json

Usage:
    from src.augment import augment_multi
    augment_multi(input_rgb_dir=composite_dir,
                  gt_rgb_dirs=[fg_rgb_dir], gt_rgb_labels=["gt_rgb"],
                  alpha_dirs=[alpha_dir],    alpha_labels=["gt_alpha"],
                  output_dir=out, seed=0, fps=24)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Codec pool
# ---------------------------------------------------------------------------

_ALL_CODECS = [
    {"name": "h264",  "encoder": "libx264",     "mode": "crf",    "range": (28, 42), "container": "mp4"},
    {"name": "h265",  "encoder": "libx265",     "mode": "crf",    "range": (32, 46), "container": "mp4"},
    {"name": "mpeg2", "encoder": "mpeg2video",  "mode": "qscale", "range": (10, 25), "container": "mpg"},
    {"name": "vp9",   "encoder": "libvpx-vp9",  "mode": "crf",    "range": (40, 55), "container": "webm"},
]


def _available_codecs() -> list[dict]:
    """Filter _ALL_CODECS to encoders the local ffmpeg actually supports.

    HPC ffmpeg builds often ship without libx264 / libx265 / libvpx, so we
    probe once and keep whatever's left. If nothing is available we raise —
    a codec-less preprocess pass would silently skip the codec stage.
    """
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"Could not probe ffmpeg encoders: {e}")
    listing = r.stdout.decode(errors="replace")
    available = [c for c in _ALL_CODECS if f" {c['encoder']} " in listing]
    if not available:
        raise RuntimeError(
            "None of the expected encoders are available in this ffmpeg build: "
            f"{[c['encoder'] for c in _ALL_CODECS]}"
        )
    return available


CODECS = _available_codecs()


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
# Stage 1: geometric — shared warp across RGB + all alphas
# ---------------------------------------------------------------------------

def _apply_geometric_all(
    rgb_dirs: list[Path],
    alpha_dirs: list[Path],
    rgb_outs: list[Path],
    alpha_outs: list[Path],
    rng: np.random.Generator,
) -> dict:
    """Roll one smooth affine trajectory and apply to every RGB + alpha stream.

    RGB uses BORDER_REPLICATE (padded edges look like extended scene); alpha
    uses BORDER_CONSTANT=0 (newly-exposed margin is transparent).
    """
    rgb_paths_list = [sorted(d.glob("*.png")) for d in rgb_dirs]
    alpha_paths_list = [sorted(d.glob("*.png")) for d in alpha_dirs]
    ref = rgb_paths_list[0] if rgb_paths_list else alpha_paths_list[0]
    first = cv2.imread(str(ref[0]), cv2.IMREAD_UNCHANGED)
    h, w = first.shape[:2]
    n = len(ref)

    tx_amp    = float(rng.uniform(0.01, 0.025) * w)
    ty_amp    = float(rng.uniform(0.01, 0.025) * h)
    theta_amp = float(rng.uniform(0.5, 1.5))
    scale_amp = float(rng.uniform(0.005, 0.015))
    tx    = _smooth_trajectory(n, amp=tx_amp, rng=rng)
    ty    = _smooth_trajectory(n, amp=ty_amp, rng=rng)
    theta = _smooth_trajectory(n, amp=theta_amp, rng=rng)
    scale = 1.0 + _smooth_trajectory(n, amp=scale_amp, rng=rng)

    for i in range(n):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(theta[i]), float(scale[i]))
        M[0, 2] += float(tx[i])
        M[1, 2] += float(ty[i])

        for paths, out in zip(rgb_paths_list, rgb_outs):
            rgb = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
            rgb_w = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(out / paths[i].name), rgb_w)
        for paths, out in zip(alpha_paths_list, alpha_outs):
            a = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
            a_w = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cv2.imwrite(str(out / paths[i].name), a_w)

    return {
        "on": True,
        "tx_amp_px": tx_amp, "ty_amp_px": ty_amp,
        "theta_amp_deg": theta_amp, "scale_amp": scale_amp,
    }


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
    rgb_paths_list: list[list[Path]],
    rgb_is_degraded: list[bool],
    alpha_paths_list: list[list[Path]],
    rgb_outs: list[Path],
    alpha_outs: list[Path],
    rng: np.random.Generator,
) -> dict:
    """Per-frame optical ops across multiple RGB streams + alphas.

    Physical (apply to all RGB streams): DoF, noise.
    Quality-degrading (apply only to `rgb_is_degraded=True` streams): banding.
    DoF on alphas uses each alpha's own edges.

    Noise is sampled once per frame and added to every RGB stream so the clean
    GT RGB and degraded input share the same noise pattern — otherwise a
    clean-GT L1 loss would penalize the model for keeping noise the input
    actually contains.
    """
    do_dof = bool(rng.random() < 0.4)
    dof_sigma = float(rng.uniform(1.5, 4.0))
    dof_thickness = int(rng.integers(6, 20))

    do_noise = bool(rng.random() < 0.6)
    noise_sigma = float(rng.uniform(3.0, 15.0))

    do_banding = bool(rng.random() < 0.25)
    banding_bits = int(rng.integers(3, 6))

    assert len(rgb_paths_list) == len(rgb_is_degraded) == len(rgb_outs)
    assert len(alpha_paths_list) == len(alpha_outs)
    n = len(rgb_paths_list[0]) if rgb_paths_list else len(alpha_paths_list[0])

    for i in range(n):
        rgbs = [cv2.imread(str(paths[i]), cv2.IMREAD_COLOR) for paths in rgb_paths_list]
        alphas = [cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for paths in alpha_paths_list]

        if do_dof:
            # Union of all alphas guides the RGB DoF blur; each alpha uses its
            # own edges.
            if alphas:
                union = np.maximum.reduce(alphas)
            else:
                union = np.zeros(rgbs[0].shape[:2], dtype=np.uint8)
            union_band = _alpha_edge_mask(union, dof_thickness)
            m = union_band[..., None]
            new_rgbs = []
            for rgb in rgbs:
                br = cv2.GaussianBlur(rgb, (0, 0), sigmaX=dof_sigma)
                new_rgbs.append(
                    (rgb.astype(np.float32) * (1 - m) + br.astype(np.float32) * m).astype(np.uint8)
                )
            rgbs = new_rgbs

            new_alphas = []
            for a in alphas:
                band = _alpha_edge_mask(a, dof_thickness)
                ba = cv2.GaussianBlur(a, (0, 0), sigmaX=dof_sigma)
                a = (a.astype(np.float32) * (1 - band) + ba.astype(np.float32) * band).astype(np.uint8)
                new_alphas.append(a)
            alphas = new_alphas

        if do_noise and rgbs:
            # One noise sample per frame, shared across all RGB streams so
            # clean-GT stays noise-aligned with degraded input.
            noise = rng.normal(0, noise_sigma, rgbs[0].shape).astype(np.float32)
            rgbs = [np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                    for rgb in rgbs]

        if do_banding:
            levels = 2 ** banding_bits
            step = 256 // levels
            rgbs = [((rgb // step) * step).astype(np.uint8) if deg else rgb
                    for rgb, deg in zip(rgbs, rgb_is_degraded)]
            # Alphas are always clean — no banding on alpha.

        for rgb, paths, out in zip(rgbs, rgb_paths_list, rgb_outs):
            cv2.imwrite(str(out / paths[i].name), rgb)
        for a, paths, out in zip(alphas, alpha_paths_list, alpha_outs):
            cv2.imwrite(str(out / paths[i].name), a)

    return {
        "dof": {"on": do_dof, "sigma": dof_sigma, "thickness": dof_thickness},
        "noise": {"on": do_noise, "sigma": noise_sigma,
                  "note": "shared sample across all RGB streams"},
        "banding": {"on": do_banding, "bits": banding_bits,
                    "note": "applied only to degraded streams"},
    }


# ---------------------------------------------------------------------------
# Stage 4: codec encode
# ---------------------------------------------------------------------------

# FFV1 in Matroska: truly lossless, 4:4:4 yuv, built into every ffmpeg core
# build (no external lib dependency). Used for all GT (clean) streams.
LOSSLESS = {"encoder": "ffv1", "container": "mkv", "pix_fmt": "yuv444p"}


def _ffmpeg_lossless_encode(in_dir: Path, out_path: Path, fps: int) -> None:
    """Encode a PNG sequence losslessly (FFV1 in .mkv, yuv444p)."""
    enc = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(in_dir / "%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", LOSSLESS["encoder"],
        "-pix_fmt", LOSSLESS["pix_fmt"],
        "-level", "3",
        str(out_path),
    ]
    r = subprocess.run(enc, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg FFV1 encode failed exit={r.returncode}\n"
            f"--- stderr ---\n{r.stderr.decode(errors='replace')}"
        )


def _ffmpeg_codec_encode(in_dir: Path, out_path: Path, fps: int,
                         codec: dict, level: int, pix_fmt: str) -> None:
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
        "-pix_fmt", pix_fmt,
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
    input_rgb_dir: Path,
    gt_rgb_dirs: list[Path],
    gt_rgb_labels: list[str],
    alpha_dirs: list[Path],
    alpha_labels: list[str],
    output_dir: Path,
    seed: int | None = None,
    fps: int = 24,
    skip: set[str] | None = None,
) -> None:
    """Augment a degraded RGB input alongside clean GT RGB + alpha streams.

    Every RGB stream shares the same physical augmentations (geometric,
    motion_blur, DoF, noise — same parameters and same noise sample). Quality-
    degrading stages (banding, lossy codec) are applied only to `input_rgb_dir`;
    GT streams are encoded with FFV1 (lossless, 4:4:4, in .mkv).

    Outputs:
        <output_dir>/input.<ext>          — degraded RGB (rolled lossy codec)
        <output_dir>/<gt_rgb_label>.mkv   — clean GT RGB (FFV1)
        <output_dir>/<alpha_label>.mkv    — clean GT alpha (FFV1)
        <output_dir>/augment_settings.json

    Args:
        input_rgb_dir: directory of 3-channel PNGs — the degraded model input.
        gt_rgb_dirs: directories of 3-channel PNGs — clean pre-composite
            foreground RGB streams. Filenames must match input_rgb_dir's.
        gt_rgb_labels: output stems for each gt_rgb dir.
        alpha_dirs: directories of single-channel PNGs — clean GT alphas.
        alpha_labels: output stems for each alpha dir.
        seed: RNG seed.
        fps: frame rate for ffmpeg stages.
        skip: optional stage names to skip: {"geometric", "motion_blur",
            "per_frame", "codec"}. Skipping codec writes PNG sequences into
            per-stream subdirectories instead of encoded video.
    """
    if len(gt_rgb_dirs) != len(gt_rgb_labels):
        raise ValueError("gt_rgb_dirs and gt_rgb_labels length mismatch")
    if len(alpha_dirs) != len(alpha_labels):
        raise ValueError("alpha_dirs and alpha_labels length mismatch")

    skip = skip or set()
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings: dict = {"seed": seed, "fps": fps, "skip": sorted(skip)}

    rgb_paths = sorted(p for p in input_rgb_dir.iterdir() if p.suffix.lower() == ".png")
    if not rgb_paths:
        raise ValueError(f"No PNG frames in {input_rgb_dir}")
    settings["n_frames"] = len(rgb_paths)

    # All RGB streams unified: input first (degraded), GT streams after (clean).
    all_rgb_dirs = [input_rgb_dir, *gt_rgb_dirs]
    all_rgb_labels = ["input", *gt_rgb_labels]
    rgb_is_degraded = [True] + [False] * len(gt_rgb_dirs)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp = Path(tmp_root)

        def _mk_parallel_dirs(tag: str, n: int) -> list[Path]:
            dirs = [tmp / f"{tag}_{i}" for i in range(n)]
            for d in dirs:
                d.mkdir()
            return dirs

        # Stage 0: sequential rename to %06d.png (ffmpeg needs this).
        cur_rgbs = _mk_parallel_dirs("00_rgb", len(all_rgb_dirs))
        cur_alphas = _mk_parallel_dirs("00_a", len(alpha_dirs))

        seq_names: list[str] = []
        for i, p in enumerate(rgb_paths):
            name = f"{i:06d}.png"
            seq_names.append(name)
            for src_dir, dst_dir in zip(all_rgb_dirs, cur_rgbs):
                s = src_dir / p.name
                if not s.exists():
                    raise ValueError(f"Missing RGB frame {p.name} in {src_dir}")
                shutil.copy(s, dst_dir / name)
            for src_dir, dst_dir in zip(alpha_dirs, cur_alphas):
                s = src_dir / p.name
                if not s.exists():
                    raise ValueError(f"Missing alpha frame {p.name} in {src_dir}")
                shutil.copy(s, dst_dir / name)

        # Stage 1: geometric
        if "geometric" not in skip:
            nxt_rgbs = _mk_parallel_dirs("01_rgb", len(cur_rgbs))
            nxt_alphas = _mk_parallel_dirs("01_a", len(cur_alphas))
            # _stage_geometric_multi takes one "reference" RGB for sizing;
            # run it once across ALL streams by passing the combined list as
            # alphas is awkward. Instead, we roll the affine here and apply it
            # uniformly. Keep existing helper but extend it inline:
            settings["geometric"] = _apply_geometric_all(
                cur_rgbs, cur_alphas, nxt_rgbs, nxt_alphas, rng,
            )
            cur_rgbs, cur_alphas = nxt_rgbs, nxt_alphas
        else:
            settings["geometric"] = {"on": False}

        # Stage 2: motion blur (physical — applied to all streams)
        do_blur = "motion_blur" not in skip and rng.random() < 0.7
        if do_blur:
            factor = int(rng.integers(3, 6))
            n_mix = factor
            settings["motion_blur"] = {"on": True, "factor": factor, "n_mix": n_mix}
            nxt_rgbs = _mk_parallel_dirs("02_rgb", len(cur_rgbs))
            nxt_alphas = _mk_parallel_dirs("02_a", len(cur_alphas))
            for d_in, d_out in zip(cur_rgbs + cur_alphas, nxt_rgbs + nxt_alphas):
                _ffmpeg_motion_blur(d_in, d_out, fps, factor, n_mix)
            cur_rgbs, cur_alphas = nxt_rgbs, nxt_alphas
        else:
            settings["motion_blur"] = {"on": False}

        # Stage 3: per-frame optical (DoF+noise physical; banding degraded-only)
        if "per_frame" not in skip:
            nxt_rgbs = _mk_parallel_dirs("03_rgb", len(cur_rgbs))
            nxt_alphas = _mk_parallel_dirs("03_a", len(cur_alphas))
            settings["per_frame"] = _stage_per_frame_multi(
                [sorted(d.glob("*.png")) for d in cur_rgbs],
                rgb_is_degraded,
                [sorted(d.glob("*.png")) for d in cur_alphas],
                nxt_rgbs, nxt_alphas, rng,
            )
            cur_rgbs, cur_alphas = nxt_rgbs, nxt_alphas
        else:
            settings["per_frame"] = {"on": False}

        # Stage 4: encode. Degraded stream gets the rolled lossy codec; clean
        # streams always get FFV1 lossless.
        if "codec" not in skip:
            codec = CODECS[int(rng.integers(len(CODECS)))]
            lo, hi = codec["range"]
            level = int(rng.integers(lo, hi + 1))
            ext = codec["container"]
            pix_fmt = "yuv422p" if rng.random() < 0.5 else "yuv420p"
            settings["codec"] = {
                "input": {"encoder": codec["encoder"], "mode": codec["mode"],
                          "level": level, "container": ext, "pix_fmt": pix_fmt},
                "gt": {"encoder": LOSSLESS["encoder"],
                       "container": LOSSLESS["container"],
                       "pix_fmt": LOSSLESS["pix_fmt"]},
            }
            # input (degraded) — rolled codec
            _ffmpeg_codec_encode(cur_rgbs[0], output_dir / f"input.{ext}",
                                 fps, codec, level, pix_fmt)
            # clean GT RGBs — FFV1
            for d, label in zip(cur_rgbs[1:], gt_rgb_labels):
                _ffmpeg_lossless_encode(d, output_dir / f"{label}.mkv", fps)
            # clean alphas — FFV1
            for d, label in zip(cur_alphas, alpha_labels):
                _ffmpeg_lossless_encode(d, output_dir / f"{label}.mkv", fps)
            (output_dir / "augment_settings.json").write_text(
                json.dumps(settings, indent=2)
            )
            return
        settings["codec"] = {"on": False}

        # Codec skipped: copy final PNG streams into per-stream subdirectories.
        for d, label in zip(cur_rgbs, all_rgb_labels):
            out = output_dir / label; out.mkdir(exist_ok=True)
            for name, orig in zip(seq_names, rgb_paths):
                shutil.copy(d / name, out / orig.name)
        for d, label in zip(cur_alphas, alpha_labels):
            out = output_dir / label; out.mkdir(exist_ok=True)
            for name, orig in zip(seq_names, rgb_paths):
                shutil.copy(d / name, out / orig.name)
        (output_dir / "augment_settings.json").write_text(
            json.dumps(settings, indent=2)
        )


