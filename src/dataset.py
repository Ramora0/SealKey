"""SealKey v1 IterableDataset with K-clip in-worker buffer.

Each DataLoader worker owns a disjoint shard of sample directories. It holds
up to K decoded clips in RAM at once. Every iter step picks a random buffer
slot and yields the next frame from that clip; when a clip is exhausted it's
evicted and a fresh one is loaded. No mid-video seeking.

Shuffle entropy = K * frames_per_clip * num_workers, plenty for SGD on this
scale of dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.hint_sampling import sample_hint
from src.video_io import decode_gt_pair, decode_rgb_video, find_input_file


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def _deterministic_hash(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], "big")


def _sample_is_complete(d: Path) -> bool:
    """Required preprocess outputs exist. Labels: 'gt' for solos,
    'gt_target' + 'gt_all' for doubles (determined by manifest.json).
    """
    manifest_path = d / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    is_double = "target" in manifest and "distractor" in manifest
    labels = ("gt_target", "gt_all") if is_double else ("gt",)
    for label in labels:
        if not (d / f"{label}_rgb.mp4").is_file():
            return False
        if not (d / f"{label}_alpha.mkv").is_file():
            return False
    try:
        find_input_file(d)
    except FileNotFoundError:
        return False
    return True


def build_splits(data_root: Path, val_frac: float = 0.05) -> tuple[list[Path], list[Path]]:
    """Enumerate sample dirs under data_root/{solos,doubles}/* and split by a
    deterministic hash of the dir basename. Stable under additions. Silently
    drops dirs missing required files (partial preprocess outputs).
    """
    dirs: list[Path] = []
    dropped = 0
    for kind in ("solos", "doubles"):
        root = data_root / kind
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            if _sample_is_complete(p):
                dirs.append(p)
            else:
                dropped += 1
    if dropped:
        print(f"[dataset] build_splits: dropped {dropped} incomplete sample dirs")
    thresh = int(val_frac * 10000)
    train, val = [], []
    for d in dirs:
        if _deterministic_hash(d.name) % 10000 < thresh:
            val.append(d)
        else:
            train.append(d)
    return train, val


# ---------------------------------------------------------------------------
# In-worker clip buffer
# ---------------------------------------------------------------------------

@dataclass
class _Clip:
    sample_dir: Path
    kind: str                  # "solo" | "double_target" | "double_all"
    input_rgb: np.ndarray      # (T, H, W, 3) uint8
    gt_rgba: np.ndarray        # (T, H, W, 4) uint8  — supervising GT
    target_alpha: np.ndarray | None  # (T, H, W) uint8, only for doubles
    cursor: int = 0

    @property
    def n(self) -> int:
        return self.input_rgb.shape[0]


def _load_clip(sample_dir: Path, rng: np.random.Generator, max_frames: int) -> _Clip:
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    is_double = "target" in manifest and "distractor" in manifest

    input_path = find_input_file(sample_dir)
    input_rgb = decode_rgb_video(input_path, max_frames=max_frames)
    T = input_rgb.shape[0]

    if is_double:
        use_all = rng.random() < 0.5
        kind = "double_all" if use_all else "double_target"
        label = "gt_all" if use_all else "gt_target"
        gt_rgba = decode_gt_pair(sample_dir, label, max_frames=max_frames)
        # target_alpha always needed (chroma_key_gated). Decode gt_target
        # even in "all" mode — fast since file is local.
        if use_all:
            tgt = decode_gt_pair(sample_dir, "gt_target", max_frames=max_frames)
            target_alpha = tgt[..., 3]
        else:
            target_alpha = gt_rgba[..., 3]
    else:
        kind = "solo"
        gt_rgba = decode_gt_pair(sample_dir, "gt", max_frames=max_frames)
        target_alpha = None

    # Align lengths — augment pipeline may drop a frame on encode boundary.
    n_min = min(T, gt_rgba.shape[0])
    if target_alpha is not None:
        n_min = min(n_min, target_alpha.shape[0])
    input_rgb = input_rgb[:n_min]
    gt_rgba = gt_rgba[:n_min]
    if target_alpha is not None:
        target_alpha = target_alpha[:n_min]

    return _Clip(sample_dir, kind, input_rgb, gt_rgba, target_alpha)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _random_scale_crop(
    rgb: np.ndarray,
    gt_rgb: np.ndarray,
    gt_a: np.ndarray,
    tgt_a: np.ndarray | None,
    crop_hw: tuple[int, int],
    scale_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Shared random scale + random crop across all supervising arrays.
    Pads (reflect) if scaled frame smaller than crop in any dimension.
    """
    ch, cw = crop_hw
    h, w = rgb.shape[:2]
    s = float(rng.uniform(*scale_range))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    # Downscale → INTER_AREA; upscale → INTER_CUBIC.
    interp_rgb = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, (nw, nh), interpolation=interp_rgb)
    gt_rgb = cv2.resize(gt_rgb, (nw, nh), interpolation=interp_rgb)
    gt_a = cv2.resize(gt_a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if tgt_a is not None:
        tgt_a = cv2.resize(tgt_a, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, ch - nh)
    pad_w = max(0, cw - nw)
    if pad_h or pad_w:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        gt_rgb = cv2.copyMakeBorder(gt_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        gt_a = cv2.copyMakeBorder(gt_a, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        if tgt_a is not None:
            tgt_a = cv2.copyMakeBorder(tgt_a, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        nh, nw = rgb.shape[:2]

    y0 = int(rng.integers(0, nh - ch + 1))
    x0 = int(rng.integers(0, nw - cw + 1))
    rgb = rgb[y0:y0 + ch, x0:x0 + cw]
    gt_rgb = gt_rgb[y0:y0 + ch, x0:x0 + cw]
    gt_a = gt_a[y0:y0 + ch, x0:x0 + cw]
    if tgt_a is not None:
        tgt_a = tgt_a[y0:y0 + ch, x0:x0 + cw]
    return rgb, gt_rgb, gt_a, tgt_a


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SealKeyDataset(IterableDataset):
    def __init__(
        self,
        sample_dirs: list[Path],
        crop_hw: tuple[int, int] = (480, 832),
        scale_range: tuple[float, float] = (0.75, 1.5),
        k_clips: int = 4,
        max_frames_per_clip: int = 96,
        seed: int = 0,
        augment: bool = True,
    ):
        super().__init__()
        self.sample_dirs = sorted(sample_dirs)
        self.crop_hw = crop_hw
        self.scale_range = scale_range
        self.k_clips = k_clips
        self.max_frames = max_frames_per_clip
        self.seed = seed
        self.augment = augment

    def _my_shard(self) -> tuple[list[Path], np.random.Generator]:
        info = get_worker_info()
        if info is None:
            rng = np.random.default_rng(self.seed)
            return list(self.sample_dirs), rng
        shard = self.sample_dirs[info.id :: info.num_workers]
        rng = np.random.default_rng(self.seed + info.id + 1)
        return shard, rng

    def __iter__(self):
        pool, rng = self._my_shard()
        if not pool:
            return
        rng.shuffle(pool)
        cursor = 0
        buffer: list[_Clip] = []

        while True:
            # Refill.
            while len(buffer) < self.k_clips and cursor < len(pool):
                try:
                    buffer.append(_load_clip(pool[cursor], rng, self.max_frames))
                except Exception as e:
                    # Skip broken clips rather than killing the worker.
                    print(f"[dataset] skip {pool[cursor].name}: {e}")
                cursor += 1
            if not buffer:
                # Reshuffle and loop forever (IterableDataset with no epochs).
                rng.shuffle(pool)
                cursor = 0
                continue

            i = int(rng.integers(0, len(buffer)))
            clip = buffer[i]
            yield self._make_sample(clip, rng)
            clip.cursor += 1
            if clip.cursor >= clip.n:
                buffer.pop(i)

    def _make_sample(self, clip: _Clip, rng: np.random.Generator) -> dict:
        f = clip.cursor
        rgb_u8 = clip.input_rgb[f]                        # (H,W,3) RGB
        gt_rgb_u8 = clip.gt_rgba[f, ..., :3]              # (H,W,3) RGB
        gt_a_u8 = clip.gt_rgba[f, ..., 3]                 # (H,W)
        tgt_a_u8 = clip.target_alpha[f] if clip.target_alpha is not None else None

        if self.augment:
            rgb_u8, gt_rgb_u8, gt_a_u8, tgt_a_u8 = _random_scale_crop(
                rgb_u8, gt_rgb_u8, gt_a_u8, tgt_a_u8,
                self.crop_hw, self.scale_range, rng,
            )
        else:
            # Center-crop/pad to crop_hw so batch is shape-consistent.
            rgb_u8, gt_rgb_u8, gt_a_u8, tgt_a_u8 = _random_scale_crop(
                rgb_u8, gt_rgb_u8, gt_a_u8, tgt_a_u8,
                self.crop_hw, (1.0, 1.0), rng,
            )

        hint_u8, hint_name = sample_hint(
            clip.kind, rgb_u8, gt_a_u8, tgt_a_u8, rng,
        )

        rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        gt_rgb = torch.from_numpy(gt_rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        gt_a = torch.from_numpy(gt_a_u8.astype(np.float32) / 255.0).unsqueeze(0)
        hint = torch.from_numpy(hint_u8.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            "rgb": rgb, "hint": hint, "gt_rgb": gt_rgb, "gt_alpha": gt_a,
            "mode": clip.kind, "hint_name": hint_name,
        }


def collate(batch: list[dict]) -> dict:
    out: dict = {}
    tensor_keys = ("rgb", "hint", "gt_rgb", "gt_alpha")
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["mode"] = [b["mode"] for b in batch]
    out["hint_name"] = [b["hint_name"] for b in batch]
    return out
