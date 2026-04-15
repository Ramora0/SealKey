"""Run MatAnyone on a single image frame (no video).

    uv pip install git+https://github.com/pq-yang/MatAnyone
    python -m src.run_matanyone --image out/test.png --mask out/hint.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from matanyone import InferenceCore
# MatAnyone's model module captures `get_default_device()` into a module
# global at import time (matanyone/model/matanyone.py:18) and uses it inside
# encode_image, overriding whatever we pass to InferenceCore. On Mac that
# pins ops to MPS, which then hits a hard Metal assertion in the attention
# path. Overwrite the captured global.
import matanyone.model.matanyone as _mam
_mam.device = torch.device("cpu")


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise SystemExit(f"Failed to read {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=Path("out/test.png"))
    p.add_argument("--mask", type=Path, required=True,
                   help="Binary first-frame mask (white=foreground).")
    p.add_argument("--ckpt", type=str, default="PeiqingYang/MatAnyone",
                   help="HF repo id or local ckpt path for MatAnyone.")
    p.add_argument("--warmup", type=int, default=10,
                   help="Re-run first_frame_pred N times so memory stabilizes.")
    p.add_argument("--out", type=Path, default=Path("out"))
    default_device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
    p.add_argument("--device", type=str, default=default_device)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rgb = _read_rgb(args.image)
    mask = _read_mask(args.mask, rgb.shape[:2])

    processor = InferenceCore(args.ckpt, device=torch.device(args.device))
    image_t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(args.device) / 255.0
    mask_t = torch.from_numpy(mask).float().to(args.device)

    frames_dir = args.out / "matanyone_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _save(prob, name: str) -> np.ndarray:
        a = processor.output_prob_to_mask(prob)
        a_u8 = (a.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(frames_dir / f"{name}.png"), a_u8)
        return a_u8

    with torch.inference_mode():
        processor.step(image_t, mask_t, objects=[1])
        out_prob = processor.step(image_t, first_frame_pred=True)
        _save(out_prob, "iter_00")
        for i in tqdm(range(args.warmup), desc="warmup", disable=args.warmup == 0):
            out_prob = processor.step(image_t, first_frame_pred=True)
            _save(out_prob, f"iter_{i+1:02d}")
        out_prob = processor.step(image_t)
        a_u8 = _save(out_prob, "final")

    dst = args.out / "matanyone_alpha.png"
    cv2.imwrite(str(dst), a_u8)
    print(f"wrote {dst} (+ {args.warmup+2} frames in {frames_dir})")


if __name__ == "__main__":
    main()
