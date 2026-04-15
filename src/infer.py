"""Run a trained SealKey checkpoint on a single image.

Default hint = chroma key (HSV threshold on green/blue screen colors), derived
straight from the input RGB. No extra inputs needed for a quick test.

    python -m src.infer --ckpt runs/v1/ckpt/best.pt --image test.png

Optional overrides:
    --mask rough.png      # build a trimap via erode/dilate of a binary mask
    --trimap trimap.png   # pre-built 3-level trimap (0/128/255)

Writes into --out (default "out/"):
    hint.png, alpha.png, rgb_pred.png, composite.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.hint_generators import chroma_key_hint
from src.model import SealKeyNet
from src.viz import _checkerboard


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Failed to read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_trimap(mask_u8: np.ndarray, erode_k: int, dilate_k: int) -> np.ndarray:
    binary = (mask_u8 >= 128).astype(np.uint8) * 255
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    fg = cv2.erode(binary, ek)
    bg_not = cv2.dilate(binary, dk)
    out = np.zeros_like(binary)
    out[bg_not > 0] = 128
    out[fg > 0] = 255
    return out


def _pad_to_multiple(x: np.ndarray, m: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = x.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return x, (0, 0)
    return cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT_101), (ph, pw)


def load_model(ckpt_path: Path, device: str) -> SealKeyNet:
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = SealKeyNet(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def run(model: SealKeyNet, rgb_u8: np.ndarray, hint_u8: np.ndarray,
        device: str, bf16: bool) -> dict[str, np.ndarray]:
    rgb_p, _ = _pad_to_multiple(rgb_u8)
    hint_p, _ = _pad_to_multiple(hint_u8)
    rgb_t = torch.from_numpy(rgb_p.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    hint_t = torch.from_numpy(hint_p.astype(np.float32) / 255.0)[None, None].to(device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16):
        pred = model(rgb_t, hint_t)
    H, W = rgb_u8.shape[:2]
    alpha = pred["alpha_pred"][0, 0, :H, :W].float().cpu().clamp(0, 1).numpy()
    rgb_pred = pred["rgb_pred"][0, :, :H, :W].float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return {
        "alpha": (alpha * 255.0).astype(np.uint8),
        "rgb_pred": (rgb_pred * 255.0).astype(np.uint8),
    }


def composite_over_checkerboard(rgb_u8: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    bg = _checkerboard(rgb_u8.shape[0], rgb_u8.shape[1])
    a = alpha_u8.astype(np.float32)[..., None] / 255.0
    out = rgb_u8.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, default=None,
                   help="Optional binary mask → trimap. Overrides chroma-key default.")
    p.add_argument("--trimap", type=Path, default=None,
                   help="Optional pre-built 3-level trimap. Overrides chroma-key default.")
    p.add_argument("--erode", type=int, default=15)
    p.add_argument("--dilate", type=int, default=15)
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the chroma keyer's thresholds (s_min, v_min).")
    p.add_argument("--resize", type=str, default=None,
                   help="Resize input before inference. 'train' uses the training crop "
                        "(480x832); HxW (e.g. '720x1280') uses a custom size. Model is "
                        "fully convolutional so this is optional — only useful if output "
                        "quality depends on scale matching training.")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    args = p.parse_args()

    if args.mask is not None and args.trimap is not None:
        raise SystemExit("--mask and --trimap are mutually exclusive.")

    args.out.mkdir(parents=True, exist_ok=True)

    rgb = _read_rgb(args.image)
    if args.resize is not None:
        if args.resize == "train":
            target_h, target_w = 480, 832
        else:
            try:
                target_h, target_w = (int(x) for x in args.resize.lower().split("x"))
            except Exception:
                raise SystemExit(f"--resize must be 'train' or HxW (e.g. '720x1280'), got {args.resize!r}")
        interp = cv2.INTER_AREA if target_h * target_w < rgb.shape[0] * rgb.shape[1] else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=interp)
    if args.trimap is not None:
        hint = _read_gray(args.trimap)
    elif args.mask is not None:
        hint = build_trimap(_read_gray(args.mask), args.erode, args.dilate)
    else:
        hint = chroma_key_hint(rgb, np.random.default_rng(args.seed))
    if hint.shape[:2] != rgb.shape[:2]:
        hint = cv2.resize(hint, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    model = load_model(args.ckpt, args.device)
    out = run(model, rgb, hint, args.device, args.bf16)

    cv2.imwrite(str(args.out / "hint.png"), hint)
    cv2.imwrite(str(args.out / "alpha.png"), out["alpha"])
    cv2.imwrite(str(args.out / "rgb_pred.png"), cv2.cvtColor(out["rgb_pred"], cv2.COLOR_RGB2BGR))
    comp = composite_over_checkerboard(out["rgb_pred"], out["alpha"])
    cv2.imwrite(str(args.out / "composite.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"wrote outputs to {args.out}/")


if __name__ == "__main__":
    main()
