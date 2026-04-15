"""Visualization panels for logged images.

Composites are drawn on a checkerboard, not black, so alpha=1 collapse is
visible. Output panels are uint8 (H_total, W_total, 3) RGB arrays.
"""

from __future__ import annotations

import numpy as np
import torch


def _checkerboard(h: int, w: int, square: int = 16) -> np.ndarray:
    ys = (np.arange(h)[:, None] // square) % 2
    xs = (np.arange(w)[None, :] // square) % 2
    board = (ys ^ xs).astype(np.uint8)
    return np.where(board[..., None], 180, 110).repeat(3, axis=-1).astype(np.uint8)


def _to_np_img(t: torch.Tensor) -> np.ndarray:
    # (C,H,W) [0,1] → (H,W,C) uint8.
    arr = t.detach().float().cpu().clamp(0, 1).numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return (arr * 255.0).astype(np.uint8)


def _composite(rgb_u8: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    bg = _checkerboard(rgb_u8.shape[0], rgb_u8.shape[1])
    a = alpha_u8.astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    out = rgb_u8.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_panel(batch: dict, pred: dict, n: int = 2) -> np.ndarray:
    """Build a row-per-sample panel:
    [input_rgb | hint | pred_α | gt_α | pred_composite | gt_composite]
    """
    n = min(n, batch["rgb"].shape[0])
    rows = []
    for i in range(n):
        rgb = _to_np_img(batch["rgb"][i])
        hint = _to_np_img(batch["hint"][i])
        pa = _to_np_img(pred["alpha_pred"][i])
        ga = _to_np_img(batch["gt_alpha"][i])
        pc = _composite(_to_np_img(pred["rgb_pred"][i]), pa[..., 0])
        gc = _composite(_to_np_img(batch["gt_rgb"][i]), ga[..., 0])
        rows.append(np.concatenate([rgb, hint, pa, ga, pc, gc], axis=1))
    return np.concatenate(rows, axis=0)
