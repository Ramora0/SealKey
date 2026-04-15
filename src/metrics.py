"""Matting metrics. All inputs are alpha arrays in [0,1], shape (H,W) or (B,H,W)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _iter(arr: np.ndarray):
    arr = _to_np(arr)
    if arr.ndim == 2:
        yield arr
    elif arr.ndim == 3:
        for i in range(arr.shape[0]):
            yield arr[i]
    elif arr.ndim == 4 and arr.shape[1] == 1:
        for i in range(arr.shape[0]):
            yield arr[i, 0]
    else:
        raise ValueError(f"Unsupported alpha shape: {arr.shape}")


def sad(pred, gt) -> float:
    """Sum of absolute differences, matting convention (/ 1000)."""
    ps, gs = list(_iter(pred)), list(_iter(gt))
    return float(np.mean([np.abs(p - g).sum() / 1000.0 for p, g in zip(ps, gs)]))


def mse(pred, gt) -> float:
    ps, gs = list(_iter(pred)), list(_iter(gt))
    return float(np.mean([((p - g) ** 2).mean() for p, g in zip(ps, gs)]))


def _sobel_mag(a: np.ndarray) -> np.ndarray:
    gx = sobel(a, axis=1)
    gy = sobel(a, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def grad(pred, gt) -> float:
    ps, gs = list(_iter(pred)), list(_iter(gt))
    vals = [np.abs(_sobel_mag(p) - _sobel_mag(g)).sum() / 1000.0 for p, g in zip(ps, gs)]
    return float(np.mean(vals))


def edge_weighted_l1(pred_rgb, gt_rgb, gt_alpha) -> float:
    pr = _to_np(pred_rgb); gr = _to_np(gt_rgb); ga = _to_np(gt_alpha)
    if pr.ndim == 4 and pr.shape[1] == 3:
        diff = np.abs(pr - gr).mean(axis=1, keepdims=True)
        w = ga * (1 - ga) * 4.0
        num = (w * diff).sum(axis=(1, 2, 3))
        denom = np.maximum(w.sum(axis=(1, 2, 3)), 1e-6)
        return float((num / denom).mean())
    raise ValueError(f"expected (B,3,H,W) for rgb, got {pr.shape}")
