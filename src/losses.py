"""SealKey v1 losses: alpha BCE (or L1), edge-weighted RGB L1, delta reg."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def alpha_loss(alpha_logit: torch.Tensor, alpha_pred: torch.Tensor,
               gt_alpha: torch.Tensor, mode: str = "bce") -> torch.Tensor:
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(alpha_logit, gt_alpha)
    if mode == "l1":
        return (alpha_pred - gt_alpha).abs().mean()
    raise ValueError(f"alpha_loss mode: {mode}")


def edge_weighted_rgb_l1(rgb_pred: torch.Tensor, gt_rgb: torch.Tensor,
                         gt_alpha: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L1 on RGB, weighted by α·(1-α) so only semi-transparent edges
    contribute. Normalized by per-sample weight sum to keep scale stable.
    """
    w = gt_alpha * (1.0 - gt_alpha) * 4.0      # peak=1 at α=0.5
    diff = (rgb_pred - gt_rgb).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    num = (w * diff).sum(dim=(1, 2, 3))
    denom = w.sum(dim=(1, 2, 3)).clamp(min=eps)
    return (num / denom).mean()


def delta_regularizer(delta: torch.Tensor) -> torch.Tensor:
    return delta.abs().mean()


def compose_losses(pred: dict, batch: dict, cfg) -> tuple[torch.Tensor, dict]:
    l_alpha = alpha_loss(pred["alpha_logit"], pred["alpha_pred"],
                         batch["gt_alpha"], mode=cfg.alpha_loss)
    l_rgb = edge_weighted_rgb_l1(pred["rgb_pred"], batch["gt_rgb"], batch["gt_alpha"])
    l_delta = delta_regularizer(pred["delta"])
    total = cfg.w_alpha * l_alpha + cfg.w_rgb * l_rgb + cfg.delta_reg * l_delta
    return total, {
        "loss": total.detach(),
        "l_alpha": l_alpha.detach(),
        "l_rgb": l_rgb.detach(),
        "l_delta": l_delta.detach(),
    }
