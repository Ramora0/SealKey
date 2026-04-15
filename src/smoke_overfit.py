"""Smoke-test ladder. Each rung must pass before the next.

    python -m src.smoke_overfit --rung 0                   # shape/FCN/stem grad
    python -m src.smoke_overfit --rung 1 --data_root ...   # overfit 2 frames
    python -m src.smoke_overfit --rung 2 --data_root ...   # overfit 1 clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import TrainConfig
from src.dataset import SealKeyDataset, build_splits, collate
from src.losses import compose_losses
from src.model import SealKeyNet


def rung0() -> None:
    """Shape + FCN + stem-grad sanity. No data required."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use pretrained=False here to avoid hitting the network during smoke tests.
    model = SealKeyNet(pretrained=False).to(device)
    model.train()

    for h, w in [(480, 832), (256, 256), (720, 1280)]:
        rgb = torch.rand(2, 3, h, w, device=device)
        hint = torch.rand(2, 1, h, w, device=device)
        out = model(rgb, hint)
        for k in ("rgb_pred", "alpha_pred", "delta", "alpha_logit"):
            assert out[k].shape[-2:] == (h, w), f"{k} shape {out[k].shape}"
        assert out["rgb_pred"].shape[1] == 3
        assert out["alpha_pred"].shape[1] == 1
        print(f"  [{h}x{w}] shapes OK")

    # Stem-grad sanity: confirm the 4th input channel receives gradient.
    rgb = torch.rand(2, 3, 128, 128, device=device)
    hint = torch.rand(2, 1, 128, 128, device=device)
    out = model(rgb, hint)
    loss = out["alpha_pred"].mean() + out["rgb_pred"].mean()
    loss.backward()
    # Locate stem conv (same logic as _find_stem_conv but we know structure).
    for _, mod in model.encoder.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.in_channels == 4:
            g = mod.weight.grad
            assert g is not None, "no grad on stem conv"
            rgb_grad = g[:, :3].abs().sum().item()
            hint_grad = g[:, 3:].abs().sum().item()
            print(f"  stem grad: rgb={rgb_grad:.4e}  hint={hint_grad:.4e}")
            assert rgb_grad > 0, "stem RGB grad is zero"
            assert hint_grad > 0, "stem hint-slice grad is zero"
            break
    else:
        raise RuntimeError("did not find 4-ch stem conv")
    print("rung 0: PASS")


def _data_rung(cfg: TrainConfig, n_clips: int, steps: int) -> None:
    """Shared body for rungs 1 and 2. Overfits `n_clips` clips for `steps`."""
    train_dirs, _ = build_splits(cfg.data_root, val_frac=0.0)
    if not train_dirs:
        raise SystemExit(f"No clips under {cfg.data_root}")
    train_dirs = train_dirs[:n_clips]
    print(f"  overfitting {len(train_dirs)} clip(s) for {steps} steps")

    ds = SealKeyDataset(
        train_dirs, crop_hw=(cfg.crop_h, cfg.crop_w),
        scale_range=(1.0, 1.0),
        k_clips=min(cfg.k_clips, len(train_dirs)),
        max_frames_per_clip=cfg.max_frames_per_clip,
        seed=cfg.seed, augment=False,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0,
                        collate_fn=collate)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = SealKeyNet(pretrained=True).to(device)
    model.train()
    opt = torch.optim.AdamW(model.param_groups(cfg.lr, cfg.encoder_lr_mult),
                            weight_decay=cfg.weight_decay)

    it = iter(loader)
    losses: list[float] = []
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        pred = model(batch["rgb"], batch["hint"])
        total, parts = compose_losses(pred, batch, cfg)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        losses.append(float(parts["loss"]))
        if step % 50 == 0 or step == steps - 1:
            print(f"    step {step:>5d}  loss {losses[-1]:.4f}  "
                  f"α {parts['l_alpha'].item():.4f}  rgb {parts['l_rgb'].item():.4f}")

    start_mean = float(np.mean(losses[:10]))
    end_mean = float(np.mean(losses[-10:]))
    print(f"  mean loss  start={start_mean:.4f}  end={end_mean:.4f}")
    if end_mean >= start_mean * 0.5:
        print("  WARN: loss did not drop >=2x; inspect.")
    else:
        print("  PASS: loss dropped >=2x")


def rung1(cfg: TrainConfig) -> None:
    """Overfit 2 clips (one batch worth). Model-level sanity."""
    cfg.batch_size = 2
    _data_rung(cfg, n_clips=2, steps=2000)


def rung2(cfg: TrainConfig) -> None:
    """Overfit 1 clip with random frames. Dataloader-level sanity."""
    _data_rung(cfg, n_clips=1, steps=5000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rung", type=int, required=True)
    p.add_argument("--data_root", type=Path, default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root

    if args.rung == 0:
        rung0()
    elif args.rung == 1:
        rung1(cfg)
    elif args.rung == 2:
        rung2(cfg)
    else:
        raise SystemExit(f"Unknown rung: {args.rung}")


if __name__ == "__main__":
    main()
