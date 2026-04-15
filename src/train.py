"""SealKey v1 training entrypoint.

Plain PyTorch. Argparse auto-generated from TrainConfig. TensorBoard logging.

    python -m src.train --data_root /path/to/preprocessed --out_dir runs/v1
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.configs import TrainConfig
from src.dataset import SealKeyDataset, build_splits, collate
from src.losses import compose_losses
from src.metrics import edge_weighted_l1, grad, mse, sad
from src.model import SealKeyNet
from src.viz import make_panel


# ---------------------------------------------------------------------------
# Argparse <-> dataclass
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        default = getattr(defaults, f.name)
        ty = f.type if not isinstance(f.type, str) else eval(f.type)  # noqa: S307
        if ty is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=default)
        elif ty is Path:
            p.add_argument(f"--{f.name}", type=Path, default=default)
        else:
            p.add_argument(f"--{f.name}", type=ty, default=default)
    return p


def parse_config() -> TrainConfig:
    args = _build_argparser().parse_args()
    return TrainConfig(**vars(args))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def compute_total_steps(cfg: TrainConfig, n_train_samples: int,
                        frames_per_clip: float | None = None) -> int:
    """One epoch ≈ n_train_samples * frames_per_clip / batch_size steps.

    Uses `max_frames_per_clip` as a conservative upper bound when the actual
    average is unknown; once training has seen enough samples to estimate,
    pass the observed average via `frames_per_clip` to tighten the estimate.
    """
    fpc = frames_per_clip if frames_per_clip is not None else float(cfg.max_frames_per_clip)
    steps_per_epoch = max(
        1,
        int((n_train_samples * fpc + cfg.batch_size - 1) // cfg.batch_size),
    )
    return cfg.epochs * steps_per_epoch


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    t = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    t = min(1.0, max(0.0, t))
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * t))


def set_lr(opt: torch.optim.Optimizer, base_lr: float, encoder_lr_mult: float) -> None:
    opt.param_groups[0]["lr"] = base_lr * encoder_lr_mult   # encoder
    opt.param_groups[1]["lr"] = base_lr                     # rest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: SealKeyNet, loader: DataLoader, cfg: TrainConfig,
             max_samples: int) -> tuple[dict[str, float], np.ndarray | None]:
    model.eval()
    by_mode: dict[str, dict[str, list[float]]] = {}
    seen = 0
    first_panel: np.ndarray | None = None
    pbar = tqdm(total=max_samples, desc="val", leave=False, unit="samp")
    for batch in loader:
        batch = _to_device(batch, cfg.device)
        with torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.bf16):
            pred = model(batch["rgb"], batch["hint"])
        if first_panel is None:
            first_panel = make_panel(batch, pred, n=2)
        for i, mode in enumerate(batch["mode"]):
            d = by_mode.setdefault(mode, {"sad": [], "mse": [], "grad": [], "rgb": []})
            d["sad"].append(sad(pred["alpha_pred"][i, 0], batch["gt_alpha"][i, 0]))
            d["mse"].append(mse(pred["alpha_pred"][i, 0], batch["gt_alpha"][i, 0]))
            d["grad"].append(grad(pred["alpha_pred"][i, 0], batch["gt_alpha"][i, 0]))
            d["rgb"].append(edge_weighted_l1(
                pred["rgb_pred"][i:i + 1], batch["gt_rgb"][i:i + 1],
                batch["gt_alpha"][i:i + 1],
            ))
        bs = batch["rgb"].shape[0]
        seen += bs
        pbar.update(bs)
        if seen >= max_samples:
            break
    pbar.close()
    model.train()

    out: dict[str, float] = {}
    for mode, d in by_mode.items():
        for k, v in d.items():
            if v:
                out[f"{mode}/{k}"] = float(np.mean(v))
    # Overall primary score: unweighted mean across modes.
    if out:
        all_sad = [v for k, v in out.items() if k.endswith("/sad")]
        out["val/sad_mean"] = float(np.mean(all_sad))
    return out, first_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: str) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
        for k, v in batch.items()
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_ckpt(path: Path, step: int, model: SealKeyNet, opt: torch.optim.Optimizer,
               cfg: TrainConfig, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
        "cfg": asdict(cfg), **extra,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_config()
    _seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs, val_dirs = build_splits(cfg.data_root, val_frac=cfg.val_frac)
    print(f"train: {len(train_dirs)} samples  val: {len(val_dirs)} samples")
    if not train_dirs:
        raise SystemExit(f"No samples under {cfg.data_root}")

    total_steps = compute_total_steps(cfg, len(train_dirs))
    steps_per_epoch = total_steps // cfg.epochs
    print(f"schedule: {cfg.epochs} epochs × {steps_per_epoch} steps/epoch "
          f"= {total_steps} total steps (batch_size={cfg.batch_size})")

    train_ds = SealKeyDataset(
        train_dirs, crop_hw=(cfg.crop_h, cfg.crop_w),
        scale_range=(cfg.scale_min, cfg.scale_max),
        k_clips=cfg.k_clips, max_frames_per_clip=cfg.max_frames_per_clip,
        seed=cfg.seed, augment=True,
    )
    val_ds = SealKeyDataset(
        val_dirs or train_dirs[: max(1, len(train_dirs) // 20)],
        crop_hw=(cfg.crop_h, cfg.crop_w), scale_range=(1.0, 1.0),
        k_clips=min(cfg.k_clips, 2), max_frames_per_clip=cfg.max_frames_per_clip,
        seed=cfg.seed + 1, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        collate_fn=collate, persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=min(2, cfg.num_workers),
        collate_fn=collate, pin_memory=True,
    )

    model = SealKeyNet(pretrained=True).to(cfg.device)
    opt = torch.optim.AdamW(
        model.param_groups(cfg.lr, cfg.encoder_lr_mult),
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.wandb_run_name or None,
        mode=cfg.wandb_mode,
        dir=str(cfg.out_dir),
        config=asdict(cfg),
    )
    best_val = float("inf")
    t0 = time.time()

    model.train()
    step = 0
    it = iter(train_loader)
    pbar = tqdm(total=total_steps, desc="train", unit="step", dynamic_ncols=True)
    # Rolling frames/clip estimate: weighted mean with w = 1/clip_len gives
    # the per-clip (not per-frame) average, matching our steps formula.
    # We update total_steps (and therefore the LR schedule horizon) early,
    # then freeze once the estimate stabilizes — avoids jerking the cosine
    # schedule late in training.
    running_inv_sum = 0.0
    running_samples = 0
    est_refresh_every = 200
    est_freeze_at = 2000
    est_frozen = False
    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        clip_lens = batch["clip_len"]  # CPU int32 tensor
        running_inv_sum += float((1.0 / clip_lens.float()).sum().item())
        running_samples += int(clip_lens.numel())
        batch = _to_device(batch, cfg.device)

        set_lr(opt, lr_at(step, total_steps, cfg), cfg.encoder_lr_mult)

        with torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.bf16):
            pred = model(batch["rgb"], batch["hint"])
            total, parts = compose_losses(pred, batch, cfg)

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 20 == 0:
            pbar.set_postfix(
                loss=f"{parts['loss'].item():.4f}",
                α=f"{parts['l_alpha'].item():.4f}",
                rgb=f"{parts['l_rgb'].item():.4f}",
                δ=f"{parts['l_delta'].item():.4f}",
            )
            log = {f"train/{k}": v.item() for k, v in parts.items()}
            log["train/lr"] = opt.param_groups[1]["lr"]
            wandb.log(log, step=step)

        if step > 0 and step % cfg.img_log_every == 0:
            panel = make_panel(batch, pred, n=2)
            wandb.log({"train/panel": wandb.Image(panel)}, step=step)

        if step > 0 and step % cfg.val_every == 0:
            vmetrics, val_panel = validate(model, val_loader, cfg, cfg.val_samples)
            log = {f"val/{k}": v for k, v in vmetrics.items()}
            if val_panel is not None:
                log["val/panel"] = wandb.Image(val_panel)
            wandb.log(log, step=step)
            primary = vmetrics.get("val/sad_mean", float("inf"))
            tqdm.write(f"[val @ step {step}] {vmetrics}")
            if primary < best_val:
                best_val = primary
                _save_ckpt(cfg.out_dir / "ckpt" / "best.pt", step, model, opt, cfg,
                           {"val": vmetrics})

        if step > 0 and step % cfg.ckpt_every == 0:
            _save_ckpt(cfg.out_dir / "ckpt" / "last.pt", step, model, opt, cfg, {})

        step += 1
        pbar.update(1)

        if (not est_frozen and step % est_refresh_every == 0
                and running_inv_sum > 0):
            avg_fpc = running_samples / running_inv_sum
            new_total = compute_total_steps(cfg, len(train_dirs), avg_fpc)
            if new_total != total_steps:
                total_steps = new_total
                pbar.total = new_total
                pbar.refresh()
            if step >= est_freeze_at:
                est_frozen = True
                tqdm.write(
                    f"[schedule] frozen: avg_frames/clip≈{avg_fpc:.1f} → "
                    f"total_steps={total_steps}"
                )

    pbar.close()
    _save_ckpt(cfg.out_dir / "ckpt" / "last.pt", step, model, opt, cfg, {})
    wandb.finish()
    print("done.")


if __name__ == "__main__":
    main()
