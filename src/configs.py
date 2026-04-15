"""SealKey v1 training config.

Single source of truth for all knobs. train.py auto-generates argparse flags
from this dataclass via dataclasses.fields().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # Data
    data_root: Path = Path("/fs/scratch/PAS2836/lees_stuff/sealkey_preprocessed")
    crop_h: int = 480
    crop_w: int = 832
    scale_min: float = 0.75
    scale_max: float = 1.5
    val_frac: float = 0.05
    max_frames_per_clip: int = 96  # cap decoded frames per clip (memory)

    # Dataloader
    batch_size: int = 8
    num_workers: int = 8
    k_clips: int = 4

    # Optim
    lr: float = 1e-4
    encoder_lr_mult: float = 0.5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    total_steps: int = 200_000
    warmup_steps: int = 1_000

    # Losses
    alpha_loss: str = "l1"  # "bce" | "l1"
    w_alpha: float = 1.0
    w_rgb: float = 1.0
    delta_reg: float = 0.01

    # Logging / checkpointing
    val_every: int = 2_000
    ckpt_every: int = 5_000
    img_log_every: int = 500
    val_samples: int = 200

    # Runtime
    bf16: bool = True
    seed: int = 0
    out_dir: Path = Path("runs/v1")
    device: str = "cuda"

    # Smoke-test rung (0–5); 0 = normal training, see smoke_overfit.py
    rung: int = 0
