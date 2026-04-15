"""SealKey v1 model: ConvNeXt-Small encoder + hand-rolled U-Net decoder.

Input  : 4-channel [B, 4, H, W] = RGB (ImageNet-normalized) + hint ([0,1] raw).
Output : residual RGB delta + alpha logits. Forward also returns the applied
         composites so the training loop never touches denorm math.

Fully convolutional. No learned positional information. Arbitrary inference
resolution. Encoder is timm ConvNeXt-Small with the stem conv patched from
3→4 input channels (pretrained RGB weights preserved, hint slice zero-init).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_stem_conv(encoder: nn.Module) -> tuple[nn.Module, str, nn.Conv2d]:
    """Locate the first Conv2d in a timm ConvNeXt encoder and return
    (parent_module, attribute_name, conv) so we can replace it in place."""
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Conv2d):
            parent_name, _, leaf = name.rpartition(".")
            parent = encoder.get_submodule(parent_name) if parent_name else encoder
            return parent, leaf, mod
    raise RuntimeError("No Conv2d found in encoder")


def _patch_stem_to_4ch(encoder: nn.Module) -> None:
    """Replace the 3-channel stem conv with a 4-channel one. Copies pretrained
    RGB weights; zero-inits the hint slice so initial behavior equals
    RGB-only pretrained."""
    parent, attr, old = _find_stem_conv(encoder)
    assert old.in_channels == 3, f"expected 3-ch stem, got {old.in_channels}"
    new = nn.Conv2d(
        in_channels=4,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        new.weight[:, :3].copy_(old.weight)
        new.weight[:, 3:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, attr, new)


class UpBlock(nn.Module):
    """Bilinear upsample ×2, optional concat with a skip tensor, two 3×3 convs
    with GroupNorm + GELU. No learned upsampling — keeps everything FCN."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.skip_ch = skip_ch
        mid_in = in_ch + skip_ch
        self.conv1 = nn.Conv2d(mid_in, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class SealKeyNet(nn.Module):
    """4-channel in → 4-channel out. Encoder: ConvNeXt-Small (4 stages).
    Decoder: 5 UpBlocks taking 1/32 → 1/1 via skips at 1/16, 1/8, 1/4."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_small.fb_in22k_ft_in1k",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=3,
        )
        _patch_stem_to_4ch(self.encoder)

        # ConvNeXt-Small feature channels at strides 4/8/16/32.
        c0, c1, c2, c3 = 96, 192, 384, 768
        self.up3 = UpBlock(c3, c2, 384)        # 1/32 → 1/16, skip c2
        self.up2 = UpBlock(384, c1, 192)       # 1/16 → 1/8,  skip c1
        self.up1 = UpBlock(192, c0, 96)        # 1/8  → 1/4,  skip c0
        self.up0 = UpBlock(96, 0, 48)          # 1/4  → 1/2
        self.up_full = UpBlock(48, 0, 32)      # 1/2  → 1/1
        self.head = nn.Conv2d(32, 4, kernel_size=1)

        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        return (rgb - self.imagenet_mean) / self.imagenet_std

    def forward(self, rgb: torch.Tensor, hint: torch.Tensor) -> dict[str, torch.Tensor]:
        """rgb: [B,3,H,W] in [0,1]. hint: [B,1,H,W] in [0,1].
        Returns dict with rgb_pred, alpha_pred, delta, alpha_logit."""
        rgb_n = self.normalize(rgb)
        x = torch.cat([rgb_n, hint], dim=1)
        f0, f1, f2, f3 = self.encoder(x)
        d = self.up3(f3, f2)
        d = self.up2(d, f1)
        d = self.up1(d, f0)
        d = self.up0(d)
        d = self.up_full(d)
        out = self.head(d)
        # Encoder strides may not divide H,W exactly. Resample to input shape.
        if out.shape[-2:] != rgb.shape[-2:]:
            out = F.interpolate(out, size=rgb.shape[-2:], mode="bilinear",
                                align_corners=False)
        delta = torch.tanh(out[:, :3]) * 0.5
        alpha_logit = out[:, 3:4]
        rgb_pred = (rgb + delta).clamp(0, 1)
        alpha_pred = torch.sigmoid(alpha_logit)
        return {
            "rgb_pred": rgb_pred,
            "alpha_pred": alpha_pred,
            "delta": delta,
            "alpha_logit": alpha_logit,
        }

    def param_groups(self, lr: float, encoder_lr_mult: float) -> list[dict]:
        enc_params = list(self.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params, "lr": lr * encoder_lr_mult},
            {"params": other, "lr": lr},
        ]
