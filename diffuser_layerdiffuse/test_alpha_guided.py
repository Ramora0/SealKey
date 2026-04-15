"""
Loss-guided LayerDiffuse sampling: condition on a target alpha image and let the
model fill in the RGB.

At each denoise step:
  1. Predict x0 from x_t (Tweedie).
  2. Decode x0 through the transparent VAE to get predicted RGBA.
  3. Compute alpha loss vs. the user-provided target alpha.
  4. Backprop into x_t and nudge it before the scheduler step.

Usage:
    python test_alpha_guided.py --alpha path/to/alpha.png --prompt "a cute corgi"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from layer_diffuse.models import TransparentVAEDecoder


def load_alpha(path: str, target_hw: tuple[int, int]) -> torch.Tensor:
    """Load alpha image as [1,1,H,W] float in [0,1], resized to target_hw."""
    img = Image.open(path)
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1])
    elif img.mode == "LA":
        alpha = np.array(img.split()[-1])
    elif img.mode in ("L", "I"):
        alpha = np.array(img.convert("L"))
    else:
        alpha = np.array(img.convert("L"))
    alpha = alpha.astype(np.float32) / 255.0
    t = torch.from_numpy(alpha)[None, None]
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.clamp(0, 1)


@torch.no_grad()
def encode_prompt_sdxl(pipeline, prompt: str, negative_prompt: str, device):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def decode_alpha_only(vae: TransparentVAEDecoder, latents: torch.Tensor) -> torch.Tensor:
    """
    Differentiable-ish alpha decode. Mirrors TransparentVAEDecoder.decode but only
    runs a single (non-augmented) pass to keep gradients tractable.
    Returns alpha in [0,1] shape [B,1,H,W].
    """
    z = latents / vae.config.scaling_factor
    pixel = vae.decoder(vae.post_quant_conv(z))
    pixel = pixel / 2 + 0.5
    y = vae.transparent_decoder(pixel, z)  # [B,4,H,W] : alpha + fg
    y = y.clamp(0, 1)
    alpha = y[:, :1]
    return alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True, help="Path to target alpha image")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--out", default="alpha_guided_out.png")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--guidance-scale", type=float, default=200.0,
                    help="Strength of alpha-loss guidance (tune per prompt).")
    ap.add_argument("--guidance-start", type=float, default=0.1,
                    help="Fraction of schedule before which to skip guidance (too noisy).")
    ap.add_argument("--guidance-end", type=float, default=0.9,
                    help="Fraction of schedule after which to skip guidance (already locked).")
    ap.add_argument("--guidance-stride", type=int, default=2,
                    help="Apply guidance every Nth step.")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alpha-blur-sigma", type=float, default=0.0,
                    help="Gaussian blur sigma applied to the target alpha (softens hard edges).")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float16

    print("Loading transparent VAE...")
    transparent_vae = TransparentVAEDecoder.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
    )
    transparent_vae.config.force_upcast = False
    decoder_path = hf_hub_download(
        "LayerDiffusion/layerdiffusion-v1", "vae_transparent_decoder.safetensors"
    )
    transparent_vae.set_transparent_decoder(load_file(decoder_path))

    print("Loading SDXL pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=transparent_vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)
    pipeline.load_lora_weights(
        "rootonchair/diffuser_layerdiffuse",
        weight_name="diffuser_layer_xl_transparent_attn.safetensors",
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    unet = pipeline.unet
    scheduler = pipeline.scheduler

    print("Loading target alpha...")
    target_alpha = load_alpha(args.alpha, (args.height, args.width)).to(device, dtype=torch.float32)
    if args.alpha_blur_sigma > 0:
        k = max(3, int(args.alpha_blur_sigma * 6) | 1)
        coords = torch.arange(k, device=device, dtype=torch.float32) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * args.alpha_blur_sigma ** 2))
        g = (g / g.sum()).view(1, 1, 1, k)
        target_alpha = F.conv2d(target_alpha, g, padding=(0, k // 2))
        target_alpha = F.conv2d(target_alpha, g.transpose(-1, -2), padding=(k // 2, 0))

    seed = args.seed if args.seed is not None else torch.randint(0, 1_000_000, (1,)).item()
    print(f"Seed: {seed}")
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Encoding prompt...")
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = encode_prompt_sdxl(pipeline, args.prompt, args.negative_prompt, device)

    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    add_time_ids = pipeline._get_add_time_ids(
        (args.height, args.width),
        (0, 0),
        (args.height, args.width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
    ).to(device)
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    added_cond_kwargs = {"text_embeds": pooled_cfg, "time_ids": add_time_ids}

    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps

    latent_h = args.height // 8
    latent_w = args.width // 8
    latents = torch.randn(
        (1, 4, latent_h, latent_w), generator=generator, device=device, dtype=dtype
    )
    latents = latents * scheduler.init_noise_sigma

    print(f"Sampling {args.steps} steps with alpha guidance...")
    n_steps = len(timesteps)
    for i, t in enumerate(timesteps):
        progress = i / max(1, n_steps - 1)
        do_guidance = (
            args.guidance_start <= progress <= args.guidance_end
            and (i % args.guidance_stride == 0)
            and args.guidance_scale > 0
        )

        if do_guidance:
            latents = latents.detach().requires_grad_(True)

        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)

        with torch.set_grad_enabled(do_guidance):
            noise_pred = unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + args.cfg * (noise_text - noise_uncond)

            if do_guidance:
                alpha_prod_t = scheduler.alphas_cumprod.to(device)[t]
                beta_prod_t = 1 - alpha_prod_t
                x0_hat = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

                pred_alpha = decode_alpha_only(transparent_vae, x0_hat.to(dtype))
                pred_alpha = pred_alpha.float()
                loss = F.l1_loss(pred_alpha, target_alpha)
                grad = torch.autograd.grad(loss, latents)[0]
                grad_norm = grad.norm().clamp(min=1e-8)
                normalized = grad / grad_norm
                with torch.no_grad():
                    latents = latents - args.guidance_scale * normalized
                print(f"  step {i:02d} t={int(t):4d} alpha_loss={loss.item():.4f}")
                latents = latents.detach()

        with torch.no_grad():
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    print("Final decode (full augmented transparent decoder)...")
    with torch.no_grad():
        latents = latents.to(dtype)
        decoded = transparent_vae.decode(latents / transparent_vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        rgba = decoded[0].permute(1, 2, 0).cpu().float().numpy()

    rgb = rgba[..., :3]
    alpha_out = rgba[..., 3]
    rgba_u8 = (np.concatenate([rgb, alpha_out[..., None]], axis=-1) * 255).astype(np.uint8)
    Image.fromarray(rgba_u8, mode="RGBA").save(args.out)

    target_u8 = (target_alpha[0, 0].cpu().numpy() * 255).astype(np.uint8)
    pred_u8 = (alpha_out * 255).astype(np.uint8)
    side_by_side = np.concatenate([target_u8, pred_u8], axis=1)
    cmp_path = Path(args.out).with_name(Path(args.out).stem + "_alpha_cmp.png")
    Image.fromarray(side_by_side, mode="L").save(cmp_path)

    print(f"Saved {args.out} and {cmp_path}")
    print(f"Final alpha L1 vs target: {np.abs(alpha_out - target_alpha[0,0].cpu().numpy()).mean():.4f}")


if __name__ == "__main__":
    main()
