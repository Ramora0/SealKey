"""Batch transparent image generation using LayerDiffusion (SDXL)."""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the diffuser_layerdiffuse repo to sys.path so layer_diffuse is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffuser_layerdiffuse"))

os.environ.setdefault("HF_HOME", "/fs/scratch/PAS2836/lees_stuff/hf_cache")

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def build_pipeline(device: str = "cuda") -> StableDiffusionXLPipeline:
    # Import the transparent VAE decoder from the layerdiffuse package.
    # This requires the diffuser_layerdiffuse repo to be cloned and on sys.path
    # (or installed as a package). See README for setup instructions.
    from layer_diffuse.models import TransparentVAEDecoder

    print("Loading transparent VAE decoder...")
    transparent_vae = TransparentVAEDecoder.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    transparent_vae.config.force_upcast = False

    vae_weights = hf_hub_download(
        "LayerDiffusion/layerdiffusion-v1",
        "vae_transparent_decoder.safetensors",
    )
    transparent_vae.set_transparent_decoder(load_file(vae_weights))

    print("Loading SDXL pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=transparent_vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)

    print("Loading LayerDiffusion LoRA weights...")
    pipeline.load_lora_weights(
        "rootonchair/diffuser_layerdiffuse",
        weight_name="diffuser_layer_xl_transparent_attn.safetensors",
    )

    return pipeline


def generate_batch(
    pipeline: StableDiffusionXLPipeline,
    prompts: list[str],
    output_dir: Path,
    *,
    negative_prompt: str = "blurry, low quality, bad anatomy",
    steps: int = 25,
    batch_size: int = 4,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch = prompts[i : i + batch_size]
        images = pipeline(
            prompt=batch,
            negative_prompt=[negative_prompt] * len(batch),
            generator=generator,
            num_images_per_prompt=1,
            num_inference_steps=steps,
            width=width,
            height=height,
            return_dict=False,
        )[0]

        for j, img in enumerate(images):
            idx = i + j
            out_path = output_dir / f"{idx:04d}.png"
            img.save(out_path)
            saved.append(out_path)
            tqdm.write(f"  Saved {out_path.name}: {prompts[idx][:60]}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="LayerDiffusion batch generator")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts.txt"),
        help="Path to prompts file (one per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for generated images",
    )
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Images per batch (4-6 safe on A100 40GB)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, bad anatomy",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"No prompts found in {args.prompts}")
        return

    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    pipeline = build_pipeline()

    t0 = time.time()
    saved = generate_batch(
        pipeline,
        prompts,
        args.output,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    elapsed = time.time() - t0

    print(f"\nDone — {len(saved)} images in {elapsed:.1f}s ({elapsed/len(saved):.1f}s each)")


if __name__ == "__main__":
    main()
