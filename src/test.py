"""Interactive test UI — sample a random prompt, generate images, and display them.

Usage:
    python -m src.test                          # defaults: batch_size=4, prompts from data/prompts.txt
    python -m src.test --batch-size 2 --steps 20
"""

import argparse
import random
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from src.generate import build_pipeline, load_prompts

pipeline = None
prompts: list[str] = []


def ensure_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = build_pipeline()
    return pipeline


def sample_and_generate(
    batch_size: int,
    steps: int,
    negative_prompt: str,
    width: int,
    height: int,
):
    pipe = ensure_pipeline()
    prompt = random.choice(prompts)

    images = pipe(
        prompt=[prompt] * batch_size,
        negative_prompt=[negative_prompt] * batch_size,
        generator=torch.Generator(device="cuda").manual_seed(random.randint(0, 2**32 - 1)),
        num_images_per_prompt=1,
        num_inference_steps=steps,
        width=width,
        height=height,
        return_dict=False,
    )[0]

    # Create green-background versions to visualize transparency
    green_images = []
    # Alpha mask: white = opaque, black = transparent
    alpha_images = []
    for img in images:
        alpha = img.split()[3]

        green_bg = Image.new("RGBA", img.size, (0, 255, 0, 255))
        green_bg.paste(img, mask=alpha)
        green_images.append(green_bg.convert("RGB"))

        alpha_images.append(alpha.convert("RGB"))

    return images, green_images, alpha_images, prompt


def run_ui(args):
    global prompts
    prompts = load_prompts(args.prompts)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    def on_generate(batch_size, steps, negative_prompt, width, height):
        images, green_images, alpha_images, prompt = sample_and_generate(
            int(batch_size), int(steps), negative_prompt, int(width), int(height),
        )
        return prompt, images, green_images, alpha_images

    with gr.Blocks(title="SealKey — LayerDiffusion Test") as demo:
        gr.Markdown("# SealKey — LayerDiffusion Test\nSample a random prompt and generate transparent images.")

        with gr.Row():
            batch_size = gr.Slider(1, 8, value=args.batch_size, step=1, label="Batch size")
            steps = gr.Slider(5, 50, value=args.steps, step=1, label="Steps")
        with gr.Row():
            width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
            height = gr.Slider(512, 1536, value=1024, step=64, label="Height")

        negative_prompt = gr.Textbox(
            value="blurry, low quality, bad anatomy",
            label="Negative prompt",
        )

        generate_btn = gr.Button("Generate", variant="primary", size="lg")
        prompt_display = gr.Textbox(label="Sampled prompt", interactive=False)
        gallery = gr.Gallery(label="Original (RGBA)", columns=4, height="auto")
        green_gallery = gr.Gallery(label="Green background", columns=4, height="auto")
        alpha_gallery = gr.Gallery(label="Alpha mask (white=opaque, black=transparent)", columns=4, height="auto")

        generate_btn.click(
            fn=on_generate,
            inputs=[batch_size, steps, negative_prompt, width, height],
            outputs=[prompt_display, gallery, green_gallery, alpha_gallery],
        )

    demo.launch(server_name="0.0.0.0", server_port=args.port)


def main():
    parser = argparse.ArgumentParser(description="LayerDiffusion interactive test")
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts.txt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--port", type=int, default=7860)
    run_ui(parser.parse_args())


if __name__ == "__main__":
    main()
