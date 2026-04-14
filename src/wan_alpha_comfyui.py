"""Launcher for Wan-Alpha text-to-video (RGBA) via ComfyUI.

Usage:
    # Batch generate from a prompt file (starts server automatically)
    python src/wan_alpha_comfyui.py --prompt-file data/prompts.txt --frame-num 81

    # Single prompt
    python src/wan_alpha_comfyui.py \
        --prompt "This video has a transparent background. A colorful parrot flying. Realistic style."

    # Setup only (download models, install nodes, don't launch)
    python src/wan_alpha_comfyui.py --setup-only
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "wan_alpha_workspace"
COMFYUI_DIR = WORK / "ComfyUI"
WORKFLOW_SRC = WORK / "Wan-Alpha" / "Wan-Alpha_v1.0" / "comfyui"
MODELS_DIR = Path("/fs/scratch/PAS2836/lees_stuff/models")
OUTPUT_DIR = Path("/fs/scratch/PAS2836/lees_stuff/sealkey_wan_alpha")

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"

# Model sources: (hf_repo, hf_filename, comfyui_subdir, local_filename)
# The base diffusion model, text encoder, and VAEs come from Comfy-Org's
# repackaged repo; the RGBA LoRA from htdong; LightX2V from Kijai.
MODELS = [
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors",
        "diffusion_models",
        "wan2.1_t2v_14B_fp16.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "text_encoders",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    (
        "htdong/Wan-Alpha_ComfyUI",
        "epoch-13-1500_changed.safetensors",
        "loras",
        "epoch-13-1500_changed.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
        "loras",
        "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/vae/wan_alpha_2.1_vae_rgb_channel.safetensors",
        "vae",
        "wan_alpha_2.1_vae_rgb_channel.safetensors.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/vae/wan_alpha_2.1_vae_alpha_channel.safetensors",
        "vae",
        "wan_alpha_2.1_vae_alpha_channel.safetensors.safetensors",
    ),
]


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.check_call(cmd, **kwargs)


def ensure_comfyui():
    """Clone ComfyUI if not present, install its requirements."""
    if not COMFYUI_DIR.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", COMFYUI_REPO, str(COMFYUI_DIR)])
    else:
        print(f"ComfyUI already present at {COMFYUI_DIR}")

    req = COMFYUI_DIR / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def download_models():
    """Download all required model files into the shared models directory."""
    from huggingface_hub import hf_hub_download

    for repo_id, hf_file, subdir, local_name in MODELS:
        dest_dir = MODELS_DIR / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / local_name

        if dest.exists():
            print(f"Already exists: {dest}")
            continue

        print(f"Downloading {repo_id}/{hf_file} -> {dest}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_file,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place the file in a subdirectory matching the
        # HF path; move it to the expected location if needed.
        downloaded = Path(downloaded)
        if downloaded != dest:
            shutil.move(str(downloaded), str(dest))
            # Clean up any empty parent dirs left behind
            for parent in downloaded.parents:
                if parent == dest_dir:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break


def write_extra_model_paths():
    """Write extra_model_paths.yaml so ComfyUI finds models in the shared dir."""
    yaml_path = COMFYUI_DIR / "extra_model_paths.yaml"
    content = f"""\
wan_alpha:
    base_path: {MODELS_DIR}
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    loras: loras/
    vae: vae/
"""
    yaml_path.write_text(content)
    print(f"Wrote {yaml_path}")


def install_custom_nodes():
    """Install the RGBA save node into ComfyUI's custom_nodes."""
    custom_nodes = COMFYUI_DIR / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)

    node_dir = custom_nodes / "wan_alpha_rgba"
    node_dir.mkdir(parents=True, exist_ok=True)

    src_node = WORKFLOW_SRC / "RGBA_save_tools.py"
    dst_node = node_dir / "RGBA_save_tools.py"
    init_file = node_dir / "__init__.py"

    shutil.copy2(str(src_node), str(dst_node))
    print(f"Copied custom node: {dst_node}")

    # Write __init__.py so ComfyUI discovers the node
    init_file.write_text(
        'from .RGBA_save_tools import NODE_CLASS_MAPPINGS\n'
        'NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}\n'
        '__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]\n'
    )
    print(f"Wrote {init_file}")


def install_workflow():
    """Copy the workflow JSON into ComfyUI's user workflows directory."""
    workflow_dest = COMFYUI_DIR / "user" / "default" / "workflows"
    workflow_dest.mkdir(parents=True, exist_ok=True)

    src = WORKFLOW_SRC / "wan_alpha_t2v_14B.json"
    dst = workflow_dest / "wan_alpha_t2v_14B.json"
    shutil.copy2(str(src), str(dst))
    print(f"Installed workflow: {dst}")

    # Also keep a copy at the workspace root for easy access
    shutil.copy2(str(src), str(WORK / "wan_alpha_t2v_14B.json"))


def build_api_prompt(
    prompt_text: str,
    frame_num: int = 81,
    filename_prefix: str = "wan_alpha",
    seed: int = None,
    workflow_path: Path = None,
) -> dict:
    """Build an API-compatible prompt payload from the workflow JSON."""
    if workflow_path is None:
        workflow_path = WORKFLOW_SRC / "wan_alpha_t2v_14B.json"
    if seed is None:
        seed = random.randint(0, 2**53)

    with open(workflow_path) as f:
        workflow = json.load(f)

    # Convert the workflow to API format (node id -> node config)
    api_prompt = {}
    for node in workflow["nodes"]:
        node_id = str(node["id"])
        entry = {
            "class_type": node["type"],
            "inputs": {},
        }

        widgets = node.get("widgets_values", [])

        if node["type"] == "CLIPLoader" and widgets:
            entry["inputs"]["clip_name"] = widgets[0]
            entry["inputs"]["type"] = widgets[1] if len(widgets) > 1 else "wan"
            if len(widgets) > 2:
                entry["inputs"]["device"] = widgets[2]

        elif node["type"] == "CLIPTextEncode":
            text = widgets[0] if widgets else ""
            if "Positive" in node.get("title", ""):
                text = prompt_text
            entry["inputs"]["text"] = text

        elif node["type"] == "UNETLoader" and widgets:
            entry["inputs"]["unet_name"] = widgets[0]
            entry["inputs"]["weight_dtype"] = widgets[1] if len(widgets) > 1 else "default"

        elif node["type"] == "LoraLoaderModelOnly" and widgets:
            entry["inputs"]["lora_name"] = widgets[0]
            entry["inputs"]["strength_model"] = widgets[1] if len(widgets) > 1 else 1.0

        elif node["type"] == "ModelSamplingSD3" and widgets:
            entry["inputs"]["shift"] = widgets[0]

        elif node["type"] == "EmptyHunyuanLatentVideo" and widgets:
            entry["inputs"]["width"] = widgets[0]
            entry["inputs"]["height"] = widgets[1]
            entry["inputs"]["length"] = frame_num
            entry["inputs"]["batch_size"] = widgets[3] if len(widgets) > 3 else 1

        elif node["type"] == "KSampler" and widgets:
            entry["inputs"]["seed"] = seed
            entry["inputs"]["control_after_generate"] = "fixed"
            entry["inputs"]["steps"] = widgets[2] if len(widgets) > 2 else 4
            entry["inputs"]["cfg"] = widgets[3] if len(widgets) > 3 else 1
            entry["inputs"]["sampler_name"] = widgets[4] if len(widgets) > 4 else "uni_pc"
            entry["inputs"]["scheduler"] = widgets[5] if len(widgets) > 5 else "simple"
            entry["inputs"]["denoise"] = widgets[6] if len(widgets) > 6 else 1

        elif node["type"] == "VAELoader" and widgets:
            entry["inputs"]["vae_name"] = widgets[0]

        elif node["type"] == "VAEDecode":
            pass  # inputs come from links

        elif node["type"] == "SavePNGZIP_and_Preview_RGBA_AnimatedWEBP" and widgets:
            entry["inputs"]["filename_prefix"] = filename_prefix
            entry["inputs"]["fps"] = widgets[1] if len(widgets) > 1 else 16
            entry["inputs"]["lossless"] = widgets[2] if len(widgets) > 2 else True
            entry["inputs"]["quality"] = widgets[3] if len(widgets) > 3 else 80
            entry["inputs"]["method"] = widgets[4] if len(widgets) > 4 else "default"

        # Wire up input links
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                link_id = inp["link"]
                for link in workflow["links"]:
                    if link[0] == link_id:
                        src_node_id = str(link[1])
                        src_slot = link[2]
                        entry["inputs"][inp["name"]] = [src_node_id, src_slot]
                        break

        api_prompt[node_id] = entry

    return {"prompt": api_prompt}


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(port: int = 8188) -> subprocess.Popen:
    """Start the ComfyUI server as a background process and wait until ready."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(port),
        "--output-directory", str(OUTPUT_DIR),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    server = f"127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://{server}/system_stats", timeout=2)
            print(f"ComfyUI server ready at {server} (pid {proc.pid})")
            return proc
        except Exception:
            time.sleep(2)
    proc.kill()
    raise RuntimeError("ComfyUI server failed to start within 4 minutes")


def queue_prompt(payload: dict, server: str = "127.0.0.1:8188") -> str:
    """POST a prompt payload; return the prompt_id."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{server}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    if result.get("node_errors"):
        raise RuntimeError(f"Node errors: {result['node_errors']}")
    return result["prompt_id"]


def wait_for_prompt(prompt_id: str, server: str = "127.0.0.1:8188"):
    """Poll the queue until prompt_id is no longer running or pending."""
    while True:
        with urllib.request.urlopen(f"http://{server}/queue") as resp:
            q = json.loads(resp.read())
        active_ids = {item[1] for item in q["queue_running"]}
        pending_ids = {item[1] for item in q["queue_pending"]}
        if prompt_id not in active_ids and prompt_id not in pending_ids:
            return
        time.sleep(2)


def read_prompts(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from a prompt file."""
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Set up and run Wan-Alpha via ComfyUI"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", help="Single prompt to generate")
    group.add_argument("--prompt-file", type=Path, help="File with one prompt per line")
    group.add_argument(
        "--setup-only", action="store_true",
        help="Only set up ComfyUI, models, and nodes",
    )
    parser.add_argument("--frame-num", type=int, default=None,
                        help="Fixed frame count (4n+1). If omitted, randomizes between 17-49 per prompt.")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    # Setup phase
    if not args.skip_install:
        ensure_comfyui()
    if not args.skip_download:
        download_models()
    write_extra_model_paths()
    install_custom_nodes()
    install_workflow()

    print(f"\n--- Setup complete ---")
    print(f"Models:  {MODELS_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    if args.setup_only:
        return

    # Build prompt list
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        prompts = read_prompts(args.prompt_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        parser.error("Provide --prompt, --prompt-file, or --setup-only")

    # Start server, run all prompts, shut down
    server = f"127.0.0.1:{args.port}"
    proc = start_server(port=args.port)
    try:
        pbar = tqdm(enumerate(prompts), total=len(prompts), unit="prompt")
        for i, prompt_text in pbar:
            prefix = f"wan_alpha_{i:05d}"
            if list(OUTPUT_DIR.glob(f"{prefix}_*.zip")):
                pbar.set_postfix_str("skip (exists)")
                continue
            if args.frame_num is not None:
                frames = args.frame_num
            else:
                # Random 4n+1 value between 17 and 49 (i.e. n in [4..12])
                frames = random.randint(4, 12) * 4 + 1
            pbar.set_postfix_str(f"f={frames} {prompt_text[:60]}")

            payload = build_api_prompt(
                prompt_text,
                frame_num=frames,
                filename_prefix=prefix,
            )
            prompt_id = queue_prompt(payload, server=server)
            wait_for_prompt(prompt_id, server=server)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        print(f"\nServer stopped. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
