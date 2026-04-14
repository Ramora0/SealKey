"""Launcher for Wan-Alpha text-to-video (RGBA) inference.

Clones the repo, fetches weights from HuggingFace, builds the gaussian mask
cache, then runs generate_dora_lightx2v_mask.py via torchrun.

Usage:
    python src/try_wan_alpha.py \
        --prompt "This video has a transparent background. A colorful parrot flying. Realistic style." \
        --gpus 1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "wan_alpha_workspace"
REPO_DIR = WORK / "Wan-Alpha"
WEIGHTS_DIR = WORK / "weights"
OUTPUT_DIR = ROOT / "output" / "wan_alpha"

REPO_URL = "https://github.com/WeChatCV/Wan-Alpha.git"

HF_REPOS = {
    "wan21_t2v_14b": ("Wan-AI/Wan2.1-T2V-14B", None),
    "wan_alpha": ("htdong/Wan-Alpha", ["decoder.bin", "epoch-13-1500.safetensors"]),
    "lightx2v": (
        "Kijai/WanVideo_comfy",
        ["Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"],
    ),
}


def run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=cwd, env=env)


def ensure_repo():
    if not REPO_DIR.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    else:
        print(f"Repo already present at {REPO_DIR}")


def install_requirements():
    req = REPO_DIR / "requirements.txt"
    run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    run([sys.executable, "-m", "pip", "install", "huggingface_hub"])


def download_weights():
    from huggingface_hub import snapshot_download, hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (repo_id, files) in HF_REPOS.items():
        local = WEIGHTS_DIR / key
        local.mkdir(parents=True, exist_ok=True)
        if files is None:
            snapshot_download(repo_id=repo_id, local_dir=str(local), local_dir_use_symlinks=False)
        else:
            for f in files:
                hf_hub_download(
                    repo_id=repo_id, filename=f, local_dir=str(local), local_dir_use_symlinks=False
                )
        paths[key] = local
    return paths


def gauss_mask_path() -> Path:
    p = REPO_DIR / "gauss_mask"
    if not p.exists():
        raise FileNotFoundError(f"Expected pre-made mask at {p}; re-clone the repo.")
    return p


def write_prompt_file(prompt: str) -> Path:
    p = WORK / "prompt.txt"
    p.write_text(prompt.strip() + "\n")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default=(
            "This video has a transparent background. Close-up shot. "
            "A colorful parrot flying. Realistic style."
        ),
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--size", default="832*480", help="W*H")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--sample_guide_scale", type=float, default=1.0)
    parser.add_argument("--lora_ratio", type=float, default=1.0)
    parser.add_argument("--alpha_shift_mean", type=float, default=0.05)
    parser.add_argument("--master_port", default="29501")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    ensure_repo()
    if not args.skip_install:
        install_requirements()

    if args.skip_download:
        paths = {k: WEIGHTS_DIR / k for k in HF_REPOS}
    else:
        paths = download_weights()

    wan21 = paths["wan21_t2v_14b"]
    decoder = paths["wan_alpha"] / "decoder.bin"
    lora = paths["wan_alpha"] / "epoch-13-1500.safetensors"
    lightx2v = (
        paths["lightx2v"]
        / "Lightx2v"
        / "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
    )

    mask_cache = gauss_mask_path()
    prompt_file = write_prompt_file(args.prompt)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        "generate_dora_lightx2v_mask.py",
        "--size", args.size,
        "--ckpt_dir", str(wan21),
        "--ulysses_size", str(args.gpus),
        "--vae_lora_checkpoint", str(decoder),
        "--lora_path", str(lora),
        "--lightx2v_path", str(lightx2v),
        "--sample_guide_scale", str(args.sample_guide_scale),
        "--frame_num", str(args.frame_num),
        "--sample_steps", str(args.sample_steps),
        "--lora_ratio", str(args.lora_ratio),
        "--lora_prefix", "",
        "--alpha_shift_mean", str(args.alpha_shift_mean),
        "--cache_path_mask", str(mask_cache),
        "--prompt_file", str(prompt_file),
        "--output_dir", str(OUTPUT_DIR),
    ]
    if args.gpus > 1:
        cmd += ["--dit_fsdp", "--t5_fsdp"]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_DIR))
    run(cmd, cwd=str(REPO_DIR), env=env)
    print(f"\nDone. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
