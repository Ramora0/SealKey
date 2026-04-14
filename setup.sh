#!/usr/bin/env bash
# Run this on the A100 machine to set up the environment.
set -euo pipefail

echo "=== Setting up LayerDiffusion environment ==="

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Created venv"
fi
source venv/bin/activate

# Clone the diffusers port of LayerDiffusion (provides layer_diffuse package)
if [ ! -d "diffuser_layerdiffuse" ]; then
    git clone https://github.com/rootonchair/diffuser_layerdiffuse.git
    echo "Cloned diffuser_layerdiffuse"
else
    echo "diffuser_layerdiffuse already exists, skipping clone"
fi

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -e .

# Install requirements from the layerdiffuse repo
pip install -r diffuser_layerdiffuse/requirements.txt

# Warn if ffmpeg is missing or older than 6.0. Preprocess stores codec output
# as MP4 directly; decode-time behavior can vary across ffmpeg versions, so
# pin the build you train against if you care about bit-exact reproducibility.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found on PATH. src/preprocess.py and src/augment.py require it."
else
    ffmpeg_major=$(ffmpeg -version 2>&1 | head -n1 | sed -E 's/^ffmpeg version ([0-9]+).*/\1/' | grep -E '^[0-9]+$' || echo 0)
    if [ "${ffmpeg_major:-0}" -lt 6 ]; then
        echo "WARNING: ffmpeg < 6.0 detected. Newer codec options may behave differently; upgrade if possible."
    fi
fi

echo ""
echo "=== Setup complete ==="
echo "Run:  python -m src.generate --prompts data/prompts.txt --output output/"
