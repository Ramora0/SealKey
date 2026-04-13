#!/usr/bin/env bash
# Run this on the A100 machine to set up the environment.
set -euo pipefail

echo "=== Setting up LayerDiffusion environment ==="

# Clone the diffusers port of LayerDiffusion (provides layer_diffuse package)
if [ ! -d "diffuser_layerdiffuse" ]; then
    git clone https://github.com/rootonchair/diffuser_layerdiffuse.git
    echo "Cloned diffuser_layerdiffuse"
else
    echo "diffuser_layerdiffuse already exists, skipping clone"
fi

# Install project dependencies
uv pip install -e .

# Install the layer_diffuse package from the cloned repo
uv pip install -e diffuser_layerdiffuse/

echo ""
echo "=== Setup complete ==="
echo "Run:  python -m src.generate --prompts data/prompts.txt --output output/"
