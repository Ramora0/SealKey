#!/usr/bin/env python3
"""Generate unique image prompts from template.json.

Produces short, focused prompts matching LayerDiffusion's training style:
"{subject/scenario}, {quality}" — typically 5-15 tokens.

Each prompt targets one edge-case challenge (or is a clean baseline).
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
TEMPLATE_PATH = DATA_DIR / "template.json"


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def generate_prompt(rng: random.Random, data: dict) -> str:
    challenges = data["challenges"]
    names = list(challenges.keys())
    weights = [challenges[n]["weight"] for n in names]

    chosen = rng.choices(names, weights=weights, k=1)[0]
    subject = rng.choice(challenges[chosen]["prompts"])
    quality = rng.choice(data["quality"])

    return f"{subject}, {quality}"


def main():
    parser = argparse.ArgumentParser(description="Generate unique image prompts.")
    parser.add_argument("-n", type=int, default=10_000, help="Number of prompts (default: 10000)")
    parser.add_argument("-o", "--output", type=str, default=str(DATA_DIR / "prompts.txt"),
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = load_template()

    prompts: set[str] = set()
    attempts = 0
    max_attempts = args.n * 20

    while len(prompts) < args.n and attempts < max_attempts:
        prompts.add(generate_prompt(rng, data))
        attempts += 1

    if len(prompts) < args.n:
        print(f"Warning: only generated {len(prompts)} unique prompts after {attempts} attempts")

    prompt_list = list(prompts)
    rng.shuffle(prompt_list)
    with open(args.output, "w") as f:
        for p in prompt_list:
            f.write(p + "\n")

    print(f"Wrote {len(prompt_list)} unique prompts to {args.output}")


if __name__ == "__main__":
    main()
