#!/usr/bin/env python3
"""Generate unique video prompts from template.json.

Output format: plain text, one prompt per line. Frame length is randomized
downstream at video generation time.

Each prompt targets a matting challenge (fine hair in motion, sheer fabric,
fluids, transparent objects, smoke/mist, fuzzy motion, color contamination,
light hair) or is a common baseline subject.
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


def build_scene(rng: random.Random, challenge: dict) -> str:
    if challenge["type"] == "scene_list":
        return rng.choice(challenge["scenes"])

    fmt = challenge["format"]
    fields = {}
    if "{subject}" in fmt:
        fields["subject"] = rng.choice(challenge["subjects"])
    if "{modifier}" in fmt:
        fields["modifier"] = rng.choice(challenge["modifiers"])
    if "{motion}" in fmt:
        fields["motion"] = rng.choice(challenge["motions"])
    return fmt.format(**fields)


def generate_prompt(rng: random.Random, data: dict) -> str:
    challenges = data["challenges"]
    names = list(challenges.keys())
    weights = [challenges[n]["weight"] for n in names]

    chosen = rng.choices(names, weights=weights, k=1)[0]
    challenge = challenges[chosen]

    shot_type_key = rng.choice(challenge["shot_types"])
    shot_type = rng.choice(data["shot_types"][shot_type_key])

    angle = ""
    angles = data.get("angles")
    if angles and rng.random() < angles["probability"]:
        angle = rng.choice(angles["items"])

    scene = build_scene(rng, challenge)
    return data["template"].format(shot_type=shot_type, angle=angle, scene=scene)


def main():
    parser = argparse.ArgumentParser(description="Generate unique video prompts.")
    parser.add_argument("-n", type=int, default=1000, help="Number of prompts (default: 1000)")
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
