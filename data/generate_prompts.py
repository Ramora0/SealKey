#!/usr/bin/env python3
"""Generate unique image prompts from template.json.

Uses a multiplicative structure:
  - Common: subject × pose/placement
  - Challenge modifiers: subject × modifier (e.g. "a woman with wild frizzy hair")
  - Challenge subjects: special subject × pose (e.g. "a glass of wine on a table")
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
    weights = data["weights"]
    category = rng.choices(
        ["common", "challenge_modifiers", "challenge_subjects"],
        weights=[weights["common"], weights["challenge_modifiers"], weights["challenge_subjects"]],
        k=1,
    )[0]

    if category == "common":
        # Pick subject type, then subject × pose
        type_weights = data["common_type_weights"]
        type_names = list(type_weights.keys())
        chosen_type = rng.choices(type_names, weights=[type_weights[t] for t in type_names], k=1)[0]
        subject = rng.choice(data["subjects"][chosen_type])
        pose = rng.choice(data["poses"][chosen_type])
        return f"{subject}, {pose}"

    elif category == "challenge_modifiers":
        # Pick challenge, then pick a compatible subject × modifier
        challenges = data["challenge_modifiers"]
        names = list(challenges.keys())
        ch_weights = [challenges[n]["weight"] for n in names]
        chosen = rng.choices(names, weights=ch_weights, k=1)[0]
        challenge = challenges[chosen]

        subject_type = rng.choice(challenge["applies_to"])
        subject = rng.choice(data["subjects"][subject_type])
        modifier = rng.choice(challenge["items"])
        return f"{subject} {modifier}"

    else:
        # Challenge subject × pose
        challenges = data["challenge_subjects"]
        names = list(challenges.keys())
        ch_weights = [challenges[n]["weight"] for n in names]
        chosen = rng.choices(names, weights=ch_weights, k=1)[0]
        challenge = challenges[chosen]

        subject = rng.choice(challenge["items"])
        pose = rng.choice(challenge["poses"])
        return f"{subject}, {pose}"


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
