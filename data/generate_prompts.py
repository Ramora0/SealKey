#!/usr/bin/env python3
"""Generate unique image prompts from template.json."""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
TEMPLATE_PATH = DATA_DIR / "template.json"


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def pick_from_tiered(rng: random.Random, tiers: dict[str, dict]) -> str:
    """Pick an item from a dict of {tier_name: {weight, items}} entries."""
    names, weights, item_lists = [], [], []
    for name, tier in tiers.items():
        if "weight" not in tier or "items" not in tier:
            continue
        names.append(name)
        weights.append(tier["weight"])
        item_lists.append(tier["items"])
    chosen_tier = rng.choices(item_lists, weights=weights, k=1)[0]
    return rng.choice(chosen_tier)


def pick_optional(rng: random.Random, prob: float, items: list[str]) -> str:
    """Return a random item with given probability, else empty string."""
    if rng.random() < prob:
        return rng.choice(items)
    return ""


def pick_optional_tiered(rng: random.Random, prob: float, section: dict) -> str:
    """Return a tiered pick with given probability, else empty string."""
    if rng.random() < prob:
        tiers = {k: v for k, v in section.items() if isinstance(v, dict) and "items" in v}
        return pick_from_tiered(rng, tiers)
    return ""


def extract_tiers(section: dict) -> dict[str, dict]:
    """Extract only the tier sub-dicts (those with weight+items) from a section."""
    return {k: v for k, v in section.items() if isinstance(v, dict) and "items" in v}


def generate_prompt(rng: random.Random, data: dict) -> str:
    quality = pick_from_tiered(rng, extract_tiers(data["quality"]))

    medium_raw = pick_optional(rng, data["medium"]["probability"], data["medium"]["items"])
    medium = f" {medium_raw}" if medium_raw else ""

    # Subject — determine category for people-only slots
    subj_tiers = extract_tiers(data["subject"])
    subj_names = list(subj_tiers.keys())
    subj_weights = [subj_tiers[n]["weight"] for n in subj_names]
    chosen_cat = rng.choices(subj_names, weights=subj_weights, k=1)[0]
    subject = rng.choice(subj_tiers[chosen_cat]["items"])
    is_person = chosen_cat == "people"

    # People-only optional slots
    hair = ""
    clothing = ""
    accessory = ""
    if is_person:
        hair_raw = pick_optional_tiered(rng, data["hair"]["probability"], data["hair"])
        hair = f" {hair_raw}" if hair_raw else ""

        clothing_raw = pick_optional_tiered(rng, data["clothing"]["probability"], data["clothing"])
        clothing = f" {clothing_raw}" if clothing_raw else ""

        accessory_raw = pick_optional_tiered(rng, data["accessory"]["probability"], data["accessory"])
        accessory = f" {accessory_raw}" if accessory_raw else ""

    action = pick_from_tiered(rng, extract_tiers(data["action"]))

    prop_raw = pick_optional(rng, data["prop"]["probability"], data["prop"]["items"])
    prop = f" {prop_raw}" if prop_raw else ""

    atmo_raw = pick_optional(rng, data["atmosphere"]["probability"], data["atmosphere"]["items"])
    atmosphere = f" {atmo_raw}" if atmo_raw else ""

    lighting = pick_from_tiered(rng, extract_tiers(data["lighting"]))

    return f"{quality}{medium} of {subject}{hair}{clothing}{accessory}, {action}{prop}{atmosphere}, {lighting}"


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
    max_attempts = args.n * 20  # safety valve

    while len(prompts) < args.n and attempts < max_attempts:
        prompts.add(generate_prompt(rng, data))
        attempts += 1

    if len(prompts) < args.n:
        print(f"Warning: only generated {len(prompts)} unique prompts after {attempts} attempts")

    # Generate in order, preserving random distribution
    prompt_list = list(prompts)
    rng.shuffle(prompt_list)
    with open(args.output, "w") as f:
        for p in prompt_list:
            f.write(p + "\n")

    print(f"Wrote {len(prompt_list)} unique prompts to {args.output}")


if __name__ == "__main__":
    main()
