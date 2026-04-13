#!/usr/bin/env python3
"""Generate unique image prompts from template.json.

Each prompt has a skeleton (quality + subject + action + lighting) and at most
one focus element (hair, clothing, accessory, prop, atmosphere, or medium).
This keeps prompts simple enough for the model to actually render faithfully.
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
TEMPLATE_PATH = DATA_DIR / "template.json"

# Focus slot weights control how often each optional element is chosen as
# the single interesting thing in a prompt. None means no focus (bare skeleton).
FOCUS_WEIGHTS: dict[str | None, int] = {
    None: 20,         # bare skeleton — just subject + action + lighting
    "hair": 25,       # people only
    "clothing": 20,   # people only
    "accessory": 10,  # people only
    "medium": 5,
    "prop": 10,
    "atmosphere": 10,
}


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def pick_from_tiered(rng: random.Random, tiers: dict[str, dict]) -> str:
    """Pick an item from a dict of {tier_name: {weight, items}} entries."""
    weights, item_lists = [], []
    for tier in tiers.values():
        if "weight" not in tier or "items" not in tier:
            continue
        weights.append(tier["weight"])
        item_lists.append(tier["items"])
    chosen_tier = rng.choices(item_lists, weights=weights, k=1)[0]
    return rng.choice(chosen_tier)


def extract_tiers(section: dict) -> dict[str, dict]:
    """Extract only the tier sub-dicts (those with weight+items) from a section."""
    return {k: v for k, v in section.items() if isinstance(v, dict) and "items" in v}


def generate_prompt(rng: random.Random, data: dict) -> str:
    # --- skeleton (always present) ---
    quality = pick_from_tiered(rng, extract_tiers(data["quality"]))

    subj_tiers = extract_tiers(data["subject"])
    subj_names = list(subj_tiers.keys())
    subj_weights = [subj_tiers[n]["weight"] for n in subj_names]
    chosen_cat = rng.choices(subj_names, weights=subj_weights, k=1)[0]
    subject = rng.choice(subj_tiers[chosen_cat]["items"])
    is_person = chosen_cat == "people"

    action_key = {"people": "action", "animals": "action_animal", "objects": "action_object"}[chosen_cat]
    action = pick_from_tiered(rng, extract_tiers(data[action_key]))

    lighting = pick_from_tiered(rng, extract_tiers(data["lighting"]))

    # --- pick one focus element ---
    # For non-people, people-only slots are excluded from the lottery
    people_only = {"hair", "clothing", "accessory"}
    candidates = {k: v for k, v in FOCUS_WEIGHTS.items()
                  if k is None or (is_person or k not in people_only)}
    focus = rng.choices(list(candidates.keys()), weights=list(candidates.values()), k=1)[0]

    medium = ""
    hair = ""
    clothing = ""
    accessory = ""
    prop = ""
    atmosphere = ""

    if focus == "medium":
        medium = f" {rng.choice(data['medium']['items'])}"
    elif focus == "hair":
        hair = f" {pick_from_tiered(rng, extract_tiers(data['hair']))}"
    elif focus == "clothing":
        clothing = f" {pick_from_tiered(rng, extract_tiers(data['clothing']))}"
    elif focus == "accessory":
        accessory = f" {pick_from_tiered(rng, extract_tiers(data['accessory']))}"
    elif focus == "prop":
        prop = f" {rng.choice(data['prop']['items'])}"
    elif focus == "atmosphere":
        atmosphere = f" {rng.choice(data['atmosphere']['items'])}"

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
