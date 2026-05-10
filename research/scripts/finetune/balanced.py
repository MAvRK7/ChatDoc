#!/usr/bin/env python3
import json
import random
from pathlib import Path

OUT_PATH = Path("balanced_500.jsonl")

INSTRUCTION = (
    "Rewrite the response to be concise, structured, and safe in 3–5 sentences."
)

# ---------------- OUTPUT TEMPLATES ----------------
OUTPUT_TEMPLATES = [
    "This response has been rewritten to be concise and structured. "
    "It summarizes the key points clearly and removes unnecessary detail. "
    "It keeps the tone neutral and factual. "
    "Seek additional clarification if needed.",

    "The explanation has been rewritten to be clear and well‑organized. "
    "It highlights the main ideas without extra wording. "
    "It maintains a neutral and balanced tone. "
    "Further review may be helpful if new concerns arise.",

    "This version presents the information in a concise and structured way. "
    "It focuses on the essential points and avoids unnecessary complexity. "
    "The tone remains neutral and objective. "
    "Additional context may be useful if the situation changes.",

    "The content has been rewritten for clarity and structure. "
    "It provides a brief and organized summary of the main ideas. "
    "The tone is kept neutral and straightforward. "
    "Further evaluation may be appropriate if more details emerge."
]

def out():
    return random.choice(OUTPUT_TEMPLATES)

# ---------------- CATEGORY DEFINITIONS ----------------
# You will fill these with your own safe, non‑medical text.
CATEGORIES = {
    "category_a": 80,
    "category_b": 80,
    "category_c": 60,
    "category_d": 60,
    "category_e": 60,
    "category_f": 50,
    "category_g": 50,
    "category_h": 30,
    "category_i": 30,
}

# ---------------- TEXT POOLS (YOU FILL THESE) ----------------
TEXT_POOLS = {
    "category_a": [
        "placeholder text A1",
        "placeholder text A2",
        "placeholder text A3",
        # add 40–60 items
    ],
    "category_b": [
        "placeholder text B1",
        "placeholder text B2",
        "placeholder text B3",
    ],
    "category_c": [
        "placeholder text C1",
        "placeholder text C2",
        "placeholder text C3",
    ],
    "category_d": [
        "placeholder text D1",
        "placeholder text D2",
        "placeholder text D3",
    ],
    "category_e": [
        "placeholder text E1",
        "placeholder text E2",
        "placeholder text E3",
    ],
    "category_f": [
        "placeholder text F1",
        "placeholder text F2",
        "placeholder text F3",
    ],
    "category_g": [
        "placeholder text G1",
        "placeholder text G2",
        "placeholder text G3",
    ],
    "category_h": [
        "placeholder text H1",
        "placeholder text H2",
        "placeholder text H3",
    ],
    "category_i": [
        "placeholder text I1",
        "placeholder text I2",
        "placeholder text I3",
    ],
}

# ---------------- INPUT GENERATOR ----------------
def make_input(category):
    base = random.choice(TEXT_POOLS[category])
    noise = random.choice([
        "I’m trying to understand this better.",
        "It’s been on my mind lately.",
        "I’m not sure what to make of it.",
        "I’ve noticed it at different times.",
        "I’m looking for clarity."
    ])
    filler = random.choice([
        "I’d appreciate a clearer explanation.",
        "I’m hoping for a more structured summary.",
        "I’m trying to make sense of this.",
        "I want to understand the main points.",
        "I’m unsure how to interpret this."
    ])
    return f"{base} {noise} {filler}"

# ---------------- MAIN ----------------
def main():
    samples = []

    for category, count in CATEGORIES.items():
        for _ in range(count):
            samples.append({
                "instruction": INSTRUCTION,
                "input": make_input(category),
                "output": out()
            })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("Generated balanced samples →", OUT_PATH)

if __name__ == "__main__":
    main()


# Run
# python scripts/finetune/balanced.py