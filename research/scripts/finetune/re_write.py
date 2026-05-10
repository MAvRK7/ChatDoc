#!/usr/bin/env python3
import json
import random
from pathlib import Path

OUT_PATH = Path("rewrite_100.jsonl")

INSTRUCTION = (
    "Rewrite the response to be concise, structured, and medically safe in 3–5 sentences."
)

# ---------------- OUTPUT TEMPLATES ----------------
OUTPUT_TEMPLATES = [
    "These symptoms may arise from mild irritation, strain, or temporary inflammation. "
    "Staying hydrated and avoiding known triggers may help reduce discomfort. "
    "Monitoring when the symptoms occur can offer useful clues. "
    "Seek medical care if the symptoms worsen, persist, or interfere with daily activities.",

    "A number of common factors can contribute to these symptoms, including tension, fatigue, "
    "or environmental triggers. Gentle stretching and regular breaks may help. "
    "Tracking symptom patterns can be informative. "
    "Seek medical attention if the symptoms worsen, persist, or are accompanied by concerning changes.",

    "These symptoms can occur due to mild strain, sensitivity, or changes in routine. "
    "Rest, hydration, and avoiding aggravating activities may help. "
    "Observing when the symptoms appear can provide insight. "
    "Seek medical evaluation if the symptoms escalate, persist, or significantly affect daily functioning.",

    "A variety of non-serious factors may contribute to these symptoms, such as posture, stress, "
    "or temporary irritation. Taking breaks and staying hydrated may help ease discomfort. "
    "Monitoring symptom frequency can be useful. "
    "Seek medical care if the symptoms become severe, long-lasting, or are associated with new concerning features."
]

def out():
    return random.choice(OUTPUT_TEMPLATES)

# ---------------- SAFE MESSY INPUT GENERATOR ----------------

BODY_PARTS = [
    "neck", "lower back", "upper back", "shoulder", "abdomen", "chest",
    "left arm", "right arm", "left leg", "right leg", "hip", "jaw",
    "temple", "calf", "thigh", "ankle", "wrist", "elbow"
]

SENSATIONS = [
    "tightness", "pressure", "warmth", "tingling", "numbness", "itching",
    "heaviness", "soreness", "pulsing", "stiffness", "sensitivity",
    "dull ache", "light throbbing", "mild cramping"
]

TRIGGERS = [
    "when I wake up", "after sitting for a long time", "after walking",
    "when I bend forward", "when I turn my head", "after a long day",
    "during light activity", "after carrying things", "when I stretch",
    "in cold weather", "in warm weather"
]

INTENSITY = [
    "slight", "mild", "noticeable", "intermittent", "occasional", "frequent"
]

def generate_symptom():
    part = random.choice(BODY_PARTS)
    sens = random.choice(SENSATIONS)
    trig = random.choice(TRIGGERS)
    intensity = random.choice(INTENSITY)
    return f"I feel {intensity} {sens} in my {part} {trig}."

def generate_messy_input():
    symptom = generate_symptom()
    ramble = random.choice([
        "I’m not sure what’s going on and it’s making me a bit stressed.",
        "It’s been happening on and off and I can’t figure out why.",
        "I tried resting but it still feels strange and I don’t know what to make of it.",
        "It’s been bothering me for a few days and I’m not sure if it’s something serious.",
        "I keep noticing it at random times and it’s confusing me."
    ])
    filler = random.choice([
        "I was hoping someone could explain what might be happening.",
        "I’m just looking for some clarity because it’s been on my mind.",
        "I don’t know if it’s something simple or something I should watch closely.",
        "It’s not stopping me from doing things but it’s definitely noticeable.",
        "I’m not sure if I should be concerned or just wait it out."
    ])
    return f"{symptom} {ramble} {filler}"

# ---------------- MAIN ----------------

def main():
    samples = []

    for _ in range(100):
        messy = generate_messy_input()
        samples.append({
            "instruction": INSTRUCTION,
            "input": messy,
            "output": out()
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("Generated 100 rewrite samples →", OUT_PATH)

if __name__ == "__main__":
    main()

# run
# python scripts/finetune/re_write.py --output data/finetune/train.jsonl --samples 1