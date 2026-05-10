#!/usr/bin/env python3
import json
import random
from pathlib import Path

OUT_PATH = Path("next_500.jsonl")

INSTRUCTION = (
    "Give a short, structured medical explanation in 3–5 sentences. "
    "Include likely causes, simple advice, and when to seek medical care."
)

# ---------------- OUTPUT TEMPLATES ----------------
OUTPUT_TEMPLATES = [
    "These symptoms may arise from mild irritation, strain, or temporary inflammation. Staying hydrated and avoiding known triggers may help reduce discomfort. Monitoring when the symptoms occur can offer useful clues. Seek medical care if the symptoms worsen, persist, or interfere with daily activities.",
    "A number of common factors can contribute to these symptoms, including tension, fatigue, or environmental triggers. Gentle stretching and regular breaks may help. Tracking symptom patterns can be informative. Seek medical attention if the symptoms worsen, persist, or are accompanied by concerning changes.",
    "These symptoms can occur due to mild strain, sensitivity, or changes in routine. Rest, hydration, and avoiding aggravating activities may help. Observing when the symptoms appear can provide insight. Seek medical evaluation if the symptoms escalate, persist, or significantly affect daily functioning.",
    "A variety of non-serious factors may contribute to these symptoms, such as posture, stress, or temporary irritation. Taking breaks and staying hydrated may help ease discomfort. Monitoring symptom frequency can be useful. Seek medical care if the symptoms become severe, long-lasting, or are associated with new concerning features."
]

def out():
    return random.choice(OUTPUT_TEMPLATES)

# ---------------- MASSIVE SYMPTOM SPACE ----------------

BODY_PARTS = [
    "upper back","lower back","left shoulder","right shoulder","jaw","temple","left calf","right calf",
    "left thigh","right thigh","left hip","right hip","left ankle","right ankle","left wrist","right wrist",
    "left elbow","right elbow","left knee","right knee","abdomen","stomach","chest","neck","scalp","forehead",
    "left foot","right foot","left hand","right hand","rib area","pelvis","groin","lower abdomen","upper abdomen"
]

SENSATIONS = [
    "tightness","pressure","warmth","tingling","numbness","itching","burning","heaviness","soreness",
    "pulsing","stiffness","sensitivity","dull ache","light throbbing","sharp discomfort","mild cramping",
    "fluttering","pulling sensation","pinching","radiating discomfort","intermittent tingling"
]

TRIGGERS = [
    "after waking up","after eating","during light activity","during exercise","when I stand up",
    "when I sit for a long time","when I bend forward","when I lie down","when I turn my head",
    "when I walk uphill","in cold weather","in warm weather","when I feel stressed","after long screen time",
    "when I stretch","when I climb stairs","after resting","after carrying something","after talking a long walk"
]

INTENSITY = ["slight","mild","noticeable","more than usual","intermittent","occasional","frequent"]

# ---------------- GENERATOR ----------------

def generate_symptom():
    part = random.choice(BODY_PARTS)
    sens = random.choice(SENSATIONS)
    trig = random.choice(TRIGGERS)
    intensity = random.choice(INTENSITY)
    return f"Patient: I feel {intensity} {sens} in my {part} {trig}."

# ---------------- MAIN ----------------

def main():
    samples = []
    used = set()

    while len(samples) < 500:
        inp = generate_symptom()
        if inp in used:
            continue
        used.add(inp)
        samples.append({
            "instruction": INSTRUCTION,
            "input": inp,
            "output": out()
        })

    random.shuffle(samples)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Generated {len(samples)} unique samples → {OUT_PATH}")

if __name__ == "__main__":
    main()

# run
# python scripts/finetune/struc.py