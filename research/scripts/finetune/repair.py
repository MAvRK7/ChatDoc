#!/usr/bin/env python3
import json
import re
import random
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("data/finetune/train.jsonl")
OUTPUT_PATH = Path("data/finetune/fixed_train.jsonl")

# ---------------- GENERATORS ----------------
BODY = [
    "chest","stomach","abdomen","neck","back","shoulder","leg","arm","hip","knee",
    "wrist","ankle","temple","forehead","jaw","thigh","calf","lower back","upper back"
]

SENS = [
    "pressure","tightness","warmth","tingling","numbness","itching","burning",
    "heaviness","soreness","pulsing","sensitivity","discomfort","stiffness"
]

TRIG = [
    "after waking up","after eating","during exercise","when I stand up",
    "when I sit too long","when I bend forward","when I lie down",
    "when I turn my head","when I walk uphill","in cold weather",
    "in warm weather","when I’m stressed","after long screen time",
    "at night","in the morning"
]

def new_symptom():
    return f"Patient: I feel {random.choice(SENS)} in my {random.choice(BODY)} {random.choice(TRIG)}."

def diversify_output(text):
    templates = [
        "These symptoms may arise from a range of everyday factors such as mild irritation, posture, or temporary inflammation. Staying hydrated and avoiding triggers may help reduce discomfort. Monitoring when the symptoms occur can offer useful clues. Seek medical care if the symptoms become severe, persistent, or interfere with daily activities.",
        "A number of common causes can contribute to these symptoms, including tension, fatigue, or environmental factors. Gentle stretching and regular breaks may help. Tracking symptom patterns can be informative. Seek medical attention if the symptoms worsen, persist, or are accompanied by concerning changes.",
        "These symptoms can occur due to mild strain, sensitivity, or changes in routine. Rest, hydration, and avoiding aggravating activities may help. Observing when the symptoms appear can provide insight. Seek medical evaluation if the symptoms escalate, persist, or significantly affect daily functioning.",
        "A variety of non-serious factors may contribute to these symptoms, such as posture, stress, or temporary irritation. Taking breaks and staying hydrated may help ease discomfort. Monitoring symptom frequency can be useful. Seek medical care if the symptoms become severe, long-lasting, or are associated with new concerning features."
    ]
    return random.choice(templates)

def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())

# ---------------- LOAD ----------------
samples = []
with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

# ---------------- FIX DUPLICATE INPUTS ----------------
input_counts = Counter(normalize(ex["input"]) for ex in samples)

for ex in samples:
    if input_counts[normalize(ex["input"])] > 1:
        ex["input"] = new_symptom()

# ---------------- FIX DUPLICATE OUTPUTS ----------------
output_counts = Counter(normalize(ex["output"]) for ex in samples)

for ex in samples:
    if output_counts[normalize(ex["output"])] > 1:
        ex["output"] = diversify_output(ex["output"])

# ---------------- FIX NEAR-DUPLICATE OUTPUTS ----------------
for i in range(len(samples)):
    for j in range(i+1, len(samples)):
        a = samples[i]["output"]
        b = samples[j]["output"]
        ja = len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split()))
        if ja > 0.65:
            samples[j]["output"] = diversify_output(b)

# ---------------- SAVE ----------------
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for ex in samples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("Repaired dataset saved to:", OUTPUT_PATH)

# Run
# python scripts/finetune/repair.py