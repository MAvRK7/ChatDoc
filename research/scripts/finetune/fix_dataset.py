#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import Counter
import random

INPUT_PATH = Path("data/finetune/train.jsonl")
OUTPUT_PATH = Path("data/finetune/cleaned_train.jsonl")
REPORT_PATH = Path("data/finetune/clean_report.json")

# ---------------- HELPERS ----------------
def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))

def rewrite_symptom(text):
    """Rewrite repeated symptoms using algorithmic variation."""
    BODY = ["chest", "stomach", "abdomen", "neck", "back", "shoulder", "leg", "arm",
            "hip", "knee", "wrist", "ankle", "temple", "forehead", "jaw"]
    SENS = ["pressure", "tightness", "warmth", "tingling", "numbness", "itching",
            "burning", "heaviness", "soreness", "pulsing", "sensitivity"]
    TRIG = ["after waking up", "after eating", "during exercise", "when I stand up",
            "when I sit too long", "when I bend forward", "when I lie down",
            "when I turn my head", "when I walk uphill", "in cold weather",
            "in warm weather", "when I’m stressed", "after long screen time"]

    new = f"I feel {random.choice(SENS)} in my {random.choice(BODY)} {random.choice(TRIG)}."
    return "Patient: " + new

# ---------------- LOAD ----------------
samples = []
with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

original_count = len(samples)

# ---------------- REMOVE EXACT DUPLICATES ----------------
seen_inputs = set()
seen_outputs = set()

cleaned = []
removed_exact_input = 0
removed_exact_output = 0

for ex in samples:
    ni = normalize(ex["input"])
    no = normalize(ex["output"])

    if ni in seen_inputs:
        removed_exact_input += 1
        continue
    if no in seen_outputs:
        removed_exact_output += 1
        continue

    seen_inputs.add(ni)
    seen_outputs.add(no)
    cleaned.append(ex)

# ---------------- REMOVE NEAR-DUPLICATE OUTPUTS ----------------
final = []
removed_similar = 0

for ex in cleaned:
    too_similar = False
    for other in final:
        if jaccard(ex["output"], other["output"]) > 0.65:
            too_similar = True
            removed_similar += 1
            break
    if not too_similar:
        final.append(ex)

# ---------------- FIX OVERUSED SYMPTOMS ----------------
symptoms = Counter()
for ex in final:
    m = re.search(r"patient:\s*(.*)", ex["input"], re.I)
    if m:
        symptoms[m.group(1).strip().lower()] += 1

overused = {s for s, c in symptoms.items() if c > 10}

rewritten = 0
for ex in final:
    m = re.search(r"patient:\s*(.*)", ex["input"], re.I)
    if not m:
        continue
    sym = m.group(1).strip().lower()
    if sym in overused:
        ex["input"] = rewrite_symptom(ex["input"])
        rewritten += 1

# ---------------- SAVE CLEANED DATASET ----------------
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for ex in final:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# ---------------- SAVE REPORT ----------------
report = {
    "original_count": original_count,
    "final_count": len(final),
    "removed_exact_input_duplicates": removed_exact_input,
    "removed_exact_output_duplicates": removed_exact_output,
    "removed_near_duplicate_outputs": removed_similar,
    "rewritten_overused_symptoms": rewritten,
    "overused_symptom_patterns": list(overused)
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Dataset cleaned.")
print("Original:", original_count)
print("Final:", len(final))
print("Exact input dupes removed:", removed_exact_input)
print("Exact output dupes removed:", removed_exact_output)
print("Near-duplicate outputs removed:", removed_similar)
print("Overused symptoms rewritten:", rewritten)

# Run
# python scripts/finetune/fix_dataset.py