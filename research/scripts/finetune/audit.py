#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import math

DATA_PATH = Path("data/finetune/train_deduped.jsonl")

def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def word_count(s):
    return len(s.split())

def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))

samples = []
with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            samples.append(json.loads(line))
        except:
            print("❌ Bad JSON:", line[:80])

print("\n==============================")
print(" ADVANCED DATASET QUALITY AUDIT")
print("==============================\n")

print(f"Total samples: {len(samples)}\n")

# ---------------- FIELD CHECK ----------------
missing = {
    "instruction": 0,
    "input": 0,
    "output": 0
}

for ex in samples:
    for k in missing:
        if k not in ex:
            missing[k] += 1

print("Field completeness:")
for k, v in missing.items():
    print(f"  Missing {k}: {v}")
print()

# ---------------- LENGTH STATS ----------------
inputs = [ex["input"] for ex in samples]
outputs = [ex["output"] for ex in samples]

input_lens = [word_count(x) for x in inputs]
output_lens = [word_count(x) for x in outputs]

print("Length statistics:")
print(f"  Input avg: {sum(input_lens)/len(input_lens):.2f}")
print(f"  Input min: {min(input_lens)}")
print(f"  Input max: {max(input_lens)}")
print(f"  Output avg: {sum(output_lens)/len(output_lens):.2f}")
print(f"  Output min: {min(output_lens)}")
print(f"  Output max: {max(output_lens)}\n")

# ---------------- DUPLICATES ----------------
norm_inputs = [normalize(x) for x in inputs]
norm_outputs = [normalize(x) for x in outputs]

input_counts = Counter(norm_inputs)
output_counts = Counter(norm_outputs)

dupe_inputs = {k: v for k, v in input_counts.items() if v > 1}
dupe_outputs = {k: v for k, v in output_counts.items() if v > 1}

print("Duplicate counts:")
print(f"  Duplicate inputs:  {len(dupe_inputs)}")
print(f"  Duplicate outputs: {len(dupe_outputs)}\n")

# ---------------- TEMPLATE SIMILARITY ----------------
print("Template similarity (output Jaccard index):")
pairs = []
for i in range(0, len(outputs), 200):
    for j in range(i+1, min(i+200, len(outputs))):
        sim = jaccard(outputs[i], outputs[j])
        if sim > 0.65:
            pairs.append(sim)

if pairs:
    print(f"  High-similarity output pairs (>0.65): {len(pairs)}")
else:
    print("  No high-similarity clusters detected.")
print()

# ---------------- SYMPTOM PATTERN EXTRACTION ----------------
symptoms = Counter()
for text in norm_inputs:
    m = re.search(r"i feel ([^.]+)", text)
    if m:
        symptoms[m.group(1).strip()] += 1

print("Top 20 repeated symptom patterns:")
for s, c in symptoms.most_common(20):
    print(f"  {s} — {c}")
print()

# ---------------- CATEGORY COVERAGE ESTIMATION ----------------
categories = {
    "chest": 0, "stomach": 0, "abdomen": 0, "back": 0, "neck": 0,
    "throat": 0, "head": 0, "eyes": 0, "ears": 0, "nose": 0,
    "legs": 0, "arms": 0, "joints": 0, "skin": 0, "breathing": 0,
    "urinary": 0, "menstrual": 0, "sleep": 0, "fatigue": 0
}

for text in norm_inputs:
    for cat in categories:
        if cat in text:
            categories[cat] += 1

print("Category coverage:")
for cat, count in categories.items():
    print(f"  {cat}: {count}")
print()

# ---------------- NEXT BATCH RECOMMENDATION ----------------
print("Next batch recommendations:")

if len(dupe_inputs) > 50:
    print("  • Reduce repeated symptom patterns (algorithmic generator needed).")

if len(dupe_outputs) > 20:
    print("  • Add new output templates to diversify phrasing.")

if categories["skin"] < 80:
    print("  • Add more skin-related symptoms.")

if categories["urinary"] < 60:
    print("  • Add more urinary symptoms.")

if categories["sleep"] < 60:
    print("  • Add more sleep-related symptoms.")

if categories["menstrual"] < 40:
    print("  • Add more menstrual-cycle symptoms (general, safe).")

print("\n==============================")
print(" END OF ADVANCED AUDIT")
print("==============================\n")

# Run
# python scripts/finetune/audit.py