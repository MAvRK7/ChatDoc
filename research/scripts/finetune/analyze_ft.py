#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
import re

# ---------------- CONFIG ----------------
DATA_PATH = Path("data/finetune/train.jsonl")   # <-- change if needed

# ---------------- HELPERS ----------------
def safe_len(text):
    return len(text.split())

def tokenize_estimate(text):
    # crude but consistent estimate
    return int(len(text) / 4)

def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())

# ---------------- LOAD DATA ----------------
samples = []
with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            samples.append(json.loads(line))
        except:
            print("❌ Bad JSON line:", line[:80])

total = len(samples)
print(f"\n==============================")
print(f" DATASET ANALYSIS REPORT")
print(f"==============================")
print(f"Total samples: {total}\n")

# ---------------- FIELD CHECK ----------------
missing_instruction = 0
missing_input = 0
missing_output = 0

for ex in samples:
    if "instruction" not in ex: missing_instruction += 1
    if "input" not in ex: missing_input += 1
    if "output" not in ex: missing_output += 1

print("Field presence:")
print(f"  Missing instruction: {missing_instruction}")
print(f"  Missing input:       {missing_input}")
print(f"  Missing output:      {missing_output}\n")

# ---------------- LENGTH STATS ----------------
input_lens = [safe_len(ex["input"]) for ex in samples if "input" in ex]
output_lens = [safe_len(ex["output"]) for ex in samples if "output" in ex]

print("Length statistics (words):")
print(f"  Input avg: {sum(input_lens)/len(input_lens):.2f}")
print(f"  Input min: {min(input_lens)}")
print(f"  Input max: {max(input_lens)}")
print(f"  Output avg: {sum(output_lens)/len(output_lens):.2f}")
print(f"  Output min: {min(output_lens)}")
print(f"  Output max: {max(output_lens)}\n")

# ---------------- TOKEN ESTIMATES ----------------
input_tokens = [tokenize_estimate(ex["input"]) for ex in samples]
output_tokens = [tokenize_estimate(ex["output"]) for ex in samples]

print("Token estimates:")
print(f"  Input avg tokens:  {sum(input_tokens)/len(input_tokens):.2f}")
print(f"  Output avg tokens: {sum(output_tokens)/len(output_tokens):.2f}")
print(f"  Total estimated tokens: {sum(input_tokens)+sum(output_tokens)}\n")

# ---------------- DUPLICATE DETECTION ----------------
norm_inputs = [normalize(ex["input"]) for ex in samples]
norm_outputs = [normalize(ex["output"]) for ex in samples]

input_counts = Counter(norm_inputs)
output_counts = Counter(norm_outputs)

dupe_inputs = [s for s, c in input_counts.items() if c > 1]
dupe_outputs = [s for s, c in output_counts.items() if c > 1]

print("Duplicate detection:")
print(f"  Duplicate inputs:  {len(dupe_inputs)}")
print(f"  Duplicate outputs: {len(dupe_outputs)}\n")

# ---------------- TOP REPEATED PHRASES ----------------
def top_phrases(texts, n=20):
    words = " ".join(texts).split()
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    return Counter(bigrams).most_common(n), Counter(trigrams).most_common(n)

bi_in, tri_in = top_phrases(norm_inputs)
bi_out, tri_out = top_phrases(norm_outputs)

print("Most common bigrams in INPUT:")
for phrase, count in bi_in[:10]:
    print(f"  {phrase} — {count}")

print("\nMost common trigrams in INPUT:")
for phrase, count in tri_in[:10]:
    print(f"  {phrase} — {count}")

print("\nMost common bigrams in OUTPUT:")
for phrase, count in bi_out[:10]:
    print(f"  {phrase} — {count}")

print("\nMost common trigrams in OUTPUT:")
for phrase, count in tri_out[:10]:
    print(f"  {phrase} — {count}")

# ---------------- SYMPTOM PATTERN CHECK ----------------
symptom_patterns = Counter()

for ex in samples:
    text = ex["input"].lower()
    # extract "I feel ..." patterns
    m = re.search(r"i feel ([^.]+)", text)
    if m:
        symptom_patterns[m.group(1).strip()] += 1

print("\nTop repeated symptom patterns:")
for s, c in symptom_patterns.most_common(20):
    print(f"  {s} — {c}")

print("\n==============================")
print(" END OF REPORT")
print("==============================\n")

# Run
# python scripts/finetune/analyze_ft.py