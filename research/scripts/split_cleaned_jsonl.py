import random
import os

input_path = "data/processed/merged.jsonl" # Main dataset Path
train_out = "data/processed/train.jsonl" # 95% for training
val_out = "data/processed/val.jsonl" # 5% for validation

os.makedirs("data/processed", exist_ok=True)

# Load all lines
lines = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            lines.append(line)

# Shuffle
random.shuffle(lines)

# 95/5 split
split = int(0.95 * len(lines))
train_lines = lines[:split]
val_lines = lines[split:]

# Save
with open(train_out, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(val_out, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Train size:", len(train_lines))
print("Val size:", len(val_lines))

# Run
# python scripts/split_cleaned_jsonl.py
