#!/usr/bin/env python3
import json
from pathlib import Path

IN_PATH = Path("data/finetune/train_deduped.jsonl")          # your combined dataset
OUT_PATH = Path("data/finetune/train_deduped_2.jsonl") # output file

def main():
    seen = set()
    kept = 0
    removed = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            sample = json.loads(line)
            inp = sample["input"].strip()

            if inp in seen:
                removed += 1
                continue

            seen.add(inp)
            kept += 1
            # Use separators to remove spaces in JSON
            fout.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Done.")
    print(f"Kept:    {kept}")
    print(f"Removed: {removed}")
    print(f"Output:  {OUT_PATH}")

if __name__ == "__main__":
    main()

# Run
# python scripts/finetune/dedupe.py