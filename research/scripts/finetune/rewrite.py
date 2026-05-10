#!/usr/bin/env python3
import re
from pathlib import Path

INPUT_TXT = Path("extracted_symptoms.txt")
OUTPUT_TXT = Path("rewritten_symptoms.txt")

DISCARD_PATTERNS = [
    r"\bmy father\b",
    r"\bmy dad\b",
    r"\bmy mother\b",
    r"\bmy mom\b",
    r"\bmy son\b",
    r"\bmy daughter\b",
    r"\bmy child\b",
    r"\bmy 3 year old\b",
    r"\bmy 5 year old\b",
    r"\bmy 10 year old\b",
]

def should_discard(text):
    t = text.lower()
    return any(re.search(p, t) for p in DISCARD_PATTERNS)

def rewrite(text):
    t = text.lower().strip()

    # Remove leading commas or garbage
    t = re.sub(r"^[,.\s]+", "", t)

    # If it's already symptom-like, keep it
    if t.startswith("patient: i feel"):
        return "Patient: " + t[len("patient: "):].strip()

    # Convert common patterns
    if "i have" in t:
        return "Patient: I feel concerned because " + t.split("i have",1)[1].strip()

    if "i get" in t:
        return "Patient: I get " + t.split("i get",1)[1].strip()

    if "i am having" in t:
        return "Patient: I feel " + t.split("i am having",1)[1].strip()

    if "i am experiencing" in t:
        return "Patient: I feel " + t.split("i am experiencing",1)[1].strip()

    if "i am worried" in t:
        return "Patient: I feel worried about " + t.split("i am worried",1)[1].strip()

    if "i am concerned" in t:
        return "Patient: I feel concerned about " + t.split("i am concerned",1)[1].strip()

    if "i feel" in t:
        return "Patient: " + t.split("i feel",1)[1].strip()

    # Fallback: convert to a general concern
    return "Patient: I feel concerned about " + t

def main():
    with INPUT_TXT.open("r", encoding="utf-8") as f, OUTPUT_TXT.open("w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if should_discard(line):
                continue

            rewritten = rewrite(line)
            out.write(rewritten + "\n")

    print("Rewritten symptoms saved to:", OUTPUT_TXT)

if __name__ == "__main__":
    main()

# run 
# python scripts/finetune/rewrite.py