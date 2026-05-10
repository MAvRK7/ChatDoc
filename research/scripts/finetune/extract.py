#!/usr/bin/env python3
import csv
import re
from pathlib import Path

INPUT_CSV = Path("data/raw/ai-medical-chatbot.csv")
OUTPUT_TXT = Path("extracted_symptoms.txt")

# Patterns to remove
REMOVE_PATTERNS = [
    r"\bhi doctor\b",
    r"\bhello doctor\b",
    r"\bhi\b",
    r"\bhello\b",
    r"\bplease help\b",
    r"\bkindly help\b",
    r"\bdoctor\b",
    r"\bdr\b",
    r"\bi am a \d+-year-old\b",
    r"\bi am \d+ years old\b",
    r"\bi am \d+\b",
    r"\bage\b",
    r"\bmy question is\b",
    r"\bq\.\b",
]

# Extract symptom-like phrases
SYMPTOM_HINTS = [
    r"i feel [^.]+",
    r"i have [^.]+",
    r"i am having [^.]+",
    r"i get [^.]+",
    r"i started [^.]+",
    r"i noticed [^.]+",
    r"i am experiencing [^.]+",
    r"i am worried about [^.]+",
    r"i am concerned about [^.]+",
    r"my [^.]+ hurts",
    r"my [^.]+ feels [^.]+",
]

def clean_text(text):
    text = text.lower()
    for p in REMOVE_PATTERNS:
        text = re.sub(p, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_symptom(text):
    text = clean_text(text)

    # Try symptom patterns
    for pattern in SYMPTOM_HINTS:
        m = re.search(pattern, text)
        if m:
            return "Patient: " + m.group(0).strip()

    # Fallback: first meaningful sentence
    sentences = re.split(r"[.?!]", text)
    for s in sentences:
        s = s.strip()
        if len(s.split()) > 4:
            return "Patient: " + s

    # Final fallback
    return "Patient: " + text[:120]

def main():
    with INPUT_CSV.open("r", encoding="utf-8") as f, OUTPUT_TXT.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            patient_raw = row["Patient"]
            symptom = extract_symptom(patient_raw)
            out.write(symptom + "\n")

    print("Extracted symptoms saved to:", OUTPUT_TXT)

if __name__ == "__main__":
    main()

# Run
# python scripts/finetune/extract.py