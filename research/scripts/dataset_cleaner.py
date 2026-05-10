import csv
import re
import json
import argparse
from tqdm import tqdm

# -----------------------------------
# BASIC CLEANING
# -----------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\b\+?\d[\d\-\s]{6,}\d\b', '[PHONE]', text)
    
    # Strip trailing escaped quotes and whitespace junk
    text = re.sub(r'\\+"\s*\\+"\s*$', '', text)
    text = re.sub(r'"\s+"\s*$', '', text)
    text = re.sub(r'\\+\s*$', '', text)
    
    # Remove template signatures
    text = re.sub(r"thank you.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"regards.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"chat\s*doctor.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"hope this helps.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"wish you.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"all the best.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"take care.*$", "", text, flags=re.IGNORECASE)
    
    text = re.sub(r"\s+", " ", text)
    text = text.strip(' \t\n\r"\\')
    
    return text.strip()


# -----------------------------------
# EXTRACT PAIRS FROM ONE CONVERSATION BLOCK
# -----------------------------------

def extract_pairs_from_block(block_text):
    """Extract all Human->AI pairs from a single conversation block."""
    conversations = []
    
    # Split on Human tags
    parts = block_text.split("[|Human|]")
    
    for part in parts[1:]:  # skip empty first split
        if "[|AI|]" not in part:
            continue
            
        human, ai = part.split("[|AI|]", 1)
        human = clean_text(human.strip())
        ai = clean_text(ai.strip())
        
        # VALIDATE
        if len(human) < 20 or len(ai) < 20:
            continue
        if len(human.split()) < 5 or len(ai.split()) < 10:
            continue
        if "[|Human|]" in ai or "[|AI|]" in human:
            continue
        # Reject cut-off answers
        if ai.endswith(("and", "or", "but", "to", "the", "a", "is", "are", "for")):
            continue
            
        conversations.append({
            "messages": [
                {"role": "user", "content": human},
                {"role": "assistant", "content": ai}
            ]
        })
    
    return conversations


# -----------------------------------
# PROCESS CSV FILE (ROW BY ROW)
# -----------------------------------

def process_csv_file(input_path):
    """Read CSV row by row — each row is one conversation block."""
    conversations = []
    
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        
        # Skip header
        try:
            header = next(reader)
        except StopIteration:
            return conversations
        
        for row in tqdm(reader, desc="Processing CSV rows"):
            if not row:
                continue
                
            # The conversation text is in the first column
            text = row[0] if row else ""
            if not text:
                continue
            
            # Clean up CSV quoting artifacts
            text = text.strip('"')
            text = text.replace('""', '"')  # Unescape CSV quotes
            
            # Extract pairs from this single row/block
            pairs = extract_pairs_from_block(text)
            conversations.extend(pairs)
    
    return conversations


# -----------------------------------
# PROCESS MEDDIALOG (unchanged)
# -----------------------------------

def process_med_dialogue(data):
    conversations = []
    for item in tqdm(data, desc="Processing MedDialog"):
        utts = item.get("utterances", [])
        if len(utts) < 2:
            continue
        user = clean_text(utts[0].replace("patient:", "").strip())
        assistant = clean_text(utts[1].replace("doctor:", "").strip())
        if len(user) < 5 or len(assistant) < 5:
            continue
        conversations.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ]
        })
    return conversations


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def clean_dataset(input_file, output_file):
    print(f"\n📥 Loading: {input_file}")

    if input_file.endswith(".json") or input_file.endswith(".jsonl"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        conversations = process_med_dialogue(data)
    else:
        conversations = process_csv_file(input_file)

    print(f"\n💾 Writing: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for conv in tqdm(conversations, desc="Saving JSONL"):
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\n✅ Done. Total conversations: {len(conversations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean_dataset(args.input, args.output)

# Run 
# Dataset 1 (MedDialogue)
# Train
# python scripts/dataset_cleaner.py --input data/raw/english-train.json --output data/processed/meddialog_train.jsonl
# Val
# python scripts/dataset_cleaner.py --input data/raw/english-dev.json --output data/processed/meddialog_dev.jsonl
#---
# Dataset 2 (Medical Conversation Corpus (100k+))
# python scripts/dataset_cleaner.py --input data/raw/train.csv --output data/processed/raw_clean.jsonl
