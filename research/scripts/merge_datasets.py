import json
import random
import os
from tqdm import tqdm


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(path)}", unit="lines"):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def merge_and_split(train_files, val_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Load training data
    # -------------------------
    print("\n📥 Loading training datasets...")
    train_items = []
    for fpath in tqdm(train_files, desc="Train files", unit="file"):
        print(f" → {fpath}")
        items = load_jsonl(fpath)
        train_items.extend(items)

    print(f"\n🔀 Shuffling {len(train_items):,} training samples...")
    random.shuffle(train_items)

    # -------------------------
    # Load validation data
    # -------------------------
    print("\n📥 Loading validation datasets...")
    val_items = []
    for fpath in tqdm(val_files, desc="Val files", unit="file"):
        print(f" → {fpath}")
        items = load_jsonl(fpath)
        val_items.extend(items)

    print(f"\n🔀 Shuffling {len(val_items):,} validation samples...")
    random.shuffle(val_items)

    # -------------------------
    # Save merged datasets
    # -------------------------
    train_raw = os.path.join(output_dir, "train.jsonl")
    val_raw = os.path.join(output_dir, "val.jsonl")

    print("\n💾 Saving training data...")
    with open(train_raw, "w", encoding="utf-8") as f:
        for item in tqdm(train_items, desc="Saving train", unit="samples"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n💾 Saving validation data...")
    with open(val_raw, "w", encoding="utf-8") as f:
        for item in tqdm(val_items, desc="Saving val", unit="samples"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n✅ Raw merged:")
    print(f"   → {train_raw} ({len(train_items):,} samples)")
    print(f"   → {val_raw} ({len(val_items):,} samples)")

    return train_raw, val_raw


if __name__ == "__main__":
    # Define your files
    train_files = [
        "data/raw/ultrachat_train.jsonl",
        "data/raw/smol-rewrite_train.jsonl",
        "data/raw/smol-summarize_train.jsonl",
    ]

    val_files = [
        "data/raw/ultrachat_val.jsonl",
        "data/raw/smol-rewrite_test.jsonl",
        "data/raw/smol-summarize_test.jsonl",
    ]

    train_raw, val_raw = merge_and_split(train_files, val_files, "data/processed")

    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Format training data:")
    print(f"   python scripts/format.py --input {train_raw} --output data/processed/train_formatted.jsonl")
    print("2. Format validation data:")
    print(f"   python scripts/format.py --input {val_raw} --output data/processed/val_formatted.jsonl")
    print("3. Train:")
    print("   python -m src.train")

# Run 
# python scripts/merge_datasets.py