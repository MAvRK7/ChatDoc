# File paths
files = {
    "Ultra Chat": [
        "data/raw/ultrachat_train.jsonl",
        "data/raw/ultrachat_val.jsonl",
    ],
    "Smol Summarize": [
        "data/raw/smol-summarize_train.jsonl",
        "data/raw/smol-summarize_test.jsonl"
    ],
    "Smol rewrite": [
        "data/raw/smol-rewrite_test.jsonl",
        "data/raw/smol-rewrite_train.jsonl"
    ]
}

def count_jsonl(file_path):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

# Count conversations per category
category_counts = {}

for category, paths in files.items():
    total = 0
    for path in paths:
        total += count_jsonl(path)
    category_counts[category] = total

'''
# Manually define breakdown for greetings/identity/refusal
greet_samples = 3000
identity_samples = 3000
refusal_samples = 10000
'''

# Total dataset size
total_dataset = sum(category_counts.values())

# Print results
print("Counts per category:\n")
for category, count in category_counts.items():
    percentage = (count / total_dataset) * 100
    print(f"{category}: {count} ({percentage:.2f}%)")

'''
print("\nBreakdown of Greetings/Identity/Refusal:")
gir_total = greet_samples + identity_samples + refusal_samples
print(f"Greeting samples: {greet_samples} ({(greet_samples/gir_total)*100:.2f}%)")
print(f"Identity samples: {identity_samples} ({(identity_samples/gir_total)*100:.2f}%)")
print(f"Refusal samples: {refusal_samples} ({(refusal_samples/gir_total)*100:.2f}%)")
'''

print(f"\nTotal dataset size: {total_dataset}")

files = {
    "train": "data/processed/train_formatted.jsonl",
    "val": "data/processed/val_formatted.jsonl"
}

def count_jsonl(file_path):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count 

# Count conversations per category
category_counts = {}    
for category, path in files.items():
    category_counts[category] = count_jsonl(path)

# print
print("Counts per category:\n")
for category, count in category_counts.items():
    print(f"{category}: {count}")  

# Run 
# python -m scripts.dataset_counter