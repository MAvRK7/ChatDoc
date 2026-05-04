import json
import random

'''
with open("data/raw/ultrachat_train.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
'''
with open("data/raw/smol-rewrite_train.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}\n")

for line in random.sample(lines, 10):
    obj = json.loads(line)
    msgs = obj["messages"]
    
    # Show first user message and first assistant response
    first_user = next((m["content"] for m in msgs if m["role"] == "user"), "NONE")
    first_asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "NONE")
    
    print(f"USER: {first_user[:100]}...")
    print(f"ASST: {first_asst[:100]}...")
    print(f"Turns: {len(msgs)}")
    print("---")

# Run 
# python scripts/analyze_dataset.py 