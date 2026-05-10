import os
import json
import torch
from datasets import Dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer
from transformers import AutoProcessor, AutoTokenizer

# Mock paths — point to small test files or create dummy ones
ULTRACHAT_PATH = "data/processed/train.jsonl"
MEDICAL_PATH = "data/finetune/train_deduped.jsonl"

# Create tiny dummy data if files don't exist
def make_dummy_jsonl(path, n=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        for i in range(n):
            if "ultrachat" in path or "processed" in path:
                obj = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": f"Hi {i}"}]}
            else:
                obj = {"instruction": "Be helpful", "input": f"Question {i}", "output": f"Answer {i}"}
            f.write(json.dumps(obj) + "\n")
    print(f"Created dummy: {path}")

make_dummy_jsonl(ULTRACHAT_PATH)
make_dummy_jsonl(MEDICAL_PATH)

# --- Your actual logic, stripped down ---

MODEL_ID = "google/gemma-4-E2B-it"

class TokenizerPatcher:
    def __enter__(self):
        self._orig = transformers.tokenization_utils_base.PreTrainedTokenizerBase._set_model_specific_special_tokens
        def noop(*args, **kwargs):
            pass
        transformers.tokenization_utils_base.PreTrainedTokenizerBase._set_model_specific_special_tokens = noop
        return self
    
    def __exit__(self, *args):
        transformers.tokenization_utils_base.PreTrainedTokenizerBase._set_model_specific_special_tokens = self._orig

with TokenizerPatcher():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading processor...")
#tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
#tokenizer = Gemma4TokenizerFast.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_ultrachat(example):
    messages = example["messages"]
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            content = content[0]["text"] if content else ""
        text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    return {"text": text}

def format_medical(example):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": example["instruction"]}]},
        {"role": "user", "content": [{"type": "text", "text": example["input"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["output"]}]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

def load_jsonl_safe(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Skipping bad line {i}: {e}")
    return data

print("\n--- Loading datasets ---")
ultrachat_raw = load_jsonl_safe(ULTRACHAT_PATH)
medical_raw = load_jsonl_safe(MEDICAL_PATH)

ultrachat = Dataset.from_list(ultrachat_raw).map(format_ultrachat, remove_columns=ultrachat_raw[0].keys())
medical = Dataset.from_list(medical_raw).map(format_medical, remove_columns=medical_raw[0].keys())

medical_repeated = concatenate_datasets([medical] * 5)
combined = concatenate_datasets([ultrachat, medical_repeated]).shuffle(seed=42)

print(f"Total samples: {len(combined):,}")
print(f"Columns: {combined.column_names}")

# Verify a sample
print("\n--- Sample formatted text ---")
print(combined[0]["text"][:300] + "...")

# Verify tokenization works
print("\n--- Tokenization check ---")
tokens = tokenizer(combined[0]["text"], truncation=True, max_length=512)
print(f"Input IDs length: {len(tokens['input_ids'])}")
print(f"Decoded back: {tokenizer.decode(tokens['input_ids'][:50])}...")

print("\n✅ All checks passed. Data pipeline is solid.")

# run 
# python production/train/test_setup.py