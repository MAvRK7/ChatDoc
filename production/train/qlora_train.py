'''
============================================================
Training Data & Fine-Tuning Information
============================================================

Base Model:
    google/gemma-4-E2B-it

Dataset Preparation:

This model was fine-tuned on a custom dataset created by
combining UltraChat and MedDialog conversation data.

The combined dataset was:
    - Cleaned and anonymized
    - Converted to a unified JSONL chat format
    - Split into 95% training and 5% validation data
    - Uploaded to Kaggle for storage and distribution

Dataset Files:

Training Split (95%):
https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl?select=train.jsonl
File: train.jsonl (1.38 GB)

Validation Split (5%):
https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl?select=val_formatted.jsonl
File: val_formatted.jsonl (73.25 MB)

Training Method:
    - QLoRA (4-bit NF4 quantization)
    - Assistant-only loss masking
    - LoRA Rank (r): 8
    - LoRA Alpha: 16
    - LoRA Dropout: 0.05
    - Mixed instruction-tuning using general chat and medical data

Environment:
    - Trained on Kaggle NVIDIA T4 GPUs
    - Training time: approximately 4–5 hours
    - Transformers version: 5.8.0
      (required for Gemma 4 support)

This script saves LoRA adapter weights that can be loaded 
on top of the base Gemma 4 model or merged for inference. 

The resulting adapter was published as: 
https://www.kaggle.com/datasets/satvikraghav/chat-doctor-gemma4-lora 

Package Contents: 
    - adapter_model.safetensors 
    - adapter_config.json 
    - tokenizer.json 
    - tokenizer_config.json 
    - chat_template.jinja 
    - README.md 
    
Package Size: 
    - Version 1: 92.02 MB 

Note: The published package contains LoRA adapter weights, 
not the full Gemma 4 model.

Date: 2026-05-16

============================================================
'''

import os
import json
import gc
import glob
import torch
torch.backends.cudnn.benchmark = True
from dataclasses import dataclass
from transformers import AutoProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from trl import SFTTrainer, SFTConfig
from transformers import (
    BitsAndBytesConfig,
    TrainerCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset

MODEL_ID = "google/gemma-4-E2B-it"

# =========================
# CONFIG
# =========================

class Config:
    ultrachat_path   = "data/processed/train.jsonl"
    medical_path     = "data/finetune/train_deduped.jsonl"
    output_dir       = "checkpoints/gemma-lora"
    final_dir        = "checkpoints/gemma-lora-final"
    tokenized_cache  = "cache/tokenized_dataset_v3"  # new cache for new format

    batch_size        = 1
    grad_accum_steps  = 8
    max_length        = 512      # try 512 first, fallback if OOM
    
    lr            = 1e-4         # lower for stability
    weight_decay  = 0.0          # standard for LoRA
    warmup_steps  = 100
    max_steps     = 1000         # fewer steps, curated data
    
    eval_every    = 250
    save_total    = 3
    log_every     = 10
    
    lora_r        = 8            # was 4 — minimum for real learning
    lora_alpha    = 16           # 2x r
    lora_dropout  = 0.05
    
    # Data curation
    max_samples_ultrachat = 8000   # ~10k total with medical
    max_samples_medical = 3708     # all medical, it's small and high quality

# =========================
# TOKENIZER
# =========================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# SMART MASKING: Only train on assistant responses
# =========================

def tokenize_with_assistant_masking(messages, tokenizer, max_length):
    """
    Proper assistant-only masking.

    Only assistant responses contribute to loss.
    User/system/template tokens are masked with -100.
    """

    # Full conversation text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Full tokenization
    full_encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]

    seq_len = len(input_ids)

    # Everything masked by default
    labels = [-100] * seq_len

    # Build progressively to locate assistant spans
    prev_len = 0

    for i, msg in enumerate(messages):
        partial_messages = messages[: i + 1]

        partial_text = tokenizer.apply_chat_template(
            partial_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        partial_encoded = tokenizer(
            partial_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        current_len = len(partial_encoded["input_ids"])

        # Assistant spans become trainable
        if msg["role"] == "assistant":
            start = min(prev_len, seq_len)
            end = min(current_len, seq_len)

            for j in range(start, end):
                labels[j] = input_ids[j]

        prev_len = current_len

        # Stop if already truncated
        if prev_len >= seq_len:
            break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "token_type_ids": [0] * seq_len,
        "mm_token_type_ids": [0] * seq_len,
    }

# =========================
# QUALITY FILTERING: Pick diverse, complete samples
# =========================

def score_ultrachat_sample(example):
    """Score sample quality: prefer multi-turn, complete conversations."""
    messages = example.get("messages", [])
    
    # Must have at least 2 turns (user + assistant)
    if len(messages) < 2:
        return 0
    
    # Prefer conversations with 3-6 turns (not too short, not endless)
    turn_score = 1.0 if 3 <= len(messages) <= 6 else 0.5
    
    # Check last message is assistant (complete conversation)
    last_is_assistant = messages[-1].get("role") == "assistant"
    if not last_is_assistant:
        return 0  # incomplete, skip
    
    # Prefer longer assistant responses (more learning signal)
    last_content = str(messages[-1].get("content", ""))
    length_score = min(len(last_content) / 500, 1.0)  # normalize to 500 chars
    
    return turn_score * 0.5 + length_score * 0.5

def score_medical_sample(example):
    """Medical data is already high quality, just check completeness."""
    output = str(example.get("output", ""))
    input_text = str(example.get("input", ""))
    
    # Must have both input and output
    if not output or not input_text:
        return 0
    
    # Prefer longer, detailed explanations
    length_score = min(len(output) / 300, 1.0)
    
    # Penalize very short outputs (likely low quality)
    if len(output) < 50:
        return 0.1
    
    return length_score

# =========================
# DATASET BUILDING
# =========================

def cache_is_valid(path):
    return os.path.exists(
        os.path.join(path, "dataset_info.json")
    )

def build_and_cache_dataset():
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
                    print(f"Warning: skipping bad line {i}: {e}")
        return data

    print("Loading UltraChat...")
    ultrachat_raw = load_jsonl_safe(Config.ultrachat_path)
    
    # Score and filter
    print(f"Scoring {len(ultrachat_raw)} ultrachat samples...")
    scored = [(ex, score_ultrachat_sample(ex)) for ex in ultrachat_raw]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N diverse samples
    ultrachat_selected = [ex for ex, score in scored[:Config.max_samples_ultrachat] if score > 0]
    print(f"Selected {len(ultrachat_selected)} high-quality ultrachat samples")
    
    ultrachat = Dataset.from_list(ultrachat_selected)
    
    print("Loading medical data...")
    medical = load_dataset("json", data_files=Config.medical_path, split="train")
    
    # Score medical too
    scored_med = [(medical[i], score_medical_sample(medical[i])) for i in range(len(medical))]
    scored_med.sort(key=lambda x: x[1], reverse=True)
    medical_selected = [ex for ex, score in scored_med if score > 0.3]
    print(f"Selected {len(medical_selected)} high-quality medical samples")
    
    medical = Dataset.from_list(medical_selected)
    
    # Repeat medical 5x for balance
    medical_repeated = concatenate_datasets([medical] * 5)
    combined = concatenate_datasets([ultrachat, medical_repeated]).shuffle(seed=42)
    print(f"Total training samples: {len(combined):,}")
    
    # =========================
    # SANITIZATION
    # =========================

    def sanitize_messages(messages):
        """Ensure all messages have valid string content."""
        if not messages or not isinstance(messages, list):
            return []

        cleaned = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content")

            # Handle None or non-string content
            if content is None:
                content = ""

            elif isinstance(content, list):
                # Handle multimodal-style blocks
                texts = []

                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        texts.append(block.get("text", ""))

                content = " ".join(texts) if texts else ""

            elif not isinstance(content, str):
                content = str(content)

            cleaned.append({
                "role": role,
                "content": content,
            })

        return cleaned
    
    def format_ultrachat(example):
        messages = sanitize_messages(
            example.get("messages", [])
        )

        if not messages:
            return {"text": ""}

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text,
            "messages": messages,
        }


    def format_medical(example):
        messages = sanitize_messages([
            {
                "role": "system",
                "content": str(example.get("instruction", "")),
            },
            {
                "role": "user",
                "content": str(example.get("input", "")),
            },
            {
                "role": "assistant",
                "content": str(example.get("output", "")),
            },
        ])

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text,
            "messages": messages,
        }

    # =========================
    # FORMAT DATA
    # =========================

    print("Formatting dataset...")  

    def format_example(example):
        if "messages" in example:
            return format_ultrachat(example)
        else:
            return format_medical(example)

    combined = combined.map(
        format_example,
        desc="Formatting",
    )

    # Remove empty/broken samples
    combined = combined.filter(
        lambda x: len(x.get("text", "")) > 10
    )

    # =========================
    # TOKENIZE WITH MASKING
    # =========================

    print("Tokenizing with assistant-only masking...")

    def tokenize_fn(example):
        messages = example["messages"]

        return tokenize_with_assistant_masking(
            messages,
            tokenizer,
            Config.max_length,
        )

    combined = combined.map(
        tokenize_fn,
        remove_columns=combined.column_names,
        desc="Tokenizing",
    )
    
    # Filter out samples where labels are all -100 (no assistant content)
    def has_valid_labels(example):
        return any(l != -100 for l in example["labels"])
    
    valid_count = sum(1 for ex in combined if has_valid_labels(ex))
    print(f"Samples with valid assistant labels: {valid_count}/{len(combined)}")
    
    combined = combined.filter(has_valid_labels)
    
    os.makedirs(Config.tokenized_cache, exist_ok=True)
    combined.save_to_disk(Config.tokenized_cache)
    print(f"Saved to {Config.tokenized_cache}")
    return combined

if cache_is_valid(Config.tokenized_cache):
    print(f"Loading cache from {Config.tokenized_cache}...")
    combined = Dataset.load_from_disk(Config.tokenized_cache)
    print(f"Loaded {len(combined):,} samples")
else:
    print("Building dataset...")
    combined = build_and_cache_dataset()

# =========================
# AUTO FALLBACK: Try 512, fallback to 256 if OOM
# =========================

def try_training_with_length(target_length, combined, tokenizer):
    """Attempt training with given max_length, return (success, trainer)."""
    print(f"\n{'='*50}")
    print(f"ATTEMPTING max_length={target_length}")
    print(f"{'='*50}")
    
    # Update tokenizer and re-tokenize if needed
    tokenizer.model_max_length = target_length
    
    # Check if current cache matches target length
    sample = combined[0] if len(combined) > 0 else None
    if sample and len(sample["input_ids"]) > target_length:
        print("Cache has longer sequences, need to rebuild...")
        # This would require rebuilding — for now, just filter
        pass
    
    try:
        # Build model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        
        # Unwrap
        try:
            from transformers.models.gemma4 import modeling_gemma4
            ClippableLinear = getattr(modeling_gemma4, "Gemma4ClippableLinear", None)
            if ClippableLinear:
                replaced = 0
                for _, parent in list(model.named_modules()):
                    for child_name, child in list(parent.named_children()):
                        if isinstance(child, ClippableLinear):
                            setattr(parent, child_name, child.linear)
                            replaced += 1
                print(f"Unwrapped {replaced} ClippableLinear modules")
        except ImportError:
            pass
        
        # Drop vision/audio
        model.vision_tower = None
        model.audio_tower = None
        model.config.vision_config = None
        model.config.audio_config = None
        
        # Freeze + prep
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "norm" in name:
                param.data = param.data.to(torch.float32)
        
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # LoRA
        lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=Config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                param.data = param.data.to(torch.float16)
        
        model.print_trainable_parameters()
        gc.collect()
        torch.cuda.empty_cache()
        
        return True, model
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM at max_length={target_length}: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return False, None

# Try 512 first, then 256, then 128
for attempt_length in [512, 256, 128]:
    success, model = try_training_with_length(attempt_length, combined, tokenizer)
    if success:
        Config.max_length = attempt_length
        tokenizer.model_max_length = attempt_length
        print(f"\n✅ SUCCESS: Training with max_length={attempt_length}")
        break
else:
    raise RuntimeError("Failed to train at any length")

# =========================
# COLLATOR
# =========================

@dataclass
class Gemma4Collator:
    pad_token_id: int
    
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.pad_token_id
        
        batch = {k: [] for k in ["input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids", "labels"]}
        
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append([1]*seq_len + [0]*pad_len)
            batch["token_type_ids"].append([0] * max_len)
            batch["mm_token_type_ids"].append([0] * max_len)
            batch["labels"].append(f["labels"] + [-100]*pad_len)
        
        return {k: torch.tensor(v) for k, v in batch.items()}

collator = Gemma4Collator(pad_token_id=tokenizer.pad_token_id)

# =========================
# CALLBACK
# =========================

class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, max_new_tokens=80):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What should I do if I have a fever?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 250 == 0 and state.global_step > 0:
            m = kwargs["model"]
            m.eval()
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        self.prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=64,
                    ).to(m.device)
                    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
                    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
                    
                    out = m.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    new_tokens = out[0][inputs["input_ids"].shape[1]:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                print(f"\n=== Step {state.global_step:,} ===\n{text}\n")
            except Exception as e:
                print(f"Gen error: {e}")
            finally:
                m.train()

# =========================
# TRAINING
# =========================

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    return ckpts[-1]

resume_from = find_latest_checkpoint(Config.output_dir)

sft_config = SFTConfig(
    output_dir=Config.output_dir,
    max_steps=Config.max_steps,
    per_device_train_batch_size=Config.batch_size,
    gradient_accumulation_steps=Config.grad_accum_steps,
    learning_rate=Config.lr,
    weight_decay=Config.weight_decay,
    max_grad_norm=0.3,
    warmup_steps=Config.warmup_steps,
    lr_scheduler_type="cosine",
    logging_steps=Config.log_every,
    save_strategy="steps",
    save_steps=Config.eval_every,
    save_total_limit=Config.save_total,
    fp16=False,
    bf16=False,
    optim="paged_adamw_8bit",
    report_to="tensorboard",
    dataset_text_field=None,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=combined,
    data_collator=collator,
    processing_class=tokenizer,
    callbacks=[GenerateTextCallback(tokenizer)],
)

print(f"\nStarting training with max_length={Config.max_length}...")
trainer.train(resume_from_checkpoint=resume_from)

print(f"\nSaving to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done!")