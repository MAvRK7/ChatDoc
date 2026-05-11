import os
import json
import gc
import torch
from dataclasses import dataclass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# MUST come before any transformers import
# ============================================================
# Fix: SFTConfig in newer TRL dropped max_seq_length as a
# constructor arg — it moved to SFTTrainer. We handle this
# by not passing it at all and relying on tokenizer.model_max_length.
# ============================================================

from trl import SFTTrainer, SFTConfig
from transformers import (
    BitsAndBytesConfig,
    TrainerCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset

# =========================
# CONFIG
# =========================

MODEL_ID = "google/gemma-4-E2B-it"

class Config:
    ultrachat_path   = "data/processed/train.jsonl"
    medical_path     = "data/finetune/train_deduped.jsonl"
    output_dir       = "checkpoints/gemma-lora"
    final_dir        = "checkpoints/gemma-lora-final"

    batch_size        = 1
    grad_accum_steps  = 16
    max_length        = 128      # hard limit for T4 VRAM

    lr            = 2e-4
    weight_decay  = 0.01
    warmup_steps  = 100
    max_steps     = 5000

    eval_every    = 500
    log_every     = 25

    lora_r        = 4
    lora_alpha    = 8
    lora_dropout  = 0.05

# =========================
# TOKENIZER (load first — needed for data formatting)
# =========================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token        = tokenizer.eos_token
tokenizer.padding_side     = "right"
tokenizer.model_max_length = Config.max_length

# =========================
# MODEL
# =========================

print("Loading Gemma 4 E2B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # float16, not bfloat16 — T4 has no native bf16
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

# ============================================================
# FIX 1: Post-load unwrap of Gemma4ClippableLinear
#
# Gemma 4 wraps vision/audio attention layers in ClippableLinear
# (inherits nn.Module, not nn.Linear) so PEFT rejects them.
# Replacing each with its inner .linear after load preserves
# the correct checkpoint weights while giving PEFT plain nn.Linear.
#
# Note: if your PEFT is >= 0.19.0, omitting target_modules also
# works (it uses a regex scoped to LM layers), but the unwrap is
# safer across versions and costs nothing.
# ============================================================

def unwrap_clippable_linears(model):
    from transformers.models.gemma4 import modeling_gemma4
    ClippableLinear = getattr(modeling_gemma4, "Gemma4ClippableLinear", None)
    if ClippableLinear is None:
        print("Gemma4ClippableLinear not found — skipping unwrap (may not be needed).")
        return model

    replaced = 0
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if isinstance(child_module, ClippableLinear):
                # .linear is the inner nn.Linear (or Linear4bit) with correct weights
                setattr(parent_module, child_name, child_module.linear)
                replaced += 1

    print(f"Unwrapped {replaced} Gemma4ClippableLinear modules.")
    return model

model = unwrap_clippable_linears(model)

# ============================================================
# FIX 2: Manual kbit prep — avoids prepare_model_for_kbit_training
# which OOMs on T4 by casting ALL non-quantized params to fp32
# at once (including the ~400M embedding table).
# ============================================================

# Freeze all base weights
for param in model.parameters():
    param.requires_grad = False

# Upcast ONLY layernorms to fp32 — these are tiny (KB not GB)
# and fp32 is required for numerical stability during training
for name, param in model.named_parameters():
    if "norm" in name:
        param.data = param.data.to(torch.float32)

model.config.use_cache = False
model.enable_input_require_grads()   # required for LoRA grads to flow through frozen base
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

gc.collect()
torch.cuda.empty_cache()

# =========================
# LORA
# Omit target_modules entirely — PEFT >= 0.19.0 uses Gemma 4
# default targets scoped to LM layers via regex, safely skipping
# any remaining ClippableLinear modules.
# If you need to be explicit, use "all-linear" (NOT a named list)
# which also skips wrappers by walking to nn.Linear leaves.
# =========================

lora_config = LoraConfig(
    r            = Config.lora_r,
    lora_alpha   = Config.lora_alpha,
    # No target_modules — let PEFT use Gemma 4 defaults (LM layers only)
    lora_dropout = Config.lora_dropout,
    bias         = "none",
    task_type    = "CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Cast LoRA weights to float16 to match compute dtype
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
        param.data = param.data.to(torch.float16)

model.print_trainable_parameters()

gc.collect()
torch.cuda.empty_cache()

# =========================
# DATA
# =========================

def format_ultrachat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

def format_medical(example):
    messages = [
        {"role": "system",    "content": example["instruction"]},
        {"role": "user",      "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

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
                print(f"⚠️  Skipping bad line {i}: {e}")
    return data

print("Loading UltraChat...")
ultrachat = Dataset.from_list(load_jsonl_safe(Config.ultrachat_path))
ultrachat = ultrachat.map(format_ultrachat, remove_columns=ultrachat.column_names)

print("Loading medical data...")
medical = load_dataset("json", data_files=Config.medical_path, split="train")
medical = medical.map(format_medical, remove_columns=medical.column_names)

medical_repeated = concatenate_datasets([medical] * 5)
combined = concatenate_datasets([ultrachat, medical_repeated]).shuffle(seed=42)
print(f"Total training samples: {len(combined):,}")

# ============================================================
# FIX 3: Custom data collator supplying mm_token_type_ids
#
# Gemma 4 validates mm_token_type_ids in its forward pass even
# for text-only inputs. Standard collators don't produce it.
# We pad everything manually and supply zeros for both
# token_type_ids and mm_token_type_ids.
# ============================================================

def tokenize_dataset(example):
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=Config.max_length,
        padding=False,
    )
    seq_len = len(encoded["input_ids"])
    encoded["token_type_ids"]    = [0] * seq_len
    encoded["mm_token_type_ids"] = [0] * seq_len
    encoded["labels"]            = encoded["input_ids"].copy()
    return encoded

combined = combined.map(
    tokenize_dataset,
    remove_columns=["text"],
    desc="Tokenizing",
)

@dataclass
class Gemma4Collator:
    """
    Pads all fields including mm_token_type_ids which Gemma 4
    requires even for text-only batches.
    """
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id  = self.pad_token_id

        batch = {k: [] for k in
                 ["input_ids", "attention_mask",
                  "token_type_ids", "mm_token_type_ids", "labels"]}

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            batch["input_ids"].append(
                f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(
                [1] * seq_len + [0] * pad_len)
            batch["token_type_ids"].append([0] * max_len)
            batch["mm_token_type_ids"].append([0] * max_len)
            batch["labels"].append(
                f.get("labels", f["input_ids"]) + [-100] * pad_len)

        return {k: torch.tensor(v) for k, v in batch.items()}

collator = Gemma4Collator(pad_token_id=tokenizer.pad_token_id)

# =========================
# CALLBACK
# =========================

class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="Once upon a time,", max_new_tokens=50):
        self.tokenizer      = tokenizer
        self.prompt         = prompt
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 1000 == 0 and state.global_step > 0:
            m = kwargs["model"]
            m.eval()
            with torch.no_grad():
                inputs  = self.tokenizer(self.prompt, return_tensors="pt").to(m.device)
                outputs = m.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n=== Sample at step {state.global_step:,} ===\n{text}\n")
            m.train()

# =========================
# TRAINING
# Fix: max_seq_length is NOT a valid SFTConfig kwarg in newer TRL.
# dataset_text_field must be None when using a pre-tokenized dataset
# with a custom collator. remove_unused_columns=False is required
# so mm_token_type_ids isn't stripped before reaching the model.
# =========================

sft_config = SFTConfig(
    output_dir                  = Config.output_dir,
    max_steps                   = Config.max_steps,
    per_device_train_batch_size = Config.batch_size,
    gradient_accumulation_steps = Config.grad_accum_steps,
    learning_rate               = Config.lr,
    weight_decay                = Config.weight_decay,
    max_grad_norm               = 0.3,
    warmup_steps                = Config.warmup_steps,
    lr_scheduler_type           = "cosine",
    logging_steps               = Config.log_every,
    save_strategy               = "steps",
    save_steps                  = Config.eval_every,
    save_total_limit            = 2,
    fp16                        = True,
    optim                       = "paged_adamw_8bit",
    report_to                   = "tensorboard",
    dataset_text_field          = None,        # we pre-tokenized; no on-the-fly text field
    remove_unused_columns       = False,       # keep mm_token_type_ids alive
)

trainer = SFTTrainer(
    model            = model,
    args             = sft_config,
    train_dataset    = combined,
    data_collator    = collator,
    processing_class = tokenizer,
    callbacks        = [GenerateTextCallback(tokenizer)],
)

print("Starting training...")
trainer.train()

# =========================
# SAVE
# =========================

print(f"Saving LoRA adapter to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done!")