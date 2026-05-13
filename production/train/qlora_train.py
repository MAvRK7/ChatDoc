import os
import json
import gc
import torch
from dataclasses import dataclass

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

# =========================
# CONFIG
# =========================

MODEL_ID = "google/gemma-4-E2B-it"

class Config:
    ultrachat_path   = "data/processed/train.jsonl"
    medical_path     = "data/finetune/train_deduped.jsonl"
    output_dir       = "checkpoints/gemma-lora"
    final_dir        = "checkpoints/gemma-lora-final"
    tokenized_cache  = "cache/tokenized_dataset"

    batch_size        = 1
    grad_accum_steps  = 8
    # 256 is a good compromise: fits T4 VRAM in fp16, doesn't truncate
    # most medical QA pairs (64 was cutting most answers mid-sentence)
    max_length        = 256

    lr            = 2e-4
    weight_decay  = 0.01
    warmup_steps  = 50       # shorter warmup for shorter run
    # ~5-6 hours on T4: ~1 step/sec in fp16 at batch=1, accum=8
    # so ~18-21k steps/hour. But QLoRA is slower — realistic ~600-900 steps/hour.
    # 3000 steps ≈ 4-5 hours, safe within your 5-6hr window.
    max_steps     = 3000

    eval_every    = 500
    log_every     = 10      # more frequent so you can see loss moving

    lora_r        = 4
    lora_alpha    = 8
    lora_dropout  = 0.05

# =========================
# TOKENIZER
# =========================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token        = tokenizer.eos_token
tokenizer.padding_side     = "right"
tokenizer.model_max_length = Config.max_length

# =========================
# DATASET (cached)
# =========================

def build_and_cache_dataset():

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

    print("Tokenizing...")
    combined = combined.map(
        tokenize_dataset,
        remove_columns=["text"],
        desc="Tokenizing",
        # num_proc intentionally omitted — Kaggle multiprocessing deadlocks
    )

    os.makedirs(os.path.dirname(Config.tokenized_cache) or "cache", exist_ok=True)
    combined.save_to_disk(Config.tokenized_cache)
    print(f"Saved tokenized dataset to {Config.tokenized_cache}")
    return combined

if os.path.exists(Config.tokenized_cache):
    print(f"Loading cached tokenized dataset from {Config.tokenized_cache}...")
    combined = Dataset.load_from_disk(Config.tokenized_cache)
    print(f"Loaded {len(combined):,} samples from cache.")
else:
    combined = build_and_cache_dataset()

# =========================
# MODEL
# =========================

print("Loading Gemma 4 E2B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # KEY FIX: compute dtype must be float16, NOT bfloat16.
    # T4 does not support bfloat16 in CUDA amp — this was the root
    # cause of "_amp_foreach_non_finite_check_and_unscale_cuda" error.
    # bfloat16 leaked from here into the loss scaler and crashed.
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

# =========================
# UNWRAP ClippableLinear
# Load first (weights land correctly), then replace wrappers with
# their inner .linear so PEFT sees plain nn.Linear everywhere.
# =========================

def unwrap_clippable_linears(model):
    try:
        from transformers.models.gemma4 import modeling_gemma4
        ClippableLinear = getattr(modeling_gemma4, "Gemma4ClippableLinear", None)
    except ImportError:
        ClippableLinear = None

    if ClippableLinear is None:
        print("Gemma4ClippableLinear not found — skipping unwrap.")
        return model

    replaced = 0
    for _, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if isinstance(child_module, ClippableLinear):
                setattr(parent_module, child_name, child_module.linear)
                replaced += 1

    print(f"Unwrapped {replaced} Gemma4ClippableLinear modules.")
    return model

model = unwrap_clippable_linears(model)

# =========================
# MANUAL KBIT PREP (avoids OOM from prepare_model_for_kbit_training)
# =========================

# Freeze all base weights
for param in model.parameters():
    param.requires_grad = False

# Upcast ONLY layernorms to float32 — critical for training stability,
# and safe because they're tiny (KB not GB).
# NOTE: do NOT upcast to bfloat16 — that's what caused the amp error.
for name, param in model.named_parameters():
    if "norm" in name:
        param.data = param.data.to(torch.float32)

model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

gc.collect()
torch.cuda.empty_cache()

# =========================
# LORA
# =========================

lora_config = LoraConfig(
    r            = Config.lora_r,
    lora_alpha   = Config.lora_alpha,
    lora_dropout = Config.lora_dropout,
    bias         = "none",
    task_type    = "CAUSAL_LM",
    # No target_modules — PEFT >= 0.19.0 uses Gemma 4 defaults
    # (LM layers only, scoped via regex, skips ClippableLinear remnants)
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
# COLLATOR
# Gemma 4 requires mm_token_type_ids in every forward pass,
# even text-only. Standard collators don't produce it — we do.
# =========================

@dataclass
class Gemma4Collator:
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
# mm_token_type_ids must be supplied here too or generate() will
# error / produce garbage mid-training.
# =========================

class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="Once upon a time,", max_new_tokens=50):
        self.tokenizer      = tokenizer
        self.prompt         = prompt
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0 and state.global_step > 0:
            m = kwargs["model"]
            m.eval()
            with torch.no_grad():
                inputs = self.tokenizer(
                    self.prompt, return_tensors="pt"
                ).to(m.device)
                # Supply required Gemma 4 fields for text-only generation
                inputs["token_type_ids"]    = torch.zeros_like(inputs["input_ids"])
                inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
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
    save_total_limit            = 3,
    # fp16=True: correct for T4. Do NOT use bf16 — T4 has no native bf16
    # and it causes the "_amp_foreach_non_finite_check_and_unscale_cuda" crash.
    fp16                        = True,
    optim                       = "paged_adamw_8bit",
    report_to                   = "tensorboard",
    dataset_text_field          = None,        # dataset is pre-tokenized
    remove_unused_columns       = False,       # keep mm_token_type_ids alive
    dataloader_pin_memory       = False,       # prevents stalls on Kaggle T4
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
# SAVE — adapter only (no merge here, merge separately to avoid OOM)
# merge_and_unload() dequantizes the full model into fp16 in-place;
# on a 15GB T4 with base model loaded that will OOM. Use merge.py instead.
# =========================

print(f"Saving LoRA adapter to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done! Run merge.py offline to produce the merged checkpoint.")