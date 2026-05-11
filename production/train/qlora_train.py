import os
import json
import gc
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from trl import SFTTrainer, SFTConfig
from transformers import (
    BitsAndBytesConfig,
    TrainerCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets, Dataset

# =========================
# CONFIG
# =========================

MODEL_ID = "google/gemma-4-E2B-it"   # correct casing, confirmed on HF

class Config:
    ultrachat_path   = "data/processed/train.jsonl"
    medical_path     = "data/finetune/train_deduped.jsonl"
    output_dir       = "checkpoints/gemma-lora"
    final_dir        = "checkpoints/gemma-lora-final"

    batch_size        = 1
    grad_accum_steps  = 16
    max_length        = 512

    lr            = 2e-4
    weight_decay  = 0.01
    warmup_steps  = 100
    max_steps     = 5000

    eval_every    = 500
    log_every     = 25

    lora_r        = 8
    lora_alpha    = 16
    lora_dropout  = 0.05

# =========================
# MODEL
# =========================

print("Loading Gemma 4 E2B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token        = tokenizer.eos_token
tokenizer.padding_side     = "right"
tokenizer.model_max_length = Config.max_length

# =========================
# LORA
# Key fix: instead of naming target_modules explicitly (which hits
# Gemma4ClippableLinear wrappers), we pass "all-linear" so PEFT
# walks the module tree itself and only wraps actual nn.Linear leaves.
# =========================

lora_config = LoraConfig(
    r              = Config.lora_r,
    lora_alpha     = Config.lora_alpha,
    target_modules = "all-linear",   # ← avoids the ClippableLinear error
    lora_dropout   = Config.lora_dropout,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
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
    fp16                        = True,   # T4 doesn't natively support bf16, use fp16
    optim                       = "paged_adamw_8bit",
    report_to                   = "tensorboard",
    dataset_text_field          = "text",
    max_seq_length              = Config.max_length,
)

trainer = SFTTrainer(
    model            = model,
    args             = sft_config,
    train_dataset    = combined,
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