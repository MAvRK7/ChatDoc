# production/train/qlora_train.py
import os
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from trl import SFTTrainer, SFTConfig
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
    Gemma4ForCausalLM
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset

MODEL_ID = "google/gemma-4-E2B-it"

# =========================
# 1. CONFIG
# =========================

class Config:
    # Data paths
    ultrachat_path = "data/processed/train.jsonl"
    medical_path = "data/finetune/train_deduped.jsonl"
    
    # Output
    output_dir = "checkpoints/gemma-lora"
    final_dir = "checkpoints/gemma-lora-final"
    log_dir = "outputs/runs"
    
    # Training
    batch_size = 1
    grad_accum_steps = 8
    max_length = 1024
    lr = 2e-4
    weight_decay = 0.01
    warmup_steps = 100
    max_steps = 5000
    eval_every = 500
    log_every = 25
    
    # LoRA
    lora_r = 32
    lora_alpha = 16
    lora_dropout = 0.05

# =========================
# 2. LOAD MODEL (4-bit)
# =========================
print("Loading Gemma 4-E2B...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Gemma4ForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True
)

# FIX: Modern checkpointing & text-only mode
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.config.vision_config = None 

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# 3. LoRA SETUP
# =========================
lora_config = LoraConfig(
    r=Config.lora_r,
    lora_alpha=Config.lora_alpha,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=Config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# Deep patch for Gemma 4 native layers
for name, module in model.named_modules():
    if any(target in name for target in lora_config.target_modules):
        if hasattr(module, "weight"):
            if not hasattr(module.weight, "compress_statistics"):
                module.weight.compress_statistics = None
            if not hasattr(module.weight, "quant_type"):
                module.weight.quant_type = "nf4"
            if not hasattr(module.weight, "quant_state"):
                module.weight.quant_state = None

model = get_peft_model(model, lora_config)

# Manual preparation & weight tying
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
        param.data = param.data.to(torch.bfloat16)

model.tie_weights() # FIX: Keeps GPUs in sync
model.print_trainable_parameters()

# =========================
# 4. LOAD & FORMAT DATA
# =========================

def format_ultrachat(example):
    messages = example["messages"]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def format_medical(example):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": example["instruction"]}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": example["input"]}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["output"]}]
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def load_jsonl_safe(path):
    """Load JSONL, skipping malformed lines and reporting them."""
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

print("Loading UltraChat (with bad-line recovery)...")
raw_data = load_jsonl_safe(Config.ultrachat_path)
ultrachat = Dataset.from_list(raw_data)
ultrachat = ultrachat.map(format_ultrachat, remove_columns=ultrachat.column_names)

print("Loading medical data...")
medical = load_dataset("json", data_files=Config.medical_path, split="train")
medical = medical.map(format_medical, remove_columns=medical.column_names)

# Oversample medical 5× for balance
print("Combining datasets...")
medical_repeated = concatenate_datasets([medical] * 5)
combined = concatenate_datasets([ultrachat, medical_repeated])
combined = combined.shuffle(seed=42)

print(f"Total training samples: {len(combined):,}")

# =========================
# 5. CALLBACK FOR INFERENCE
# =========================

class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="Once upon a time,", max_new_tokens=50):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 1000 == 0 and state.global_step > 0:
            model = kwargs["model"]
            model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n=== Sample generation at step {state.global_step:,} ===\n{text}\n")
            
# =========================
# 6. TRAINING
# =========================

sft_config = SFTConfig(
    output_dir=Config.output_dir,
    max_steps=Config.max_steps,
    per_device_train_batch_size=Config.batch_size,
    gradient_accumulation_steps=Config.grad_accum_steps,
    learning_rate=Config.lr,
    max_grad_norm=0.3,
    warmup_steps=Config.warmup_steps,
    lr_scheduler_type="cosine",
    logging_steps=Config.log_every,
    save_strategy="steps",
    save_steps=Config.eval_every,
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="tensorboard",
    dataset_text_field="text",
    # NOTHING ELSE — no max_seq_length, no packing
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=combined,
    processing_class=tokenizer,
    callbacks=[GenerateTextCallback(tokenizer, prompt="Once upon a time,")]
    # NO max_seq_length here either
)

# Note: We removed 'formatting_func' because 'dataset_text_field' inside 
# SFTConfig is the cleaner way to handle your format.

print("Starting training...")
trainer.train()

# =========================
# 7. SAVE
# =========================

print(f"Saving LoRA adapter to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done!")