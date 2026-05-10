# production/train/qlora_train.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from trl import SFTTrainer, SFTConfig
import torch
from transformers import (
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    Gemma4ForCausalLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets

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

print("Loading UltraChat...")
ultrachat = load_dataset("json", data_files=Config.ultrachat_path, split="train")
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

# 1. Set the length limit directly on the tokenizer to avoid the SFTTrainer argument
tokenizer.model_max_length = Config.max_length
tokenizer.padding_side = "right"

training_args = TrainingArguments(
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
    logging_dir=Config.log_dir,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

# SFTTrainer expects the formatting_func to return the text
def formatting_prompts_func(example):
    return example["text"]

# We initialize without 'max_seq_length' to stop the TypeError.
# The trainer will fall back to tokenizer.model_max_length (which we set above).
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=combined,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    # REMOVED: max_seq_length
    # REMOVED: packing
    dataset_kwargs={
        "add_special_tokens": False,
    },
    callbacks=[GenerateTextCallback(tokenizer, prompt="Once upon a time,")]
)

print("Starting training...")
trainer.train()

# =========================
# 7. SAVE
# =========================

print(f"Saving LoRA adapter to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done!")