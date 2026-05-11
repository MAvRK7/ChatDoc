import os
import json
import gc
import torch
import torch.nn as nn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# CRITICAL FIX: Patch Gemma4ClippableLinear BEFORE model loading
# This must run before ANY transformers import that triggers model loading
# =========================================================

# We need to patch the modeling module BEFORE it's loaded from Hub
# The module will be loaded dynamically via trust_remote_code
# We patch it by pre-registering a custom loader

_original_from_pretrained = None

def patch_gemma4_before_load():
    """Monkey-patch the Gemma4 modeling file before it's executed."""
    import transformers.models.gemma4.modeling_gemma4 as modeling_module
    
    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features, **kwargs):
            nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))
        
        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out
    
    modeling_module.Gemma4ClippableLinear = PatchedClippableLinear

# Alternative: Use import hook to patch as soon as module loads
import sys
import importlib.abc
import importlib.machinery

class Gemma4PatchFinder(importlib.abc.MetaPathFinder):
    def find_module(self, fullname, path=None):
        if fullname == 'transformers.models.gemma4.modeling_gemma4':
            return self
        return None
    
    def find_spec(self, fullname, path, target=None):
        if fullname == 'transformers.models.gemma4.modeling_gemma4':
            # Let normal loader handle it, we'll patch after
            return None
        return None
    
    def exec_module(self, module):
        pass  # Not used

# Actually, simpler approach: patch after first import triggers load
# But since trust_remote_code loads from Hub, not local, we need different strategy

# NUCLEAR OPTION: Override the modeling file execution
import transformers.dynamic_module_utils

_original_get_class = transformers.dynamic_module_utils.get_class_from_dynamic_module

def patched_get_class(*args, **kwargs):
    result = _original_get_class(*args, **kwargs)
    # If this returns Gemma4ClippableLinear, replace it
    if isinstance(result, type) and result.__name__ == 'Gemma4ClippableLinear':
        class PatchedClippableLinear(nn.Linear):
            def __init__(self, config, in_features, out_features, **kwargs):
                nn.Linear.__init__(self, in_features, out_features, bias=False)
                self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
                if self.use_clipped_linears:
                    self.register_buffer("input_min", torch.tensor(-float("inf")))
                    self.register_buffer("input_max", torch.tensor(float("inf")))
                    self.register_buffer("output_min", torch.tensor(-float("inf")))
                    self.register_buffer("output_max", torch.tensor(float("inf")))
            
            def forward(self, x):
                if self.use_clipped_linears:
                    x = torch.clamp(x, self.input_min, self.input_max)
                out = nn.Linear.forward(self, x)
                if self.use_clipped_linears:
                    out = torch.clamp(out, self.output_min, self.output_max)
                return out
        
        return PatchedClippableLinear
    return result

transformers.dynamic_module_utils.get_class_from_dynamic_module = patched_get_class

# =========================================================
# END CRITICAL FIX
# =========================================================

from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
    Gemma4ForCausalLM
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset
import bitsandbytes as bnb

# PATCH: Fix Int8Params constructor for PEFT compatibility
_original_int8_new = bnb.nn.Int8Params.__new__

def patched_int8_new(cls, data, requires_grad=False, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return _original_int8_new(cls, data, requires_grad=requires_grad, **kwargs)

bnb.nn.Int8Params.__new__ = staticmethod(patched_int8_new)

MODEL_ID = "google/gemma-4-E2B-it"

# =========================
# 1. CONFIG
# =========================

class Config:
    ultrachat_path = "data/processed/train.jsonl"
    medical_path = "data/finetune/train_deduped.jsonl"
    output_dir = "checkpoints/gemma-lora"
    final_dir = "checkpoints/gemma-lora-final"
    log_dir = "outputs/runs"
    batch_size = 1
    grad_accum_steps = 16
    max_length = 128
    lr = 2e-4
    weight_decay = 0.01
    warmup_steps = 100
    max_steps = 5000
    eval_every = 500
    log_every = 25
    lora_r = 4
    lora_alpha = 4
    lora_dropout = 0.05

# =========================
# 2. LOAD MODEL (8-bit)
# =========================
print("Loading Gemma 4-E2B...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"]
)

model = Gemma4ForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="balanced_low_0",
    torch_dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True
)

# Drop vision and audio towers (text-only training)
model.vision_tower = None
model.audio_tower = None
model.config.vision_config = None
model.config.audio_config = None

gc.collect()
torch.cuda.empty_cache()

# Manual prep (no prepare_model_for_kbit_training to avoid OOM)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
model.config.use_cache = False

torch.cuda.empty_cache()

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = Config.max_length

# =========================
# 3. LoRA SETUP
# =========================
lora_config = LoraConfig(
    r=Config.lora_r,
    lora_alpha=Config.lora_alpha,
    target_modules=["q_proj", "v_proj"],  # Text decoder only
    lora_dropout=Config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
        param.data = param.data.to(torch.bfloat16)

model.print_trainable_parameters()
gc.collect()
torch.cuda.empty_cache()

# =========================
# 4. LOAD & FORMAT DATA
# =========================
def format_ultrachat(example):
    messages = example["messages"]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

def format_medical(example):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": example["instruction"]}]},
        {"role": "user", "content": [{"type": "text", "text": example["input"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["output"]}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
                print(f"⚠️  Skipping bad line {i}: {e}")
    return data

print("Loading UltraChat (with bad-line recovery)...")
raw_data = load_jsonl_safe(Config.ultrachat_path)
ultrachat = Dataset.from_list(raw_data)
ultrachat = ultrachat.map(format_ultrachat, remove_columns=ultrachat.column_names)

print("Loading medical data...")
medical = load_dataset("json", data_files=Config.medical_path, split="train")
medical = medical.map(format_medical, remove_columns=medical.column_names)

medical_repeated = concatenate_datasets([medical] * 5)
combined = concatenate_datasets([ultrachat, medical_repeated])
combined = combined.shuffle(seed=42)
print(f"Total training samples: {len(combined):,}")

# =========================
# 5. CALLBACK
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
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=True, temperature=0.7, top_p=0.9)
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n=== Sample at step {state.global_step:,} ===\n{text}\n")

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
    max_length=Config.max_length,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=combined,
    processing_class=tokenizer,
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