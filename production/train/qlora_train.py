import os
import json
import gc
import glob
import torch
torch.backends.cudnn.benchmark = True
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
# CONFIG — same values as the run that worked at 9s/step
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
    max_length        = 128

    lr            = 2e-4
    weight_decay  = 0.01
    warmup_steps  = 50
    max_steps     = 3000   # ~7.5 hrs at 9s/step; kill manually when done

    eval_every    = 500    # checkpoint every ~30 min
    save_total    = 5
    log_every     = 20

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
# DATASET — skips tokenization if valid cache exists
# Uses state.json as the completion marker (written last by save_to_disk)
# =========================

def cache_is_valid(path):
    return os.path.isfile(os.path.join(path, "state.json"))

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
                    print(f"Warning: skipping bad line {i}: {e}")
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
    )

    os.makedirs(Config.tokenized_cache, exist_ok=True)
    combined.save_to_disk(Config.tokenized_cache)
    print(f"Saved tokenized dataset to {Config.tokenized_cache}")
    return combined

if cache_is_valid(Config.tokenized_cache):
    print(f"Valid cache found — loading from {Config.tokenized_cache}...")
    combined = Dataset.load_from_disk(Config.tokenized_cache)
    print(f"Loaded {len(combined):,} samples. Skipping tokenization.")
else:
    print("No valid cache — building dataset...")
    combined = build_and_cache_dataset()

# =========================
# AUTO-RESUME
# Finds highest-numbered checkpoint-N in output_dir.
# Passed to trainer.train() so optimizer state + step count resume too.
# =========================

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    latest = ckpts[-1]
    print(f"Resuming from checkpoint: {latest}")
    return latest

resume_from = find_latest_checkpoint(Config.output_dir)
if resume_from is None:
    print("No checkpoint found — training from scratch.")

# =========================
# MODEL
# =========================

print("Loading Gemma 4 E2B...")

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

# =========================
# UNWRAP ClippableLinear
# =========================

def unwrap_clippable_linears(model):
    try:
        from transformers.models.gemma4 import modeling_gemma4
        ClippableLinear = getattr(modeling_gemma4, "Gemma4ClippableLinear", None)
    except ImportError:
        ClippableLinear = None
    if ClippableLinear is None:
        print("Gemma4ClippableLinear not found — skipping.")
        return model
    replaced = 0
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, ClippableLinear):
                setattr(parent, child_name, child.linear)
                replaced += 1
    print(f"Unwrapped {replaced} Gemma4ClippableLinear modules.")
    return model

model = unwrap_clippable_linears(model)

# =========================
# FREEZE + BF16 PURGE
# =========================

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "norm" in name:
        param.data = param.data.to(torch.float32)

bf16_count = 0
for _, param in model.named_parameters():
    if param.data.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)
        bf16_count += 1
for _, buf in model.named_buffers():
    if buf.dtype == torch.bfloat16:
        buf.data = buf.data.to(torch.float16)
        bf16_count += 1
print(f"BF16 audit: {'converted ' + str(bf16_count) + ' tensors' if bf16_count else 'clean'}.")

model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

gc.collect()
torch.cuda.empty_cache()

# =========================
# LORA
# =========================

linear_names = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_names.add(name.split(".")[-1])

LORA_TARGETS = [
    n for n in linear_names
    if any(kw in n for kw in ["q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj"])
]
if not LORA_TARGETS:
    LORA_TARGETS = [n for n in linear_names if n not in {"lm_head","embed_tokens"}]

print(f"LoRA targets: {sorted(LORA_TARGETS)}")

lora_config = LoraConfig(
    r              = Config.lora_r,
    lora_alpha     = Config.lora_alpha,
    target_modules = LORA_TARGETS,
    lora_dropout   = Config.lora_dropout,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
        param.data = param.data.to(torch.float16)

model.print_trainable_parameters()
gc.collect()
torch.cuda.empty_cache()

# =========================
# COLLATOR
# =========================

@dataclass
class Gemma4Collator:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id  = self.pad_token_id
        batch = {k: [] for k in
                 ["input_ids","attention_mask","token_type_ids","mm_token_type_ids","labels"]}
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append([1]*seq_len + [0]*pad_len)
            batch["token_type_ids"].append([0] * max_len)
            batch["mm_token_type_ids"].append([0] * max_len)
            batch["labels"].append(f.get("labels", f["input_ids"]) + [-100]*pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}

collator = Gemma4Collator(pad_token_id=tokenizer.pad_token_id)

# =========================
# CALLBACK — fixed generation
# Uses proper chat template + decodes only new tokens
# so output is coherent instead of repeating commas
# =========================

class GenerateTextCallback(TrainerCallback):

    def __init__(self, tokenizer, max_new_tokens=80):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # Pre-format a medical question through the chat template
        self.prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What should I do if I have a fever?"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 200 == 0 and state.global_step > 0:
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
                    inputs["token_type_ids"]    = torch.zeros_like(inputs["input_ids"])
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
                print(f"\n=== Step {state.global_step:,} | Q: fever? ===\nA: {text}\n")
            except Exception as e:
                print(f"Generation skipped at step {state.global_step}: {e}")
            finally:
                m.train()

# =========================
# TRAINING
# fp16/bf16 both False — disables HF AMP scaler which crashes on T4
# when bnb 4-bit dequant emits bf16 activations internally
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
    save_total_limit            = Config.save_total,
    fp16                        = False,
    bf16                        = False,
    optim                       = "paged_adamw_8bit",
    report_to                   = "tensorboard",
    dataset_text_field          = None,
    remove_unused_columns       = False,
    dataloader_pin_memory       = False,
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
trainer.train(resume_from_checkpoint=resume_from)

print(f"Saving LoRA adapter to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done!")