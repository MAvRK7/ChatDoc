import os
import json
import gc
import glob
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
    output_dir       = "checkpoints/gemma-lora"       # trainer saves here every eval_every steps
    final_dir        = "checkpoints/gemma-lora-final" # manual adapter save at end
    tokenized_cache  = "cache/tokenized_dataset"

    batch_size        = 1
    grad_accum_steps  = 8
    max_length        = 256

    lr            = 2e-4
    weight_decay  = 0.01
    warmup_steps  = 50
    max_steps     = 10000   # high ceiling — resume will pick up from last checkpoint

    eval_every    = 200     # save every 200 steps so you never lose more than ~30 min
    save_total    = 5       # keep last 5 checkpoints
    log_every     = 10

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
# DATASET — cache-aware
# Checks for state.json which is written LAST by save_to_disk,
# so its presence guarantees a fully written, loadable dataset.
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
        # no num_proc — Kaggle multiprocessing deadlocks
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
    print("No valid cache found — building dataset from scratch...")
    combined = build_and_cache_dataset()

# =========================
# AUTO-RESUME: find latest checkpoint
# Scans output_dir for checkpoint-N folders and returns the highest N.
# The Trainer's resume_from_checkpoint=True does the same thing but
# only works if output_dir already has checkpoints — this makes it
# explicit and prints clearly what's happening.
# =========================

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.split("-")[-1])
    )
    latest = checkpoints[-1]
    print(f"Found {len(checkpoints)} checkpoint(s). Resuming from: {latest}")
    return latest

resume_from = find_latest_checkpoint(Config.output_dir)
if resume_from is None:
    print("No checkpoint found — training from scratch.")

# =========================
# MODEL
# device_map="auto" splits layers across both T4s (model parallelism).
# This gives 32GB combined VRAM headroom instead of 16GB.
# It does NOT give 2x training speed — that needs DDP/FSDP which is
# incompatible with bitsandbytes 4-bit in a Kaggle notebook.
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
    device_map="auto",          # splits across both T4s automatically
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
# FREEZE + BF16 PURGE
# =========================

for param in model.parameters():
    param.requires_grad = False

# Upcast layernorms to float32 for stability (they are tiny — KB not GB)
for name, param in model.named_parameters():
    if "norm" in name:
        param.data = param.data.to(torch.float32)

# Full bf16 purge — catches any straggler tensors from HF loader
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
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

gc.collect()
torch.cuda.empty_cache()

# =========================
# LORA TARGET DISCOVERY
# =========================

linear_names = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_names.add(name.split(".")[-1])

print(f"All linear leaf names: {sorted(linear_names)}")

LORA_TARGETS = [
    n for n in linear_names
    if any(kw in n for kw in [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
]

if not LORA_TARGETS:
    LORA_TARGETS = [n for n in linear_names if n not in {"lm_head", "embed_tokens"}]
    print(f"Fallback LoRA targets: {sorted(LORA_TARGETS)}")
else:
    print(f"LoRA targets: {sorted(LORA_TARGETS)}")

# =========================
# LORA
# =========================

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

bad = [(n, p.dtype) for n, p in model.named_parameters()
       if p.requires_grad and p.dtype == torch.bfloat16]
print(f"BF16 trainable leak: {bad}" if bad else "All trainable params are float16.")

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
# GENERATION CALLBACK
# Fixed: supplies mm_token_type_ids and uses a proper chat-formatted
# prompt so the model actually generates coherent text.
# The comma-repeating bug happened because the raw prompt "Once upon
# a time," doesn't match the chat template the model was trained on.
# =========================

class GenerateTextCallback(TrainerCallback):

    PROMPT = [
        {"role": "user", "content": "What should I do if I have a fever?"}
    ]

    def __init__(self, tokenizer, max_new_tokens=80):
        self.tokenizer      = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompt_text    = tokenizer.apply_chat_template(
            self.PROMPT,
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
                    # Gemma 4 requires these even for text-only inference
                    inputs["token_type_ids"]    = torch.zeros_like(inputs["input_ids"])
                    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
                    outputs = m.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,          # greedy — more stable for eval
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    # Decode only the newly generated tokens
                    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"\n=== Generation at step {state.global_step:,} ===")
                print(f"Q: What should I do if I have a fever?")
                print(f"A: {text}\n")
            except Exception as e:
                print(f"Generation failed at step {state.global_step}: {e}")
            finally:
                m.train()

# =========================
# TRAINING
#
# fp16=False, bf16=False: disables HF AMP loss scaler entirely.
# This is required because bitsandbytes 4-bit dequantization emits
# bf16 activations internally on certain bnb versions, which then
# hit the T4's missing _amp_foreach_non_finite_check_and_unscale_cuda
# CUDA kernel. Without AMP, bitsandbytes manages its own precision.
# Cost: ~10-15% slower per step vs fp16 AMP. Benefit: it actually runs.
#
# resume_from_checkpoint: automatically picks up from latest checkpoint.
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
    fp16                        = False,   # see note above
    bf16                        = False,   # see note above
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

# =========================
# SAVE FINAL ADAPTER
# Saves only the LoRA adapter weights — small and fast.
# Use merge.py in a fresh session to produce the merged model.
# =========================

print(f"Saving final LoRA adapter to {Config.final_dir}...")
model.save_pretrained(Config.final_dir)
tokenizer.save_pretrained(Config.final_dir)
print("Done! Run merge.py in a fresh session to produce the full merged model.")