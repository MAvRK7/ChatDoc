# production/merge/merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

BASE_MODEL = "google/gemma-4-E2B-it"
LORA_PATH = "checkpoints/gemma-lora-final"
OUTPUT_PATH = "checkpoints/gemma-merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, LORA_PATH)

print("Merging and unloading...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)

processor = AutoProcessor.from_pretrained(LORA_PATH)
processor.save_pretrained(OUTPUT_PATH)

print("Done! Merged model ready.")