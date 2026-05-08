from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os

MODEL_ID = "google/gemma-2-2b-it"
CHUNK_SIZE = 2000  # Process in chunks for crash safety
OUTPUT_FILE = "data/teacher_responses.jsonl"

os.makedirs("data", exist_ok=True)

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build prompts with dedup
print("Building prompts...")
prompts = []
seen = set()

with open("data/processed/train_formatted.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        
        if "<assistant>" not in text:
            continue
            
        user_text = text.split("<assistant>")[0]
        user_text = user_text.replace("<user>", "").strip()
        
        # Deduplicate
        if user_text in seen:
            continue
        seen.add(user_text)
        
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)

print(f"Total unique prompts: {len(prompts)}")

# Init vLLM with both T4s
print("Loading vLLM...")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,
    dtype="float16",
    gpu_memory_utilization=0.90,  # Leave headroom for Kaggle overhead
    max_model_len=608,  # 512 + 96
)

sampling_params = SamplingParams(
    temperature=0.8,    # ← Sampling for distillation
    top_p=0.95,
    top_k=50,
    max_tokens=96,
)

# Stream to disk in chunks
print("Generating...")
with open(OUTPUT_FILE, "w") as f_out:
    for i in tqdm(range(0, len(prompts), CHUNK_SIZE)):
        chunk = prompts[i:i + CHUNK_SIZE]
        outputs = llm.generate(chunk, sampling_params)
        
        for prompt, output in zip(chunk, outputs):
            response = output.outputs[0].text.strip()
            item = {"prompt": prompt, "response": response}
            f_out.write(json.dumps(item) + "\n")
            f_out.flush()

print(f"Done! Saved to {OUTPUT_FILE}")