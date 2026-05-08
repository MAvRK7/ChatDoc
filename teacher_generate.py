from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os

MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_FILE = "data/teacher_responses.jsonl"
CHUNK_SIZE = 1000  # Smaller chunks for memory safety

os.makedirs("data", exist_ok=True)

# Build prompts
print("Building prompts...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

# CRITICAL FIX: Use single-process engine with enforce_eager
# This bypasses the multiprocessing worker crash
print("Loading vLLM (single-process mode)...")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,  # Keep both GPUs
    dtype="bfloat16",        # Gemma-2 requires this
    gpu_memory_utilization=0.85,
    max_model_len=608,
    enforce_eager=True,      # ← Disables CUDA graph caching, but fixes Kaggle spawn
    distributed_executor_backend="mp",  # Force multiprocessing explicitly
    # OR try: worker_use_ray=True if mp still fails
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=96,
)

# Stream to disk
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