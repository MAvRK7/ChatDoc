import os
import json
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_ID = "google/gemma-2b-it"
MAX_SAMPLES = 30000  # Start with 30K, increase later if needed
MAX_NEW_TOKENS = 256

def worker(gpu_id, prompt_chunk, output_file):
    device = f"cuda:{gpu_id}"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    model.eval()
    
    results = []
    for prompt in tqdm(prompt_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<assistant>" in full_text:
            response = full_text.split("<assistant>")[-1].strip()
        else:
            response = full_text[len(prompt):].strip()
        
        results.append({"prompt": prompt, "response": response})
    
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

def main():
    # Load prompts from your formatted data
    prompts = []
    with open("data/processed/train_formatted.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if "<assistant>" in text:
                prompt = text.split("<assistant>")[0] + "<assistant> "
                prompts.append(prompt)
    
    # Deduplicate and sample
    prompts = list(set(prompts))
    if len(prompts) > MAX_SAMPLES:
        import random
        random.seed(42)
        prompts = random.sample(prompts, MAX_SAMPLES)
    
    print(f"Generating teacher data for {len(prompts)} prompts...")
    
    # Split across 2 GPUs
    mid = len(prompts) // 2
    chunk0 = prompts[:mid]
    chunk1 = prompts[mid:]
    
    # Use spawn for CUDA compatibility
    ctx = mp.get_context("spawn")
    p0 = ctx.Process(target=worker, args=(0, chunk0, "data/teacher_gpu0.jsonl"))
    p1 = ctx.Process(target=worker, args=(1, chunk1, "data/teacher_gpu1.jsonl"))
    
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    
    # Merge
    with open("data/teacher_responses.jsonl", "w") as f_out:
        for fname in ["data/teacher_gpu0.jsonl", "data/teacher_gpu1.jsonl"]:
            if os.path.exists(fname):
                with open(fname) as f_in:
                    for line in f_in:
                        f_out.write(line)
                os.remove(fname)
    
    print("Done! Saved to data/teacher_responses.jsonl")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()