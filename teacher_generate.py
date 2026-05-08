'''
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm

MODEL_ID = "google/gemma-2b-it"
OUTPUT_FILE = "data/teacher_responses.jsonl"
CHUNK_SIZE =  700

def main():
    os.makedirs("data", exist_ok=True)

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

            # Tokenize to check length, but DON'T decode back
            token_count = len(tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            ))

            if token_count > 700:
                # Truncate the text itself, not tokens
                # Rough heuristic: 3 chars per token for English
                char_limit = int(700 * 3 * 0.8)  # safety margin
                user_text = user_text[:char_limit]
                messages = [{"role": "user", "content": user_text}]

            # Pass STRING to vLLM — let vLLM tokenize correctly
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # ← String output
                add_generation_prompt=True
            )

            prompts.append(prompt)

    print(f"Total unique prompts: {len(prompts)}")

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2,
        dtype="float16",
        gpu_memory_utilization=0.85,  # ← T4s can handle 0.85 easily
        max_model_len=768,  # 700 input + 64 output + padding headroom
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=50,  # ← 50 is standard, 20 is too restrictive
        max_tokens=64,
    )

    with open(OUTPUT_FILE, "w") as f_out:
        for i in tqdm(range(0, len(prompts), CHUNK_SIZE)):
            chunk = prompts[i:i + CHUNK_SIZE]
            outputs = llm.generate(chunk, sampling_params)

            for prompt, output in zip(chunk, outputs):
                response = output.outputs[0].text.strip()
                item = {"prompt": prompt, "response": response}
                f_out.write(json.dumps(item) + "\n")

    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
'''
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_ID = "google/gemma-2b-it"
BATCH_SIZE = 20        # T4 can handle this for 2B model
MAX_NEW_TOKENS = 64
MAX_INPUT_LENGTH = 700

os.makedirs("data", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Build prompts (same logic as your vLLM version)
prompts = []
seen = set()

with open("data/processed/train_formatted.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        if "<assistant>" not in text:
            continue
        
        user_text = text.split("<assistant>")[0].replace("<user>", "").strip()
        if user_text in seen:
            continue
        seen.add(user_text)
        
        messages = [{"role": "user", "content": user_text}]
        
        # Check length, truncate text if needed
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(tokens) > MAX_INPUT_LENGTH:
            # Rough truncate: ~3 chars/token
            user_text = user_text[:int(MAX_INPUT_LENGTH * 2.5)]
            messages = [{"role": "user", "content": user_text}]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

print(f"Total unique prompts: {len(prompts)}")

# Load model across BOTH GPUs automatically
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",        # ← Splits across GPU 0 + 1
    max_memory={0: "14GiB", 1: "14GiB"},  # Leave headroom for Kaggle
)
model.eval()

# Generate in batches
print("Generating...")
with open("data/teacher_responses.jsonl", "w") as f_out:
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(model.device if not hasattr(model, 'hf_device_map') else 'cuda:0')
        
        # If device_map split the model, inputs must be on GPU 0
        # (accelerate handles the rest)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_lengths = inputs["attention_mask"].sum(dim=1)
        
        for j, prompt in enumerate(batch_prompts):
            gen_ids = outputs[j][input_lengths[j]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            f_out.write(json.dumps({
                "prompt": prompt,
                "response": response
            }) + "\n")
        
        # NO torch.cuda.empty_cache() — let PyTorch manage it

print("Done!")