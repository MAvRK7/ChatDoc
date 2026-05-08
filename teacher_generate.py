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