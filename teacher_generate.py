import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "google/gemma-2b-it"

INPUT_FILE = "data/processed/train_formatted.jsonl"
OUTPUT_FILE = "data/teacher_responses_50k.jsonl"

TARGET_EXAMPLES = 50_000
MAX_INPUT_TOKENS = 700
MAX_OUTPUT_TOKENS = 64

CHUNK_SIZE = 700

def build_prompts(tokenizer):
    prompts = []
    seen = set()

    print("Loading + filtering prompts...")

    with open(INPUT_FILE, "r") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            text = obj.get("text", "")

            if "<assistant>" not in text:
                continue

            user_text = text.split("<assistant>")[0]
            user_text = user_text.replace("<user>", "").strip()

            if not user_text:
                continue

            # Deduplicate
            if user_text in seen:
                continue

            seen.add(user_text)

            messages = [
                {
                    "role": "user",
                    "content": user_text
                }
            ]

            # Count tokens
            token_count = len(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True
                )
            )

            # Truncate if needed
            if token_count > MAX_INPUT_TOKENS:

                # ~4 chars/token heuristic for English
                char_limit = int(MAX_INPUT_TOKENS * 4)

                user_text = user_text[:char_limit]

                messages = [
                    {
                        "role": "user",
                        "content": user_text
                    }
                ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts.append(prompt)

    print(f"\nTotal unique prompts available: {len(prompts):,}")

    return prompts

def main():
    os.makedirs("data", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    prompts = build_prompts(tokenizer)

    # Deterministic shuffle
    random.seed(42)
    random.shuffle(prompts)

    # Limit to 50k
    prompts = prompts[:TARGET_EXAMPLES]

    print(f"\nUsing {len(prompts):,} prompts for generation")

    print("\nLoading vLLM model...")

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=768,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    total_chunks = (len(prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE

    print("\nStarting generation...")

    generated = 0

    with open(OUTPUT_FILE, "w") as f_out:

        for i in tqdm(
            range(0, len(prompts), CHUNK_SIZE),
            total=total_chunks,
            desc="Generating"
        ):

            chunk = prompts[i:i + CHUNK_SIZE]

            outputs = llm.generate(chunk, sampling_params)

            for prompt, output in zip(chunk, outputs):

                try:
                    response = output.outputs[0].text.strip()

                    item = {
                        "prompt": prompt,
                        "response": response
                    }

                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

                    generated += 1

                except Exception as e:
                    print(f"Skipping failed output: {e}")

    print(f"\nDone!")
    print(f"Generated examples: {generated:,}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()