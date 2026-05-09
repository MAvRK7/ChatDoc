import os
import json
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# =========================
# CONFIG
# =========================

MODEL_ID = "google/gemma-2b-it"

INPUT_FILE = "data/processed/train_formatted.jsonl"

# Kaggle persistent output directory
OUTPUT_DIR = "/kaggle/working/teacher_chunks"
FINAL_OUTPUT = "/kaggle/working/teacher_responses_50k.jsonl"

TARGET_EXAMPLES = 50_000
MAX_INPUT_TOKENS = 680   # Leave headroom for template + output
MAX_OUTPUT_TOKENS = 64

CHUNK_SIZE = 700         # vLLM batch size
CHUNKS_PER_FILE = 10     # Write to disk every 10 vLLM chunks (~7K items)


# =========================
# BUILD PROMPTS
# =========================

def build_prompts(tokenizer):
    prompts = []
    seen = set()

    print("Loading + filtering prompts...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
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

            messages = [{"role": "user", "content": user_text}]

            # Tokenize to check length
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            # HARD TRUNCATE: slice tokens, decode, rebuild
            if len(tokens) > MAX_INPUT_TOKENS:
                # Leave room for template overhead
                safe_limit = MAX_INPUT_TOKENS - 10
                tokens = tokens[:safe_limit]

                # Decode back to raw text
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)

                # Extract user content from Gemma template debris
                # Template wraps as: <start_of_turn>user\n{content}<end_of_turn>
                if "user" in decoded:
                    # Strip template markers
                    cleaned = decoded.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
                    if "user\n" in cleaned:
                        user_text = cleaned.split("user\n", 1)[1].strip()
                    else:
                        user_text = cleaned.strip()
                else:
                    user_text = decoded.strip()

                messages = [{"role": "user", "content": user_text}]

            # Build final prompt string
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # SAFETY CHECK: skip if still too long (shouldn't happen)
            final_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )
            if len(final_tokens) > MAX_INPUT_TOKENS:
                continue

            prompts.append(prompt)

    print(f"\nTotal unique prompts available: {len(prompts):,}")
    return prompts


# =========================
# MAIN
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    prompts = build_prompts(tokenizer)

    # Deterministic shuffle for diversity
    random.seed(42)
    random.shuffle(prompts)
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
    file_idx = 0
    items_buffer = []
    total_generated = 0

    print("\nStarting generation...")

    for chunk_num in tqdm(range(total_chunks), desc="Generating"):
        start = chunk_num * CHUNK_SIZE
        chunk = prompts[start:start + CHUNK_SIZE]

        outputs = llm.generate(chunk, sampling_params)

        for prompt, output in zip(chunk, outputs):
            response = output.outputs[0].text.strip()
            items_buffer.append({
                "prompt": prompt,
                "response": response
            })
            total_generated += 1

        # FLUSH TO DISK every N chunks
        if (chunk_num + 1) % CHUNKS_PER_FILE == 0 or chunk_num == total_chunks - 1:
            filepath = os.path.join(OUTPUT_DIR, f"chunk_{file_idx:04d}.jsonl")

            with open(filepath, "w", encoding="utf-8") as f:
                for item in items_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())  # Force physical write to disk

            # VERIFY
            file_size = os.path.getsize(filepath)
            if file_size < 100:
                raise RuntimeError(f"File write failed: {filepath} ({file_size} bytes)")

            print(f"  Saved {filepath} ({len(items_buffer)} items, {file_size:,} bytes)")
            items_buffer = []
            file_idx += 1

    # MERGE CHUNKS
    print(f"\nMerging {file_idx} chunks into final file...")
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as fout:
        for i in range(file_idx):
            chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{i:04d}.jsonl")
            with open(chunk_path, "r", encoding="utf-8") as fin:
                fout.write(fin.read())

    # VERIFY FINAL
    final_size = os.path.getsize(FINAL_OUTPUT)
    print(f"\nFinal file: {FINAL_OUTPUT}")
    print(f"Size: {final_size:,} bytes")
    print(f"Total examples: {total_generated:,}")

    # Quick sanity check
    with open(FINAL_OUTPUT, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        print(f"\nFirst prompt (truncated): {first_line['prompt'][:100]}...")
        print(f"First response (truncated): {first_line['response'][:100]}...")


if __name__ == "__main__":
    main()