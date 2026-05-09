import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import random
from tqdm import tqdm

MODEL_ID = "google/gemma-2b-it"

INPUT_FILE = "data/processed/train_formatted.jsonl"
OUTPUT_FILE = "data/teacher_responses_50k.jsonl"

CHUNK_SIZE = 700
TARGET_EXAMPLES = 50_000

# SAFE CONTEXT SETTINGS
MAX_INPUT_TOKENS = 680
MAX_OUTPUT_TOKENS = 64
MAX_MODEL_LEN = 768


def build_prompts(tokenizer):

    prompts = []
    seen = set()

    with open(INPUT_FILE, "r") as f:

        for line in tqdm(f, desc="Building prompts"):

            obj = json.loads(line)
            text = obj.get("text", "")

            if "<assistant>" not in text:
                continue

            user_text = text.split("<assistant>")[0]
            user_text = user_text.replace("<user>", "").strip()

            if not user_text:
                continue

            # Remove duplicates
            if user_text in seen:
                continue

            seen.add(user_text)

            # Tokenize raw user text
            user_tokens = tokenizer.encode(
                user_text,
                add_special_tokens=False
            )

            # Leave room for template overhead
            max_user_tokens = MAX_INPUT_TOKENS - 10

            if len(user_tokens) > max_user_tokens:

                user_tokens = user_tokens[:max_user_tokens]

                user_text = tokenizer.decode(
                    user_tokens,
                    skip_special_tokens=True
                )

            messages = [
                {
                    "role": "user",
                    "content": user_text
                }
            ]

            # Build final chat-formatted prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Final verification
            final_len = len(
                tokenizer.encode(
                    prompt,
                    add_special_tokens=False
                )
            )

            # Skip edge cases
            if final_len > MAX_INPUT_TOKENS:
                continue

            prompts.append(prompt)

    print(f"\nTotal unique prompts available: {len(prompts):,}")

    return prompts


def main():

    os.makedirs("data", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    prompts = build_prompts(tokenizer)

    # Deterministic shuffle
    random.seed(42)
    random.shuffle(prompts)

    # Limit dataset size
    prompts = prompts[:TARGET_EXAMPLES]

    print(f"Using {len(prompts):,} prompts")

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    with open(OUTPUT_FILE, "w") as f_out:

        for i in tqdm(
            range(0, len(prompts), CHUNK_SIZE),
            desc="Generating"
        ):

            chunk = prompts[i:i + CHUNK_SIZE]

            # EXTRA SAFETY VALIDATION
            validated_chunk = []

            for prompt in chunk:

                tok_len = len(
                    tokenizer.encode(
                        prompt,
                        add_special_tokens=False
                    )
                )

                # Ensure prompt + generation fit model context
                if tok_len + MAX_OUTPUT_TOKENS <= MAX_MODEL_LEN:
                    validated_chunk.append(prompt)

            if not validated_chunk:
                continue

            outputs = llm.generate(
                validated_chunk,
                sampling_params
            )

            for prompt, output in zip(validated_chunk, outputs):

                response = output.outputs[0].text.strip()

                item = {
                    "prompt": prompt,
                    "response": response
                }

                f_out.write(json.dumps(item) + "\n")

    print(f"\nDone! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()