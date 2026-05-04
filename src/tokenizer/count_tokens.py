import json
import sentencepiece as spm
from tqdm import tqdm

IGNORE_INDEX = -100


def extract_turns(text):
    turns = []

    text = text.rstrip()
    if text.endswith("<eos>"):
        text = text[:-5].rstrip()

    parts = text.split("<user>")

    for part in parts[1:]:
        if "<assistant>" not in part:
            continue

        split_idx = part.find("<assistant>")
        user_text = part[:split_idx].strip()
        rest = part[split_idx + len("<assistant>"):].strip()

        next_user = rest.find("<user>")
        if next_user != -1:
            assistant_text = rest[:next_user].strip()
        else:
            assistant_text = rest.strip()

        if len(user_text) > 5 and len(assistant_text) > 10:
            turns.append((user_text, assistant_text))

    return turns


def count_tokens(tokenizer_path, dataset_path, max_length=512):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    eos_id = sp.piece_to_id("<eos>")

    total_tokens = 0
    train_tokens = 0
    num_samples = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting tokens"):
            obj = json.loads(line)
            text = obj.get("text", "")

            if "<user>" not in text or "<assistant>" not in text:
                continue

            turns = extract_turns(text)

            for user_text, assistant_text in turns:
                input_ids = []
                labels = []

                # USER (no loss)
                user_tokens = sp.EncodeAsIds(f"<user> {user_text} ")
                input_ids.extend(user_tokens)
                labels.extend([IGNORE_INDEX] * len(user_tokens))

                # ASSISTANT prompt (no loss)
                assistant_prompt = sp.EncodeAsIds("<assistant> ")
                input_ids.extend(assistant_prompt)
                labels.extend([IGNORE_INDEX] * len(assistant_prompt))

                # ASSISTANT answer (trainable)
                answer_tokens = sp.EncodeAsIds(assistant_text)
                input_ids.extend(answer_tokens)
                labels.extend(answer_tokens)

                # EOS
                input_ids.append(eos_id)
                labels.append(eos_id)

                # Truncate exactly like training
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

                total_tokens += len(input_ids)
                train_tokens += sum(1 for l in labels if l != IGNORE_INDEX)
                num_samples += 1

    print(f"\nTotal samples (turn-level): {num_samples}")
    print(f"Total tokens (after truncation): {total_tokens:,}")
    print(f"Trainable tokens (assistant only): {train_tokens:,}")
    print(f"Average tokens per sample: {total_tokens / num_samples:.2f}")
    print(f"Trainable ratio: {train_tokens / total_tokens:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    count_tokens(args.tokenizer, args.dataset, args.max_length)


# Run 
# python -m src.tokenizer.count_tokens --tokenizer tokenizer/tokenizer.model --dataset data/processed/train_formatted.jsonl