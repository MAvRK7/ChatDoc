import json
import argparse
from tqdm import tqdm


def format_conversation(messages):
    text = ""
    system_prefix = ""
    
    # Extract system message if present
    for msg in messages:
        if msg.get("role") == "system":
            system_prefix = msg.get("content", "").strip()
            break
    
    # Build turns
    current_user = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        
        if not content:
            continue
        
        content = " ".join(content.split())  # normalize whitespace
        
        if role == "system":
            continue  # already captured
            
        elif role == "user":
            # Prepend system instruction to first user message
            if system_prefix and current_user is None:
                content = f"[{system_prefix}] {content}"
            current_user = content
            text += f"<user> {content}\n"
            
        elif role == "assistant":
            if current_user is not None:
                text += f"<assistant> {content}\n"
                current_user = None  # reset
    
    text += "<eos>"
    return text


def process_file(input_path, output_path):
    print(f"\n📥 Loading: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Formatting"):
            data = json.loads(line)

            messages = data.get("messages", [])
            if len(messages) < 2:
                continue

            text = format_conversation(messages)

            if len(text) < 20:
                continue

            json.dump({"text": text}, f_out, ensure_ascii=False)
            f_out.write("\n")

    print(f"\n✅ Done writing JSONL: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    process_file(args.input, args.output)

# run
# python scripts/format.py --input data/processed/train.jsonl --output data/processed/train_formatted.jsonl
# python scripts/format.py --input data/processed/val.jsonl --output data/processed/val_formatted.jsonl