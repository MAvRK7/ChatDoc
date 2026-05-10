import sentencepiece as spm
import json
import argparse
import random
import re
import os
from tqdm import tqdm

# <unk> is MANDATORY as a control symbol in SentencePiece.
# Everything else that appears in text goes to user_defined_symbols.
USER_DEFINED_SYMBOLS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"]

def normalize_text(text: str) -> str:
    text = text.rstrip()
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def load_jsonl(path, sample_size=None):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {path}")):
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text or len(text) < 50:
                continue
            text = normalize_text(text)
            texts.append(text)
            if sample_size and len(texts) >= sample_size:
                break
    return texts


def write_corpus(texts, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in texts:
            f.write(t + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="tokenizer")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()

    all_texts = []
    print("\n📥 Loading datasets...")
    for path in args.input:
        if not os.path.exists(path):
            print(f"⚠️ Skipping: {path}")
            continue
        texts = load_jsonl(path, args.sample_size)
        print(f"Loaded {len(texts)} from {path}")
        all_texts.extend(texts)

    print(f"\nTotal before dedup: {len(all_texts)}")
    all_texts = list(set(all_texts))
    random.shuffle(all_texts)
    print(f"After dedup: {len(all_texts)}")

    os.makedirs(args.output, exist_ok=True)
    corpus_path = os.path.join(args.output, "corpus.txt")
    write_corpus(all_texts, corpus_path)

    print("\n🧠 Training SentencePiece...")
    model_prefix = os.path.join(args.output, "sp")

    # FIXED: <unk> is the ONLY mandatory control symbol.
    # <pad>, <bos>, <eos> are disabled (-1) so they become user-defined symbols.
    # <user>, <assistant> are user-defined (appear in text).
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        byte_fallback=True,
        split_by_whitespace=True,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        pad_id=-1,          # disabled -> user-defined
        unk_id=1,           # MANDATORY: <unk> must be a control symbol
        bos_id=-1,          # disabled -> user-defined
        eos_id=-1,          # disabled -> user-defined
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        shuffle_input_sentence=True,
        input_sentence_size=400000,
        num_threads=4,
        max_sentencepiece_length=16,
        seed_sentencepiece_size=200000,
        shrinking_factor=0.75,
        num_sub_iterations=2,
        max_sentence_length=2048,
    )

    import shutil
    final_model = os.path.join(args.output, "tokenizer.model")
    if os.path.exists(final_model):
        os.remove(final_model)
    shutil.move(f"{model_prefix}.model", final_model)

    final_vocab = os.path.join(args.output, "tokenizer.vocab")
    if os.path.exists(final_vocab):
        os.remove(final_vocab)
    if os.path.exists(f"{model_prefix}.vocab"):
        shutil.move(f"{model_prefix}.vocab", final_vocab)

    print("\n✅ Done!")
    print(f"Model: {final_model}")
    print(f"Vocab: {final_vocab}")
    print("\n⚠️  IMPORTANT: Check the vocab file to confirm IDs for <pad>, <bos>, <eos>, <user>, <assistant>.")
    print("   <unk> is fixed at id=1 (control symbol).")
    print("   Update your model config to match the actual user-defined token IDs.")


if __name__ == "__main__":
    main()

# comand line usage
# python src/tokenizer/train_tokenizer.py --input data/processed/train_formatted.jsonl --vocab_size 20000 --output tokenizer