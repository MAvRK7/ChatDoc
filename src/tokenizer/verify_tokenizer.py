import sentencepiece as spm
import json
import sys

def verify_tokenizer(model_path="tokenizer/tokenizer.model", test_file="data/processed/train_formatted.jsonl"):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")
    print()

    # Check all special tokens exist as single-piece tokens
    print("=== Special Token Check ===")
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]
    all_ok = True
    token_ids = {}
    for tag in special_tokens:
        ids = sp.encode(tag, out_type=int)
        pieces = sp.encode(tag, out_type=str)
        is_single = len(ids) == 1
        status = "✅" if is_single else "❌"
        token_ids[tag] = ids[0] if is_single else None
        print(f"  {status} {tag:12s} -> id={token_ids[tag]}, pieces={pieces}")
        if not is_single:
            all_ok = False

    if not all_ok:
        print("\n❌ WARNING: Some special tokens are multi-piece!")
    print()

    print("=== ID Mapping (update your model config!) ===")
    for tag in special_tokens:
        print(f"  {tag:12s} id = {token_ids[tag]}")
    print()

    # Synthetic test
    print("=== Synthetic Test ===")
    test_text = "<user> hello <assistant> world <eos>"
    enc = sp.encode(test_text, out_type=int)
    dec = sp.decode(enc)
    lossless = test_text == dec
    print(f"  Original: {test_text!r}")
    print(f"  Decoded:  {dec!r}")
    print(f"  Tokens:   {enc}")
    print(f"  Lossless: {'✅ Yes' if lossless else '❌ No'}")
    print()

    # Real data test
    print("=== Real Data Test ===")
    try:
        with open(test_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  ⚠️ Could not open {test_file}, skipping real data test")
        return

    for i, line in enumerate(lines[:5]):
        obj = json.loads(line)
        text = obj["text"]
        enc = sp.encode(text, out_type=int)
        dec = sp.decode(enc)

        orig_norm = text.rstrip()
        dec_norm = dec.rstrip()
        lossless = orig_norm == dec_norm

        status = "✅" if lossless else "❌"
        print(f"  {status} Sample {i+1} ({len(enc)} tokens, {len(text)} chars)")

        if not lossless:
            print(f"      Original: {text[:80]!r}...")
            print(f"      Decoded:  {dec[:80]!r}...")
            for j in range(min(len(orig_norm), len(dec_norm))):
                if orig_norm[j] != dec_norm[j]:
                    print(f"      First mismatch at pos {j}: orig='{orig_norm[j]}' vs dec='{dec_norm[j]}'")
                    break
            if len(orig_norm) != len(dec_norm):
                print(f"      Length mismatch: orig={len(orig_norm)} vs dec={len(dec_norm)}")

    print(f"\n=== Full Dataset Stats (first 1000 samples) ===")
    total_tokens = 0
    total_chars = 0
    for line in lines[:1000]:
        obj = json.loads(line)
        text = obj["text"]
        enc = sp.encode(text, out_type=int)
        total_tokens += len(enc)
        total_chars += len(text)

    avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"  Total chars: {total_chars:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg chars/token: {avg_chars_per_token:.2f}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer/tokenizer.model"
    test_file = sys.argv[2] if len(sys.argv) > 2 else "data/processed/train_formatted.jsonl"
    verify_tokenizer(model_path, test_file)

# Run this
# python src/tokenizer/verify_tokenizer.py