import torch
import re
import sentencepiece as spm
from src.model.transformer import MoETransformer
from src.sampling import sample

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

pad_id = sp.piece_to_id("<pad>")
unk_id = sp.piece_to_id("<unk>")
bos_id = sp.piece_to_id("<bos>")
eos_id = sp.piece_to_id("<eos>")
user_id = sp.piece_to_id("<user>")
assistant_id = sp.piece_to_id("<assistant>")

# -----------------------------
# LOAD MODEL
# -----------------------------
vocab_size = sp.get_piece_size()

model = MoETransformer(
    vocab_size=vocab_size,
    dim=768,
    num_layers=12,
    num_heads=12,
    ffn_hidden_dim=1536,
    num_experts=4,
    k=1,
    max_seq_len=1024,
)

#state = torch.load("model.pt", map_location=device)
state = torch.load("checkpoints/model.pt/model.pt", map_location=device)
model.load_state_dict(state["model"])
model.to(device)
model.eval()

# -----------------------------
# REPETITION PENALTY (IMPROVED)
# -----------------------------
def apply_repetition_penalty(logits, tokens, penalty=1.25, window=64):
    recent_tokens = tokens[-window:]

    for token_id in set(recent_tokens):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty

    return logits

# -----------------------------
# N-GRAM BLOCKING
# -----------------------------
def block_repeated_ngrams(ids, logits, n=3):
    if ids.shape[1] < n:
        return logits

    generated = ids[0].tolist()
    ngrams = set()

    for i in range(len(generated) - n + 1):
        ngram = tuple(generated[i:i+n])
        ngrams.add(ngram)

    prefix = tuple(generated[-(n-1):])

    for token in range(logits.shape[-1]):
        candidate = prefix + (token,)
        if candidate in ngrams:
            logits[0, token] = -float("inf")

    return logits

# -----------------------------
# GENERATION
# -----------------------------
def generate(
    user_input,
    max_new_tokens=150,
    temperature=0.5,
    top_k=20,
    top_p=0.85,
):
    prompt = f"<user> {user_input}\n<assistant> Answer clearly:"

    ids_list = sp.EncodeAsIds(prompt)
    ids = torch.tensor([ids_list], dtype=torch.long).to(device)

    generated_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids)
            logits = logits[:, -1, :]

            # ✅ repetition penalty
            logits = apply_repetition_penalty(
                logits,
                ids[0].tolist(),
                penalty=1.25,
                window=64
            )

            # ✅ n-gram blocking
            logits = block_repeated_ngrams(ids, logits, n=3)

            # ✅ sampling
            next_id = sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        token_id = next_id.item()

        # ✅ stopping conditions
        if token_id in (eos_id, user_id):
            break

        generated_tokens.append(token_id)
        ids = torch.cat([ids, next_id], dim=1)

        # ✅ anti-loop safeguard
        if len(generated_tokens) > 30:
            recent = generated_tokens[-10:]
            if len(set(recent)) < 3:
                break

    # -----------------------------
    # DECODE
    # -----------------------------
    text = sp.DecodeIds(generated_tokens)

    # cleanup
    text = re.sub(r"<assistant>|<user>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    user_input = "Explain AI in education?"

    response = generate(user_input)

    print(f"User: {user_input}")
    print(f"Assistant: {response}")

# Run 
# python -m src.inference
