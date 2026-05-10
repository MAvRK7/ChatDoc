import torch
from tokenizers import Tokenizer
from research.model.transformer import MoETransformer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file("tokenizer.json")
    vocab_size = tok.get_vocab_size()

    # Load model architecture
    model = MoETransformer(
        vocab_size=vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_experts=4,
        k=2,
        max_seq_len=1024,
    ).to(device)

    # Load trained weights
    checkpoint = torch.load("model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Test forward pass
    x = torch.randint(0, vocab_size, (2, 64)).to(device)
    logits, moe_loss = model(x)

    print("Logits shape:", logits.shape)
    print("MoE loss:", moe_loss.item())

# Run
# python -m src.model.test_model