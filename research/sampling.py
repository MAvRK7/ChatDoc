import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    # Top-k filtering
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float("-inf")),
            logits
        )

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens with cumulative prob > top_p
        # But keep the first token that pushes it over
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    More correct repetition penalty (GPT-style)
    """
    for token_id in set(generated_ids):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


def sample(logits, temperature=1.0, top_k=40, top_p=0.9):
    # Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # Filter
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Softmax
    probs = F.softmax(logits, dim=-1)

    # Safety: prevent NaNs
    if torch.isnan(probs).any():
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # Sample
    return torch.multinomial(probs, num_samples=1)