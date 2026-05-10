import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.model.moe import MoELayer


# ----------------------------
# RoPE
# ----------------------------
def apply_rope(q, k):
    b, h, t, d = q.shape
    half = d // 2

    freq = torch.arange(half, device=q.device, dtype=q.dtype)
    freq = 1.0 / (10000 ** (freq / half))

    pos = torch.arange(t, device=q.device, dtype=q.dtype)
    angles = pos[:, None] * freq[None, :]

    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]

    def rotate(x):
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotate(q), rotate(k)


# ----------------------------
# RMSNorm
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x * self.weight / (norm / math.sqrt(x.size(-1)) + self.eps)


# ----------------------------
# Multi-Head Attention (FlashAttention + RoPE)
# ----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, t, d = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k)

        # for causal models only
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        out = att.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


# ----------------------------
# Dense FFN (SwiGLU)
# ----------------------------
class DenseFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ----------------------------
# Transformer Block (Dense or MoE)
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_hidden_dim, ffn_type="dense", num_experts=4, k=2):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)

        if ffn_type == "moe":
            self.ffn = MoELayer(dim, ffn_hidden_dim, num_experts, k)
            self.is_moe = True
        else:
            self.ffn = DenseFFN(dim, ffn_hidden_dim)
            self.is_moe = False

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, mask=mask))

        h = self.norm2(x)

        if self.is_moe:
            ffn_out, moe_loss, gate_scores = self.ffn(h)
            x = x + self.dropout(ffn_out)

            return x, {
                "moe_loss": moe_loss,
                "gate_scores": gate_scores
            }

        else:
            ffn_out = self.ffn(h)
            x = x + self.dropout(ffn_out)

            return x, {
                "moe_loss": torch.tensor(0.0, device=x.device),
                "gate_scores": None
            }


# ----------------------------
# Full Transformer
# ----------------------------
class MoETransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_experts=4,
        k=2,
        max_seq_len=1024,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.max_seq_len = max_seq_len

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            ffn_type = "moe" if i % 2 == 1 else "dense"
            self.blocks.append(
                TransformerBlock(
                    dim,
                    num_heads,
                    ffn_hidden_dim,
                    ffn_type=ffn_type,
                    num_experts=num_experts,
                    k=k
                )
            )

        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.scale = 1 / math.sqrt(dim)

    def forward(self, input_ids, attention_mask=None):

        x = self.token_emb(input_ids)

        total_moe_loss = torch.tensor(0.0, device=input_ids.device)
        gate_stats = []

        for block in self.blocks:
            x, aux = block(x, mask=attention_mask)

            if aux["moe_loss"] is not None:
                total_moe_loss = total_moe_loss + aux["moe_loss"]
            if aux["gate_scores"] is not None:
                gate_stats.append(aux["gate_scores"].detach().cpu())

        x = self.norm_f(x)
        logits = self.lm_head(x) * self.scale

        return logits, {
            "moe_loss": total_moe_loss,
            "gate_scores": gate_stats
        }