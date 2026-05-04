import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, k=2, aux_loss_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        b, s, d = x.shape
        gate_logits = self.gate(x)                    # (b, s, E)

        # Softmax + top-k
        gate_scores = F.softmax(gate_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)

        # --- Auxiliary Losses ---
        # 1. Load balancing loss
        expert_usage = gate_scores.mean(dim=(0, 1))
        load_balance_loss = -(expert_usage * torch.log(expert_usage + 1e-9)).sum()

        # 2. Z-loss (important for MoE stability)
        z_loss = torch.mean(torch.square(torch.logsumexp(gate_logits, dim=-1))) * 0.001

        aux_loss = load_balance_loss + z_loss

        # FIXED DISPATCH: accumulate properly without in-place += on masked view
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            for rank in range(self.k):
                mask = (topk_indices[..., rank] == e)   # (b, s)
                if not mask.any():
                    continue

                # Flatten mask for gathering
                flat_mask = mask.view(-1)  # (b*s,)
                flat_x = x.view(-1, d)     # (b*s, d)
                tokens = flat_x[flat_mask]   # (n, d)

                expert_out = self.experts[e](tokens)  # (n, d)

                # Weight by this expert's score at this rank
                flat_scores = topk_scores[..., rank].view(-1)  # (b*s,)
                weights = flat_scores[flat_mask].unsqueeze(-1)  # (n, 1)

                weighted_out = expert_out * weights  # (n, d)

                # Scatter back properly
                flat_out = out.view(-1, d)
                flat_out[flat_mask] = flat_out[flat_mask] + weighted_out

        return out, aux_loss, gate_scores