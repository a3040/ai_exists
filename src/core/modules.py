import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

# --- Attention Space ---

class ScaledDotAttention(BaseModule):
    def forward(self, query, memory_states, weights):
        d_k = query.size(-1)
        scores = torch.matmul(query, memory_states.transpose(0, 1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores * weights, dim=-1)
        return torch.matmul(attn_weights, memory_states)

class CosineAttention(BaseModule):
    def forward(self, query, memory_states, weights):
        sim = F.cosine_similarity(query.unsqueeze(0), memory_states, dim=1)
        attn_weights = F.softmax(sim * weights, dim=-1)
        return torch.matmul(attn_weights, memory_states)

class GatedAttention(BaseModule):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.gate = nn.Linear(dim * 2, 1)
        
    def forward(self, query, memory_states, weights):
        N = memory_states.size(0)
        q_ext = query.expand(N, -1)
        cat = torch.cat([q_ext, memory_states], dim=-1)
        scores = self.gate(cat).squeeze(-1)
        attn_weights = F.softmax(scores * weights, dim=-1)
        return torch.matmul(attn_weights, memory_states)

# --- Update Space ---

class RecursiveUpdate(BaseModule):
    def forward(self, current, target, alpha):
        return (alpha * current) + ((1 - alpha) * target)

class GatedUpdate(BaseModule):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.gru_cell = nn.GRUCell(dim, dim)
        
    def forward(self, current, target, alpha):
        return self.gru_cell(target.unsqueeze(0), current.unsqueeze(0)).squeeze(0)

# --- Norm Space ---

class IdentityNorm(BaseModule):
    def forward(self, x): return x

class LayerNormModule(BaseModule):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x.unsqueeze(0)).squeeze(0)

# --- Registry ---
SEARCH_SPACE = {
    "attention": {
        "scaled_dot": ScaledDotAttention,
        "cosine": CosineAttention,
        "gated": GatedAttention
    },
    "update": {
        "recursive": RecursiveUpdate,
        "gated": GatedUpdate
    },
    "norm": {
        "identity": IdentityNorm,
        "layer": LayerNormModule
    }
}
