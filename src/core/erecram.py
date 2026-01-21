import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class MemoryCell:
    state: torch.Tensor
    timestamp: float
    weight: float

class ErecRAM(nn.Module):
    """
    ErecRAM (Entity-Related Elastic RAM) - Heavy Version
    Optimized for High-Resolution State Spaces (e.g., 4096-dim on RTX 4090)
    """
    def __init__(
        self, 
        state_dim: int = 4096, 
        memory_size: int = 2000, 
        lambda_decay: float = 0.01,
        alpha_continuity: float = 0.95,
        tau: float = 0.0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.lambda_decay = lambda_decay
        self.alpha_continuity = alpha_continuity
        self.tau = tau
        
        # Core State: Moved to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_state = nn.Parameter(torch.zeros(state_dim), requires_grad=False).to(self.device)
        self.memory_bank: List[MemoryCell] = []

    def _calculate_attention(self, query: torch.Tensor, memory: List[MemoryCell]) -> torch.Tensor:
        if not memory:
            return query
        
        # High-performance parallel attention on GPU
        states = torch.stack([cell.state for cell in memory]).to(self.device)
        weights = torch.tensor([cell.weight for cell in memory], device=self.device)
        
        # Scaled Dot-Product Attention
        d_k = query.size(-1)
        scores = torch.matmul(query, states.transpose(0, 1)) / (d_k ** 0.5)
        
        # Fusion of temporal weights and content similarity
        attn_weights = F.softmax(scores * weights, dim=-1)
        output = torch.matmul(attn_weights, states)
        
        return output

    def forward(self, sensed_state: torch.Tensor, t_new: float):
        """
        Main existence update loop (S_t = f(S_{t-1}, Sensing))
        """
        # 1. Temporal Decay (Elasticity)
        for cell in self.memory_bank:
            delta_t = t_new - cell.timestamp
            time_weight = torch.exp(-torch.tensor(self.lambda_decay * abs(delta_t - self.tau)))
            cell.weight *= time_weight.item()

        # 2. Attention-driven Perspective
        attention_out = self._calculate_attention(self.current_state, self.memory_bank)
        
        # 3. Recursive Continuity Update
        # S_t = alpha * S_{t-1} + (1-alpha) * Perception
        new_state = (self.alpha_continuity * self.current_state) + \
                    ((1 - self.alpha_continuity) * attention_out)
        
        # 4. Layer Normalization (Stability)
        # Keeps the 4096-dim vector from exploding or collapsing
        self.current_state.data = F.layer_norm(new_state, (self.state_dim,))

        # 5. Memory Ingestion (Episodic)
        new_cell = MemoryCell(
            state=sensed_state.detach().cpu(), # Store in CPU RAM to save VRAM for LLM
            timestamp=t_new, 
            weight=1.0
        )
        self.memory_bank.append(new_cell)
        
        if len(self.memory_bank) > self.memory_size:
            self.memory_bank.pop(0)

    def update_from_action_feedback(self, feedback_state: torch.Tensor):
        fb = feedback_state.to(self.device)
        new_state = (self.alpha_continuity * self.current_state) + \
                    ((1 - self.alpha_continuity) * fb)
        self.current_state.data = F.layer_norm(new_state, (self.state_dim,))

    def save_state(self, path: str):
        state_dict = {
            "current_state": self.current_state.cpu(),
            "memory_bank": self.memory_bank,
            "tau": self.tau
        }
        torch.save(state_dict, path)
        print(f"💾 [ErecRAM] State persisted to {path}")

    def load_state(self, path: str):
        import os
        if not os.path.exists(path): return False
        
        checkpoint = torch.load(path, weights_only=False)
        saved_state = checkpoint["current_state"]
        
        # Guard for dimension changes
        if saved_state.size(-1) != self.state_dim:
            print(f"⚠️ [ErecRAM] Dimension mismatch: Saved {saved_state.size(-1)} vs Current {self.state_dim}. Resetting.")
            return False
            
        self.current_state.data = saved_state.to(self.device)
        self.memory_bank = checkpoint["memory_bank"]
        self.tau = checkpoint.get("tau", 0.0)
        print(f"✨ [ErecRAM] State resumed from {path} ({self.state_dim}-dim)")
        return True
