import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class MemoryCell:
    state: torch.Tensor
    timestamp: float
    weight: float

class ErecRAM:
    def __init__(
        self, 
        state_dim: int = 2048, # UPGRADED
        memory_size: int = 500, 
        lambda_decay: float = 0.01, # More persistence
        alpha_continuity: float = 0.95,
        tau: float = 0.0
    ):
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.lambda_decay = lambda_decay
        self.alpha_continuity = alpha_continuity
        
        # Initialize state with zeros/neutral equilibrium
        self.current_state = torch.zeros(state_dim)
        self.memory_bank: List[MemoryCell] = []
        self.tau = tau

    def _calculate_attention(self, query: torch.Tensor, memory: List[MemoryCell]) -> torch.Tensor:
        if not memory:
            return query
        
        # Keys and Values from memory
        keys = torch.stack([cell.state for cell in memory])
        values = keys # In this context, K=V
        
        # Standard scaled dot-product attention scores
        # query: (1, dim), keys: (N, dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, keys.transpose(0, 1)) / (d_k ** 0.5)
        
        # Apply temporal weights from memory cells
        temporal_weights = torch.tensor([cell.weight for cell in memory], device=query.device)
        weighted_scores = scores * temporal_weights
        
        attn_weights = F.softmax(weighted_scores, dim=-1)
        output = torch.matmul(attn_weights, values)
        
        return output

    def update_from_sensing(self, sensed_state: torch.Tensor, t_new: float):
        """
        L3-3.1 update_from_sensing() logic
        """
        # 1. Update existing memory weights based on physical delay tau
        for cell in self.memory_bank:
            delta_t = t_new - cell.timestamp
            # Time weight reflecting existence continuity: exp(-lambda * |Δt - tau|)
            time_weight = torch.exp(-torch.tensor(self.lambda_decay * abs(delta_t - self.tau)))
            cell.weight *= time_weight.item()

        # 2. Calculate attention output
        attention_out = self._calculate_attention(self.current_state, self.memory_bank)
        
        # 3. Recursive update with continuity gain (alpha)
        # S_t = alpha * S_{t-1} + (1-alpha) * attention_out
        self.current_state = (self.alpha_continuity * self.current_state) + \
                             ((1 - self.alpha_continuity) * attention_out)

        # 4. Add new sensing result to memory
        new_cell = MemoryCell(state=sensed_state.clone(), timestamp=t_new, weight=1.0)
        self.memory_bank.append(new_cell)
        
        # 5. Maintain memory size
        if len(self.memory_bank) > self.memory_size:
            self.memory_bank.pop(0)

    def update_from_action_feedback(self, feedback_state: torch.Tensor):
        """
        L3-3.2 update_from_action_feedback() logic
        """
        # current_state = alpha * current_state + (1-alpha) * S_fb
        self.current_state = (self.alpha_continuity * self.current_state) + \
                             ((1 - self.alpha_continuity) * feedback_state)
    def save_state(self, path: str):
        """Save the ongoing existence state"""
        state_dict = {
            "current_state": self.current_state,
            "memory_bank": self.memory_bank,
            "tau": self.tau
        }
        torch.save(state_dict, path)
        print(f"💾 [ErecRAM] State persisted to {path}")

    def load_state(self, path: str):
        """Load the persisted state"""
        import os
        if not os.path.exists(path):
            print(f"⚠️ [ErecRAM] No persisted state found at {path}")
            return False
            
        state_dict = torch.load(path, weights_only=False)
        self.current_state = state_dict["current_state"]
        self.memory_bank = state_dict["memory_bank"]
        self.tau = state_dict["tau"]
        print(f"✨ [ErecRAM] State resumed from {path}")
        return True
