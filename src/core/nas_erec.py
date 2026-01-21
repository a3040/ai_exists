import torch
import torch.nn as nn
from typing import List, Dict, Optional
import time
from .modules import SEARCH_SPACE
from .erecram import MemoryCell

class NASErecRAM(nn.Module):
    def __init__(
        self, 
        state_dim: int = 4096, 
        memory_size: int = 2000,
        dna: Dict[str, str] = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.memory_size = memory_size
        
        # DNA defines the architecture parts
        self.dna = dna or {
            "attention": "scaled_dot",
            "update": "recursive",
            "norm": "layer"
        }
        self.dna_history: List[Dict] = []
        
        self.current_state = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
        self.memory_bank: List[MemoryCell] = []
        
        self._build_architecture()
        self.lambda_decay = 0.01
        self.alpha_continuity = 0.95
        self.tau = 0.0

    def _build_architecture(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attn_layer = SEARCH_SPACE["attention"][self.dna["attention"]](self.state_dim).to(device)
        self.update_layer = SEARCH_SPACE["update"][self.dna["update"]](self.state_dim).to(device)
        self.norm_layer = SEARCH_SPACE["norm"][self.dna["norm"]](self.state_dim).to(device)
        self.to(device)

    def update_dna(self, new_dna: Dict[str, str]):
        print(f"🧬 [NAS] Mutating DNA: {self.dna} -> {new_dna}")
        
        # Evo-Log: Record Cause and Effect
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": timestamp,
            "old_dna": self.dna.copy(),
            "new_dna": new_dna.copy(),
            # In a real heavy system, we would log the 'reason' (e.g. specific confusing memory ID)
            "cause": "high_entropy" 
        }
        self.dna_history.append(record)
        
        self.dna = new_dna
        self._build_architecture()

    def _calculate_attention(self, query: torch.Tensor, memory: List[MemoryCell]) -> torch.Tensor:
        if not memory:
            return query
        
        device = query.device
        states = torch.stack([cell.state for cell in memory]).to(device)
        weights = torch.tensor([cell.weight for cell in memory], device=device)
        
        return self.attn_layer(query, states, weights)

    def __call__(self, sensed_state: torch.Tensor, t_new: float):
        """Standard existence update loop"""
        for cell in self.memory_bank:
            delta_t = t_new - cell.timestamp
            time_weight = torch.exp(-torch.tensor(self.lambda_decay * abs(delta_t - self.tau)))
            cell.weight *= time_weight.item()

        attention_out = self._calculate_attention(self.current_state, self.memory_bank)
        new_state = self.update_layer(self.current_state, attention_out, self.alpha_continuity)
        self.current_state.data = self.norm_layer(new_state)

        # Memory update
        new_cell = MemoryCell(state=sensed_state.detach().cpu(), timestamp=t_new, weight=1.0)
        self.memory_bank.append(new_cell)
        if len(self.memory_bank) > self.memory_size:
            self.memory_bank.pop(0)

    def update_from_action_feedback(self, feedback_state: torch.Tensor):
        fb = feedback_state.to(self.current_state.device)
        new_state = self.update_layer(self.current_state, fb, self.alpha_continuity)
        self.current_state.data = self.norm_layer(new_state)

    def save_state(self, path: str):
        state_dict = {
            "current_state": self.current_state.cpu(),
            "memory_bank": self.memory_bank,
            "dna": self.dna,
            "dna_history": self.dna_history,
            "tau": self.tau
        }
        torch.save(state_dict, path)
        print(f"💾 [ErecRAM] NAS-State persisted to {path}")

    def load_state(self, path: str):
        import os
        if not os.path.exists(path): return False
        checkpoint = torch.load(path, weights_only=False)
        saved_state = checkpoint["current_state"]
        
        if saved_state.size(-1) != self.state_dim:
            print(f"⚠️ [NAS] State dimension mismatch: Saved {saved_state.size(-1)} vs Current {self.state_dim}. Resetting.")
            return False
            
        self.current_state.data = saved_state.to(self.current_state.device)
        self.memory_bank = checkpoint["memory_bank"]
        self.tau = checkpoint.get("tau", 0.0)
        if "dna" in checkpoint:
            self.update_dna(checkpoint["dna"])
        
        self.dna_history = checkpoint.get("dna_history", [])
        print(f"✨ [ErecRAM] NAS-State resumed from {path}")
        return True
