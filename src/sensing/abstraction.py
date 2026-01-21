import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StateAbstractor:
    def __init__(self, feature_dim: int, window_size: int = 10, stable_threshold: float = 0.9, transition_threshold: float = 0.01):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.stable_threshold = stable_threshold # Lower means more picky about stability
        self.transition_threshold = transition_threshold # Lower means more sensitive to change
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        self.prev_entropy = 0.0

    def abstract(self, z_sequence: torch.Tensor):
        """
        input: z_sequence (Shape: [WindowSize, FeatureDim])
        """
        # 1. Self-Attention calculation (simplified)
        Q = self.q_proj(z_sequence)
        K = self.k_proj(z_sequence)
        V = self.v_proj(z_sequence)
        
        # scores: [WindowSize, WindowSize]
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.feature_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 2. Entropy calculation
        # H = -sum(p * log(p))
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean().item()
        
        # 3. State Determination
        # Max entropy is log(WindowSize). Normalize to [0, 1]
        relative_entropy = entropy / math.log(self.window_size)
        
        state_flag = "IDLE"
        if relative_entropy < self.stable_threshold:
            state_flag = "STABLE"
        
        # Using entropy gradient for transition detection
        entropy_diff = abs(relative_entropy - self.prev_entropy)
        if entropy_diff > self.transition_threshold:
            state_flag = "TRANSITION"
            
        self.prev_entropy = relative_entropy
        
        # Pooled abstract state (mean of V weighted by last row of attention)
        abstract_state = torch.matmul(attn_weights[-1:], V).squeeze(0)
        
        return abstract_state, relative_entropy, state_flag
