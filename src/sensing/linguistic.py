import torch
import torch.nn as nn
from typing import List

class LinguisticEncoder(nn.Module):
    def __init__(self, feature_dim: int = 4096, vocab_size: int = 256, embed_dim: int = 128, hidden_dim: int = 512):
        super(LinguisticEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, feature_dim)
        
    def encode(self, text: str) -> torch.Tensor:
        """
        Convert text into a high-dim vector.
        """
        if not text:
            return torch.zeros(self.feature_dim)
            
        # Byte-level encoding (simple & robust for test)
        bytes_data = text.encode('utf-8')
        input_ids = torch.tensor([min(b, 255) for b in bytes_data]).unsqueeze(0)
        
        embeds = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embeds)
        
        # Take the last layer's hidden states (forward and backward)
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        # We need the last forward (-2) and last backward (-1)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1) # [1, 1024]
        
        z_ling = self.projection(last_hidden) 
        return z_ling.squeeze(0) # [2048]
