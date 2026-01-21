import torch
import torch.nn as nn

class PhysicalEncoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super(PhysicalEncoder, self).__init__()
        
        # L3-2. Physical Encoder Architecture
        # Conv1D -> LayerNorm -> GELU -> TCN
        
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=feature_dim, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.gelu = nn.GELU()
        
        # Simplified TCN Layer (Residual block with dilated conv)
        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU()
        )

    def encode(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        input: raw_signal X_t (Expected shape: [Batch, Dim] or [Batch, Dim, Seq])
        """
        # Ensure input has channel dimension for Conv1D: [Batch, Dim, 1]
        if x_t.dim() == 2:
            x = x_t.unsqueeze(-1)
        else:
            x = x_t

        # 1. Conv1D
        z = self.conv1d(x)
        
        # 2. LayerNorm (Needs [B, C, L] -> [B, L, C] for LayerNorm then back)
        z = z.transpose(1, 2)
        z = self.layer_norm(z)
        z = z.transpose(1, 2)
        
        # 3. GELU
        z = self.gelu(z)
        
        # 4. TCN
        z_t = self.tcn(z)
        
        # Return pooled feature vector [Batch, FeatureDim]
        return torch.mean(z_t, dim=-1).squeeze(0)
