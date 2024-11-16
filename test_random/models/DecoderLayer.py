import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, ffw_hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)

        # FFW network
        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, ffw_hidden_dim),
            nn.GELU(),
            nn.Linear(ffw_hidden_dim, embed_dim),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + attn_output # Residual connection
        x = self.ln1(x)

        # FFW network
        ffw_output = self.ffw(x)
        x = x + ffw_output
        x = self.ln2(x)

        return x
