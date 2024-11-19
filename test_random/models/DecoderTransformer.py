import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DecoderLayer import DecoderLayer

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=640, num_layers=10, num_heads=10,
                 head_dim=64, ffw_hidden_dim=2560, max_seq_len=1024, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

        # Embedding layer (256 [one-hot] -> 640 [embedding])
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (1024 [one-hot] -> 640 [embedding])
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, head_dim, ffw_hidden_dim)
            for i in range(num_layers)
        ])

        # Output projection (640 [embedding] -> 256 [logits])
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        # self.output_projection.weight = self.word_embedding.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6}M")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
                nn.init.zeros_(module.in_proj_bias)
                nn.init.trunc_normal_(module.out_proj.weight, std=0.02)
                nn.init.zeros_(module.out_proj.bias)

    
    def forward(self, x, mask=None):
        # X is a matrix of size (batch_size, seq_len)
        seq_len = x.size(1) # maximum sequence length in batch
        positions = torch.arange(0, seq_len, 1).to(self.device)
        x.to(self.device)

        # Word & Positional Embeddings
        tok_emb = self.word_embedding(x) # (batch_size, seq_len, embd_dim)
        pos_emb = self.positional_embedding(positions) # (1, seq_len, embd_dim)
        x = tok_emb + pos_emb # (batch_size, seq_len, embd_dim) -- broadcasting magic

        # Decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output projection
        logits = self.output_projection(x)
        return logits
