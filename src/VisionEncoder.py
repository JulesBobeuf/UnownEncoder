import torch

from MultiHeadAttention import MultiHeadAttention

class VisionEncoder(torch.nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super(VisionEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size,self.num_heads, dropout=dropout)
        
        self.mlp = torch.nn.Sequential(
                          torch.nn.Linear(embed_size, 4* embed_size),
                          torch.nn.GELU(),
                          torch.nn.Dropout(self.dropout),
                          torch.nn.Linear(4* embed_size, embed_size),
                          torch.nn.Dropout(self.dropout))

    def forward(self, x):
        fwd_norm1 = self.norm1(x)
        fwd_attention = fwd_norm1 + self.attention(fwd_norm1, fwd_norm1, fwd_norm1)
        fwd_norm2 = self.norm2(fwd_attention)
        x = fwd_attention + self.mlp(fwd_norm2)
        return x
