"""
A compact transformer bi-encoder trained from scratch for medical
domain sentence embeddings. Small enough to train on CPU, self-contained
(no pretrained weights pulled from any external hub).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MedicalEmbeddingModel(nn.Module):
    """Encodes text -> a single fixed-size normalized embedding vector."""

    def __init__(self, vocab_size, dim=192, n_heads=4, n_layers=4, ff_dim=512,
                 max_len=256, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, input_ids):
        mask = (input_ids == self.pad_id)  # True where padded
        x = self.token_emb(input_ids) * math.sqrt(self.dim)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        # mean-pool over non-padded tokens
        valid = (~mask).unsqueeze(-1).float()
        summed = (x * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts

        out = self.proj(pooled)
        out = F.normalize(out, p=2, dim=-1)
        return out
