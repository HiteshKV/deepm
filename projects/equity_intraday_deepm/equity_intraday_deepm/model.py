"""Small DeePM-style hourly equity models and objective helpers."""

from __future__ import annotations

import math

import torch
from torch import nn


def sharpe_score(returns: torch.Tensor, bars_per_year: float, eps: float = 1e-8) -> torch.Tensor:
    if returns.numel() == 0:
        return torch.tensor(0.0, device=returns.device)
    return returns.mean() / (returns.std(unbiased=False) + eps) * math.sqrt(float(bars_per_year))


def softmin_sharpe_loss(
    returns: torch.Tensor,
    bars_per_year: float,
    beta: float = 5.0,
    groups: int = 4,
) -> torch.Tensor:
    if returns.numel() < groups * 2:
        return -sharpe_score(returns, bars_per_year)
    chunks = torch.chunk(returns, groups)
    sharpes = torch.stack([sharpe_score(chunk, bars_per_year) for chunk in chunks if chunk.numel() > 1])
    return torch.logsumexp(-beta * sharpes, dim=0) / beta


class EquityHourlyModel(nn.Module):
    """Multi-input/multi-output hourly equity signal model."""

    def __init__(
        self,
        input_dim: int,
        active_top_n: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_heads: int = 4,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.active_top_n = active_top_n
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [batch, assets, seq, features], got {tuple(x.shape)}")
        batch, assets, seq_len, features = x.shape
        encoded, _ = self.encoder(x.reshape(batch * assets, seq_len, features))
        hidden = self.dropout(encoded[:, -1]).reshape(batch, assets, self.hidden_dim)
        if self.use_cross_attention:
            attn, _ = self.cross_attn(hidden, hidden, hidden, need_weights=False)
            hidden = self.norm(hidden + self.dropout(attn))
        return self.head(hidden).squeeze(-1)
