"""Minimal masked point encoder for the V1-R simple-baseline matrix."""

from __future__ import annotations

try:  # Optional dependency: the rest of V1-R remains dependency-free.
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only without PyTorch
    torch = None
    nn = None


class MaskedPointEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    """PointNet-like max pool that never lets hidden points enter the pool."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128) -> None:
        if torch is None or nn is None:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("MaskedPointEncoder requires PyTorch at runtime")
        super().__init__()
        self._module = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.unknown_token = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, points, visible):
        features = self._module(points)
        mask = visible.to(dtype=torch.bool).unsqueeze(-1)
        masked = features.masked_fill(~mask, float("-inf"))
        pooled = masked.max(dim=1).values
        all_hidden = ~visible.to(dtype=torch.bool).any(dim=1)
        return torch.where(all_hidden.unsqueeze(-1), self.unknown_token.expand_as(pooled), pooled)
