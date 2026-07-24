from __future__ import annotations

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state)
        value = self.value(encoded)
        advantage = self.advantage(encoded)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
