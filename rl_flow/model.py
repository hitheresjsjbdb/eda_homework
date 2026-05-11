from __future__ import annotations

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        logits = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return logits, value
