from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, num_buckets: int = 3) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_embedding = nn.Embedding(num_buckets + 1, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
        )
        self.policy_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.value_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.shared_policy_head = nn.Linear(hidden_dim, action_dim)
        self.bucket_policy_heads = nn.ModuleList(
            nn.Linear(hidden_dim, action_dim) for _ in range(num_buckets)
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        for head in self.bucket_policy_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        bucket_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(obs)
        bucket_embed = torch.zeros_like(hidden)
        if bucket_ids is not None:
            if bucket_ids.dim() == 0:
                bucket_ids = bucket_ids.unsqueeze(0)
            bucket_ids = bucket_ids.to(device=hidden.device, dtype=torch.long).clamp(0, self.num_buckets - 1)
            bucket_embed = self.bucket_embedding(bucket_ids)

        policy_hidden = self.policy_tower(hidden + bucket_embed)
        logits = self.shared_policy_head(policy_hidden)
        if bucket_ids is not None:
            bucket_delta = torch.zeros_like(logits)
            for bucket_idx, head in enumerate(self.bucket_policy_heads):
                mask = bucket_ids == bucket_idx
                if mask.any():
                    bucket_delta[mask] = head(policy_hidden[mask])
            logits = logits + bucket_delta
        value = self.value_head(self.value_tower(hidden + bucket_embed)).squeeze(-1)
        return logits, value
