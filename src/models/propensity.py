"""Module F: IPW propensity estimator (2-layer MLP)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import UCaGNNConfig


class PropensityEstimator(nn.Module):
    """Estimate item propensity P(exposure | item) for inverse propensity weighting.

    Architecture: item_embedding → Linear(D, hidden) → ReLU → Linear(hidden, 1) → Sigmoid
    Output clipped to [clip_min, clip_max] for numerical stability.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.propensity_hidden),
            nn.ReLU(),
            nn.Linear(config.propensity_hidden, 1),
            nn.Sigmoid(),
        )
        self.clip_min = config.propensity_clip_min
        self.clip_max = config.propensity_clip_max

    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Estimate propensity scores.

        Args:
            item_embeddings: (B, D) item embeddings.

        Returns:
            (B,) propensity scores clipped to [clip_min, clip_max].
        """
        raw = self.mlp(item_embeddings).squeeze(-1)
        return torch.clamp(raw, self.clip_min, self.clip_max)
