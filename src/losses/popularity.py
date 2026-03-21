"""L_pop: Popularity bias removal loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PopularityPredictor(nn.Module):
    """Predicts item popularity from conformity embeddings.

    The loss encourages the conformity branch to capture popularity signal,
    while the interest branch focuses on genuine preference.
    """

    def __init__(self, embed_dim: int, pop_embed_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, pop_embed_dim)

    def forward(self, conformity_emb: torch.Tensor) -> torch.Tensor:
        """Predict popularity scores from conformity embeddings.

        Args:
            conformity_emb: (B, D) conformity item embeddings.

        Returns:
            (B,) predicted popularity scores.
        """
        return self.proj(conformity_emb).mean(dim=-1)


def popularity_loss(
    pop_pred: torch.Tensor,
    pop_target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between predicted and actual item popularity.

    Args:
        pop_pred: (B,) predicted popularity from conformity branch.
        pop_target: (B,) ground-truth normalized popularity.

    Returns:
        Scalar loss.
    """
    return F.mse_loss(pop_pred, pop_target)
