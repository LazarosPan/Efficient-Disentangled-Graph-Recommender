"""L_cf: Counterfactual divergence loss."""

from __future__ import annotations

import torch


def counterfactual_loss(
    interest_scores: torch.Tensor,
    conformity_scores: torch.Tensor,
) -> torch.Tensor:
    """Encourage divergence between factual (interest) and counterfactual (conformity) scores.

    L_cf = mean(||Y_interest - Y_conformity||^2)

    This pushes the two scoring branches to capture different signals,
    ensuring the counterfactual score (interest - conformity) is meaningful.

    Args:
        interest_scores: (B,) dot-product scores from interest branch.
        conformity_scores: (B,) dot-product scores from conformity branch.

    Returns:
        Scalar loss.
    """
    diff = interest_scores - conformity_scores
    return (diff ** 2).mean()
