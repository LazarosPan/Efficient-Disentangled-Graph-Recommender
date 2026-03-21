"""L_rec: BPR (Bayesian Personalized Ranking) loss.

Uses PyG's ``BPRLoss`` for the unweighted case; extends with IPW weighting
for the causal variant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn.models.lightgcn import BPRLoss as _PyGBPRLoss

_bpr_unweighted = _PyGBPRLoss(lambda_reg=0.0)


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute BPR ranking loss: -mean(log(sigmoid(s_pos - s_neg))).

    Args:
        pos_scores: (B,) scores for positive items.
        neg_scores: (B,) scores for negative items.
        weights: (B,) optional per-sample weights (e.g., IPW).

    Returns:
        Scalar loss.
    """
    if weights is None:
        return _bpr_unweighted(pos_scores, neg_scores)

    # IPW-weighted variant (U-CaGNN-specific, not in PyG)
    loss = -F.logsigmoid(pos_scores - neg_scores)
    return (loss * weights).mean()
