"""L_ortho: Cosine orthogonality loss between interest and conformity embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def orthogonality_loss(
    interest: torch.Tensor,
    conformity: torch.Tensor,
) -> torch.Tensor:
    """Encourage independence between interest and conformity representations.

    L_ortho = mean(cos_sim(interest, conformity)^2)

    This is O(B*d) — much cheaper than dCor's O(B^2*d).

    Args:
        interest: (B, D) interest embeddings.
        conformity: (B, D) conformity embeddings.

    Returns:
        Scalar loss.
    """
    cos_sim = F.cosine_similarity(interest, conformity, dim=-1)
    return (cos_sim**2).mean()
