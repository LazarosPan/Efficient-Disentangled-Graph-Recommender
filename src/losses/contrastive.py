"""L_contr: NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """NT-Xent contrastive loss between two views of the same entities.

    Positive pairs: (z_i[k], z_j[k]) for each k.
    Negative pairs: all other combinations within the batch.

    Args:
        z_i: (B, D) first view embeddings (e.g., interest).
        z_j: (B, D) second view embeddings (e.g., conformity).
        tau: Temperature parameter.

    Returns:
        Scalar loss.
    """
    B = z_i.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=z_i.device, requires_grad=True)

    # Normalize
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    # Concatenate both views: [z_i; z_j] -> (2B, D)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

    # Pairwise similarity
    sim = z @ z.t() / tau  # (2B, 2B)

    # Mask out self-similarity while keeping the positive pair logit intact.
    diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)

    # Positive pairs: anchor i matches i+B for the first half, and i-B for the second half.
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)

    return F.cross_entropy(sim, pos_idx)
