"""LossSuite: combined multi-task loss with configurable lambdas and optional curriculum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.lightgcn import BPRLoss as _PyGBPRLoss

from ..utils.config import UCaGNNConfig

_bpr_unweighted = _PyGBPRLoss(lambda_reg=0.0)


def _bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the ranking loss, optionally reweighted by IPW."""
    if weights is None:
        return _bpr_unweighted(pos_scores, neg_scores)

    loss = -F.logsigmoid(pos_scores - neg_scores)
    return (loss * weights).mean()


def _orthogonality_loss(
    interest: torch.Tensor,
    conformity: torch.Tensor,
) -> torch.Tensor:
    """Penalize correlation between interest and conformity embeddings."""
    cos_sim = F.cosine_similarity(interest, conformity, dim=-1)
    return (cos_sim**2).mean()


def _contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Compute NT-Xent loss between the two user-view batches."""
    batch_size = z_i.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=z_i.device, requires_grad=True)

    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)
    z = torch.cat([z_i, z_j], dim=0)

    sim = z @ z.t() / tau
    diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)
    pos_idx = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (
        2 * batch_size
    )
    return F.cross_entropy(sim, pos_idx)


def _counterfactual_loss(
    interest_scores: torch.Tensor,
    conformity_scores: torch.Tensor,
) -> torch.Tensor:
    """Encourage a non-trivial gap between the two scoring branches."""
    diff = interest_scores - conformity_scores
    return (diff**2).mean()


class _PopularityPredictor(nn.Module):
    """Predict item popularity from conformity embeddings."""

    def __init__(self, embed_dim: int, pop_embed_dim: int = 16) -> None:
        """Initialize the internal popularity prediction head."""
        super().__init__()
        self.proj = nn.Linear(embed_dim, pop_embed_dim)

    def forward(self, conformity_emb: torch.Tensor) -> torch.Tensor:
        """Map conformity embeddings to scalar popularity predictions."""
        return self.proj(conformity_emb).mean(dim=-1)


def _popularity_loss(
    pop_pred: torch.Tensor,
    pop_target: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between predicted and observed popularity."""
    return F.mse_loss(pop_pred, pop_target)


class LossSuite(nn.Module):
    """Combines all 5 loss terms with configurable lambda weights.

    L_total = lambda_rec * L_rec
            + lambda_ortho * L_ortho
            + lambda_contr * L_contr
            + lambda_cf * L_cf
            + lambda_pop * L_pop

    Optional curriculum: losses are phased in based on epoch thresholds.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_dual_branch and config.lambda_pop > 0:
            self.pop_predictor = _PopularityPredictor(
                config.embed_dim,
                config.pop_embed_dim,
            )

    def forward(
        self,
        model_output: dict[str, torch.Tensor | dict],
        item_popularity: torch.Tensor,
        pos_item_ids: torch.Tensor,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute all active losses.

        Args:
            model_output: Output from UCaGNN.forward().
            item_popularity: (I,) normalized popularity array.
            pos_item_ids: (B,) positive item indices (for popularity lookup).
            epoch: Current epoch (for curriculum scheduling).

        Returns:
            Dict with individual losses and 'total' combined loss.
        """
        cfg = self.config
        pos_scores = model_output["pos_scores"]
        neg_scores = model_output["neg_scores"]
        propagated = model_output["propagated"]
        ipw_weights = model_output["ipw_weights"]

        losses: dict[str, torch.Tensor] = {}
        device = pos_scores["final_score"].device
        zero = torch.tensor(0.0, device=device)

        # L_rec: BPR (always active)
        weights = ipw_weights if cfg.use_ipw else None
        losses["rec"] = _bpr_loss(
            pos_scores["final_score"],
            neg_scores["final_score"],
            weights,
        )

        # Curriculum: check phase thresholds
        phase2_active = epoch >= cfg.curriculum_phase1_end
        phase3_active = epoch >= cfg.curriculum_phase2_end

        # L_ortho: orthogonality (phase 2+)
        if cfg.use_dual_branch and cfg.lambda_ortho > 0 and phase2_active:
            losses["ortho"] = _orthogonality_loss(
                propagated["user_interest"],
                propagated["user_conformity"],
            )
        else:
            losses["ortho"] = zero

        # L_contr: contrastive (phase 2+)
        if cfg.use_dual_branch and cfg.lambda_contr > 0 and phase2_active:
            losses["contr"] = _contrastive_loss(
                propagated["user_interest"],
                propagated["user_conformity"],
                tau=cfg.contrastive_tau,
            )
        else:
            losses["contr"] = zero

        # L_cf: counterfactual divergence (phase 3+)
        if cfg.use_dual_branch and cfg.lambda_cf > 0 and phase3_active:
            losses["cf"] = _counterfactual_loss(
                pos_scores["interest_score"],
                pos_scores["conformity_score"],
            )
        else:
            losses["cf"] = zero

        # L_pop: popularity (phase 3+)
        if cfg.use_dual_branch and cfg.lambda_pop > 0 and phase3_active:
            conf_emb = propagated["item_conformity"][pos_item_ids]
            pop_pred = self.pop_predictor(conf_emb)
            pop_target = item_popularity[pos_item_ids].to(device)
            losses["pop"] = _popularity_loss(pop_pred, pop_target)
        else:
            losses["pop"] = zero

        # Weighted sum
        total = (
            cfg.lambda_rec * losses["rec"]
            + cfg.lambda_ortho * losses["ortho"]
            + cfg.lambda_contr * losses["contr"]
            + cfg.lambda_cf * losses["cf"]
            + cfg.lambda_pop * losses["pop"]
        )
        losses["total"] = total

        return losses
