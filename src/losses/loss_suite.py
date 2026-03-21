"""LossSuite: combined multi-task loss with configurable lambdas and optional curriculum."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import UCaGNNConfig
from .bpr import bpr_loss
from .orthogonality import orthogonality_loss
from .contrastive import contrastive_loss
from .counterfactual import counterfactual_loss
from .popularity import PopularityPredictor, popularity_loss


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
            self.pop_predictor = PopularityPredictor(
                config.embed_dim, config.pop_embed_dim,
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
        losses["rec"] = bpr_loss(
            pos_scores["final_score"], neg_scores["final_score"], weights,
        )

        # Curriculum: check phase thresholds
        phase2_active = epoch >= cfg.curriculum_phase1_end
        phase3_active = epoch >= cfg.curriculum_phase2_end

        # L_ortho: orthogonality (phase 2+)
        if cfg.use_dual_branch and cfg.lambda_ortho > 0 and phase2_active:
            losses["ortho"] = orthogonality_loss(
                propagated["user_interest"], propagated["user_conformity"],
            )
        else:
            losses["ortho"] = zero

        # L_contr: contrastive (phase 2+)
        if cfg.use_dual_branch and cfg.lambda_contr > 0 and phase2_active:
            losses["contr"] = contrastive_loss(
                propagated["user_interest"],
                propagated["user_conformity"],
                tau=cfg.contrastive_tau,
            )
        else:
            losses["contr"] = zero

        # L_cf: counterfactual divergence (phase 3+)
        if cfg.use_dual_branch and cfg.lambda_cf > 0 and phase3_active:
            losses["cf"] = counterfactual_loss(
                pos_scores["interest_score"], pos_scores["conformity_score"],
            )
        else:
            losses["cf"] = zero

        # L_pop: popularity (phase 3+)
        if cfg.use_dual_branch and cfg.lambda_pop > 0 and phase3_active:
            conf_emb = propagated["item_conformity"][pos_item_ids]
            pop_pred = self.pop_predictor(conf_emb)
            pop_target = item_popularity[pos_item_ids].to(device)
            losses["pop"] = popularity_loss(pop_pred, pop_target)
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
