"""Module C: Interest + conformity + counterfactual dot-product scoring."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import UCaGNNConfig


class ScoringModule(nn.Module):
    """Compute final recommendation scores from propagated embeddings.

    score = alpha * interest_score + beta * conformity_score + gamma * cf_score

    Where cf_score = interest_score - conformity_score (counterfactual difference).
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score user-item pairs.

        Args:
            propagated: Dict of propagated embeddings from DualBranchGCN.
            user_ids: (B,) user indices.
            item_ids: (B,) item indices.

        Returns:
            Dict with 'interest_score', 'conformity_score', 'cf_score', 'final_score'.
        """
        scores: dict[str, torch.Tensor] = {}

        if self.config.use_dual_branch:
            u_int = propagated["user_interest"][user_ids]
            i_int = propagated["item_interest"][item_ids]
            interest_score = (u_int * i_int).sum(dim=-1)
            scores["interest_score"] = interest_score

            u_conf = propagated["user_conformity"][user_ids]
            i_conf = propagated["item_conformity"][item_ids]
            conformity_score = (u_conf * i_conf).sum(dim=-1)
            scores["conformity_score"] = conformity_score

            if self.config.use_counterfactual:
                cf_score = interest_score - conformity_score
                scores["cf_score"] = cf_score
            else:
                scores["cf_score"] = torch.zeros_like(interest_score)

            final = (
                self.config.alpha_interest * interest_score
                + self.config.beta_conformity * conformity_score
                + self.config.gamma_counterfactual * scores["cf_score"]
            )
        else:
            u = propagated["user"][user_ids]
            i = propagated["item"][item_ids]
            interest_score = (u * i).sum(dim=-1)
            scores["interest_score"] = interest_score
            scores["conformity_score"] = torch.zeros_like(interest_score)
            scores["cf_score"] = torch.zeros_like(interest_score)
            final = interest_score

        scores["final_score"] = final
        return scores
