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
        self.component_names = ("interest", "conformity", "counterfactual")

        if config.scoring_weight_mode == "learned" and config.use_dual_branch:
            initial_weights = torch.tensor(
                [
                    config.alpha_interest,
                    config.beta_conformity,
                    config.gamma_counterfactual,
                ],
                dtype=torch.float32,
            )
            initial_weights = initial_weights.clamp_min(1e-6)
            initial_weights = initial_weights / initial_weights.sum()
            self.score_weight_logits = nn.Parameter(initial_weights.log())
        else:
            self.register_parameter("score_weight_logits", None)

    def _fixed_weight_tensor(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                self.config.alpha_interest,
                self.config.beta_conformity,
                self.config.gamma_counterfactual,
            ],
            device=device,
            dtype=dtype,
        )

    def _mode_mask(
        self,
        scoring_mode: str,
        use_counterfactual: bool,
    ) -> torch.Tensor:
        if scoring_mode == "default":
            return torch.tensor([True, True, use_counterfactual])
        if scoring_mode == "interest_only":
            return torch.tensor([True, False, False])
        if scoring_mode == "conformity_only":
            return torch.tensor([False, True, False])
        if scoring_mode == "counterfactual_only":
            return torch.tensor([False, False, use_counterfactual])
        if scoring_mode == "conformity_suppressed":
            return torch.tensor([True, False, use_counterfactual])
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

    def get_score_weight_tensor(
        self,
        scoring_mode: str = "default",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self.config.use_dual_branch:
            resolved_device = device or torch.device("cpu")
            resolved_dtype = dtype or torch.float32
            return torch.tensor(
                [1.0, 0.0, 0.0], device=resolved_device, dtype=resolved_dtype
            )

        if self.config.scoring_weight_mode == "fixed":
            resolved_device = device or torch.device("cpu")
            resolved_dtype = dtype or torch.float32
            base_weights = self._fixed_weight_tensor(resolved_device, resolved_dtype)
        else:
            if self.score_weight_logits is None:
                raise RuntimeError(
                    "Learned scoring weights requested without score_weight_logits"
                )
            base_weights = self.score_weight_logits
            if device is not None or dtype is not None:
                base_weights = base_weights.to(
                    device=device or base_weights.device,
                    dtype=dtype or base_weights.dtype,
                )

        mask = self._mode_mask(scoring_mode, self.config.use_counterfactual).to(
            base_weights.device
        )
        if self.config.scoring_weight_mode == "fixed":
            weights = torch.where(mask, base_weights, torch.zeros_like(base_weights))
            return weights

        active_logits = base_weights[mask]
        if active_logits.numel() == 0:
            return torch.zeros_like(base_weights)

        active_weights = torch.softmax(active_logits, dim=0)
        weights = torch.zeros_like(base_weights)
        weights[mask] = active_weights
        return weights

    def get_score_weight_summary(
        self, scoring_mode: str = "default"
    ) -> dict[str, float]:
        weights = (
            self.get_score_weight_tensor(scoring_mode=scoring_mode)
            .detach()
            .cpu()
            .tolist()
        )
        return {
            f"score_weight_{name}": float(weight)
            for name, weight in zip(self.component_names, weights, strict=True)
        }

    def combine_scores(
        self,
        interest_score: torch.Tensor,
        conformity_score: torch.Tensor,
        cf_score: torch.Tensor,
        scoring_mode: str = "default",
    ) -> torch.Tensor:
        """Combine branch scores for ranking or intervention-style evaluation."""
        weights = self.get_score_weight_tensor(
            scoring_mode=scoring_mode,
            device=interest_score.device,
            dtype=interest_score.dtype,
        )
        return (
            weights[0] * interest_score
            + weights[1] * conformity_score
            + weights[2] * cf_score
        )

    def forward(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        scoring_mode: str = "default",
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

            final = self.combine_scores(
                interest_score,
                conformity_score,
                scores["cf_score"],
                scoring_mode=scoring_mode,
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
