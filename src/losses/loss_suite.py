"""LossSuite: fused BPR plus branch-local auxiliaries for wave-1 U-CaGNN v2."""

from __future__ import annotations

from typing import cast

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


def _independence_loss(
    interest: torch.Tensor,
    conformity: torch.Tensor,
) -> torch.Tensor:
    """Penalize correlation between interest and conformity embeddings."""
    cos_sim = F.cosine_similarity(interest, conformity, dim=-1)
    return (cos_sim**2).mean()


def _within_branch_contrastive_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    temperature: float,
    max_pairs: int,
) -> torch.Tensor:
    """Compute a sampled within-branch contrastive loss on positive pairs.

    Args:
        user_embeddings: Positive-pair user embeddings of shape ``(B, D)``.
        item_embeddings: Positive-pair item embeddings of shape ``(B, D)``.
        temperature: Softmax temperature for the similarity logits.
        max_pairs: Maximum number of aligned positive pairs used for the loss.

    Returns:
        Scalar symmetric InfoNCE loss. Returns zero if fewer than two pairs are
        available.
    """
    pair_count = min(user_embeddings.size(0), item_embeddings.size(0), max_pairs)
    if pair_count <= 1:
        return user_embeddings.new_zeros(())

    user_view = F.normalize(user_embeddings[:pair_count].float(), dim=-1)
    item_view = F.normalize(item_embeddings[:pair_count].float(), dim=-1)
    logits = user_view @ item_view.t()
    logits = logits / temperature
    labels = torch.arange(pair_count, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _directau_alignment_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """DirectAU-style alignment loss on normalized positive user-item pairs."""
    if user_embeddings.size(0) == 0:
        return user_embeddings.new_zeros(())
    user_norm = F.normalize(user_embeddings, dim=-1)
    item_norm = F.normalize(item_embeddings, dim=-1)
    return (user_norm - item_norm).pow(2).sum(dim=-1).mean()


def _directau_uniformity_loss(
    embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """DirectAU-style uniformity loss on normalized embeddings."""
    if embeddings.size(0) <= 1:
        return embeddings.new_zeros(())
    normalized = F.normalize(embeddings, dim=-1)
    pairwise_dist = torch.pdist(normalized, p=2)
    if pairwise_dist.numel() == 0:
        return embeddings.new_zeros(())
    return torch.log(torch.exp(-temperature * pairwise_dist.pow(2)).mean() + 1e-8)


def _popularity_loss(
    pop_pred: torch.Tensor,
    pop_target: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between predicted and observed popularity."""
    return F.mse_loss(pop_pred, pop_target)


class LossSuite(nn.Module):
    """Combine fused ranking loss with branch-local auxiliary objectives."""

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config

    def _resolve_auxiliary_weight(
        self,
        lambda_max: float,
        epoch: int,
        *,
        ramp_rate: float,
        active_in_phased_schedule: bool,
    ) -> float:
        """Resolve one auxiliary-loss weight under the configured schedule.

        Args:
            lambda_max: Configured maximum loss weight.
            epoch: Current zero-based epoch.
            ramp_rate: Linear ramp slope used when the schedule is
                ``linear_ramp``.
            active_in_phased_schedule: Whether the loss is active under the
                phased schedule.

        Returns:
            Effective scalar weight for the current epoch.
        """
        if lambda_max <= 0:
            return 0.0
        if self.config.auxiliary_loss_schedule == "linear_ramp":
            return min(lambda_max, ramp_rate * max(epoch, 0))
        return lambda_max if active_in_phased_schedule else 0.0

    def forward(
        self,
        model_output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
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
        pos_scores = cast(dict[str, torch.Tensor], model_output["pos_scores"])
        neg_scores = cast(dict[str, torch.Tensor], model_output["neg_scores"])
        propagated = cast(dict[str, torch.Tensor], model_output["propagated"])
        ipw_weights = cast(torch.Tensor, model_output["ipw_weights"])
        loss_user_ids = cast(torch.Tensor, model_output["loss_user_ids"])

        losses: dict[str, torch.Tensor] = {}
        reference_score = pos_scores["final_score"]
        zero = reference_score.new_zeros(())

        # Curriculum: check phase thresholds
        phase2_active = epoch >= cfg.curriculum_phase1_end
        phase3_active = epoch >= cfg.curriculum_phase2_end
        use_dual_branch = cfg.use_dual_branch

        # Fused BPR is always active from epoch 0; only auxiliary losses phase in.
        # L_rec: fused BPR on the final score
        weights = ipw_weights if cfg.use_ipw else None
        losses["rec"] = _bpr_loss(
            pos_scores["final_score"],
            neg_scores["final_score"],
            weights,
        )

        interest_weight = self._resolve_auxiliary_weight(
            cfg.lambda_interest_bpr,
            epoch,
            ramp_rate=cfg.auxiliary_ramp_rate,
            active_in_phased_schedule=True,
        )
        conformity_weight = self._resolve_auxiliary_weight(
            cfg.lambda_conformity_bpr,
            epoch,
            ramp_rate=cfg.auxiliary_ramp_rate,
            active_in_phased_schedule=True,
        )
        independence_weight = self._resolve_auxiliary_weight(
            cfg.lambda_independence,
            epoch,
            ramp_rate=cfg.independence_ramp_rate,
            active_in_phased_schedule=phase2_active,
        )
        contrastive_weight = self._resolve_auxiliary_weight(
            cfg.lambda_contrastive,
            epoch,
            ramp_rate=cfg.auxiliary_ramp_rate,
            active_in_phased_schedule=phase2_active,
        )
        align_weight = self._resolve_auxiliary_weight(
            cfg.lambda_align,
            epoch,
            ramp_rate=cfg.auxiliary_ramp_rate,
            active_in_phased_schedule=phase2_active,
        )
        uniform_weight = self._resolve_auxiliary_weight(
            cfg.lambda_uniform,
            epoch,
            ramp_rate=cfg.auxiliary_ramp_rate,
            active_in_phased_schedule=phase2_active,
        )
        popularity_weight = self._resolve_auxiliary_weight(
            cfg.lambda_pop,
            epoch,
            ramp_rate=cfg.auxiliary_ramp_rate,
            active_in_phased_schedule=phase3_active,
        )

        # Branch-local BPR auxiliaries keep each branch predictive on its own.
        if use_dual_branch and interest_weight > 0:
            losses["interest_bpr"] = _bpr_loss(
                pos_scores["interest_score"],
                neg_scores["interest_score"],
            )
        else:
            losses["interest_bpr"] = zero

        if use_dual_branch and conformity_weight > 0:
            losses["conformity_bpr"] = _bpr_loss(
                pos_scores["conformity_score"],
                neg_scores["conformity_score"],
            )
        else:
            losses["conformity_bpr"] = zero

        # Branch independence via cosine-squared decorrelation.
        if use_dual_branch and independence_weight > 0:
            losses["independence"] = _independence_loss(
                propagated["user_interest"][loss_user_ids],
                propagated["user_conformity"][loss_user_ids],
            )
        else:
            losses["independence"] = zero

        # Branch-local contrastive regularization on aligned positive pairs.
        if use_dual_branch and contrastive_weight > 0:
            branch_losses = [
                _within_branch_contrastive_loss(
                    propagated["user_interest"][loss_user_ids],
                    propagated["item_interest"][pos_item_ids],
                    temperature=cfg.contrastive_temperature,
                    max_pairs=cfg.contrastive_max_pairs,
                ),
                _within_branch_contrastive_loss(
                    propagated["user_conformity"][loss_user_ids],
                    propagated["item_conformity"][pos_item_ids],
                    temperature=cfg.contrastive_temperature,
                    max_pairs=cfg.contrastive_max_pairs,
                ),
            ]
            losses["contrastive"] = torch.stack(branch_losses).mean()
        else:
            losses["contrastive"] = zero

        # DirectAU-style branch-local geometry lives on positive user-item pairs.
        if use_dual_branch and (align_weight > 0 or uniform_weight > 0):
            branch_align_losses: list[torch.Tensor] = []
            branch_uniform_losses: list[torch.Tensor] = []

            interest_users = propagated["user_interest"][loss_user_ids]
            interest_items = propagated["item_interest"][pos_item_ids]
            branch_align_losses.append(
                _directau_alignment_loss(interest_users, interest_items)
            )
            branch_uniform_losses.append(
                0.5
                * (
                    _directau_uniformity_loss(
                        interest_users,
                        temperature=cfg.uniformity_temperature,
                    )
                    + _directau_uniformity_loss(
                        interest_items,
                        temperature=cfg.uniformity_temperature,
                    )
                )
            )

            if cfg.use_conformity_au:
                conformity_users = propagated["user_conformity"][loss_user_ids]
                conformity_items = propagated["item_conformity"][pos_item_ids]
                branch_align_losses.append(
                    _directau_alignment_loss(
                        conformity_users,
                        conformity_items,
                    )
                )
                branch_uniform_losses.append(
                    0.5
                    * (
                        _directau_uniformity_loss(
                            conformity_users,
                            temperature=cfg.uniformity_temperature,
                        )
                        + _directau_uniformity_loss(
                            conformity_items,
                            temperature=cfg.uniformity_temperature,
                        )
                    )
                )

            losses["align"] = torch.stack(branch_align_losses).mean()
            losses["uniform"] = torch.stack(branch_uniform_losses).mean()
        else:
            losses["align"] = zero
            losses["uniform"] = zero

        # Popularity regression now supervises the scorer-owned popularity head.
        if use_dual_branch and cfg.use_popularity_head and popularity_weight > 0:
            pop_target = item_popularity[pos_item_ids].to(reference_score.device)
            losses["pop"] = _popularity_loss(
                pos_scores["popularity_score"],
                pop_target,
            )
        else:
            losses["pop"] = zero

        # Weighted sum
        total = (
            cfg.lambda_rec * losses["rec"]
            + interest_weight * losses["interest_bpr"]
            + conformity_weight * losses["conformity_bpr"]
            + independence_weight * losses["independence"]
            + contrastive_weight * losses["contrastive"]
            + align_weight * losses["align"]
            + uniform_weight * losses["uniform"]
            + popularity_weight * losses["pop"]
        )
        losses["total"] = total

        return losses
