"""LossSuite: fused BPR plus branch-local auxiliaries for U-CaGNN."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional

from ..utils.config import UCaGNNConfig


def _bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the ranking loss, optionally reweighted by IPW."""
    loss = -functional.logsigmoid(pos_scores - neg_scores)
    return (loss * weights).mean() if weights is not None else loss.mean()


def _independence_loss(
    interest: torch.Tensor,
    conformity: torch.Tensor,
) -> torch.Tensor:
    """Penalize correlation between interest and conformity embeddings."""
    cos_sim = functional.cosine_similarity(interest, conformity, dim=-1)
    return (cos_sim**2).mean()


def _prepare_contrastive_pairs(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    item_popularity: torch.Tensor,
    pos_item_ids: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cap and cast aligned positive pairs for contrastive loss computation.

    Args:
        user_embeddings: Positive-pair user embeddings of shape ``(B, D)``.
        item_embeddings: Positive-pair item embeddings of shape ``(B, D)``.
        item_popularity: Item-popularity tensor used for positive-pair lookup.
        pos_item_ids: Positive item ids aligned with the batch users.
        max_pairs: Maximum number of aligned positive pairs used for the loss.

    Returns:
        Tuple of capped ``(users, items, item_ids, popularity)`` tensors.

    """
    pair_count = min(user_embeddings.size(0), item_embeddings.size(0), max_pairs)
    return (
        user_embeddings[:pair_count].float(),
        item_embeddings[:pair_count].float(),
        pos_item_ids[:pair_count],
        item_popularity[pos_item_ids[:pair_count]].float(),
    )


def _branch_contrastive_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    positive_weights: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute one DCCL-style branch contrastive loss over batch-local negatives.

    Args:
        user_embeddings: Capped user embeddings of shape ``(B, D)``.
        item_embeddings: Capped positive-item embeddings of shape ``(B, D)``.
        positive_weights: Per-pair positive weights of shape ``(B,)``.
        negative_mask: Eligibility mask over batch-local negative items with
            shape ``(B, B)``.
        temperature: Softmax temperature applied to the dot-product logits.

    Returns:
        Scalar loss averaged over anchors with at least one eligible negative.

    """
    if user_embeddings.size(0) <= 1:
        return user_embeddings.new_zeros(())

    valid_rows = negative_mask.any(dim=1)
    if not valid_rows.any():
        return user_embeddings.new_zeros(())

    logits = (user_embeddings @ item_embeddings.t()) / temperature
    positive_logits = logits.diag()
    negative_logits = logits.masked_fill(~negative_mask, float("-inf"))
    negative_logsumexp = torch.logsumexp(negative_logits[valid_rows], dim=1)
    log_denom = torch.logaddexp(positive_logits[valid_rows], negative_logsumexp)
    log_positive_weights = torch.log(positive_weights[valid_rows].clamp_min(1e-8))
    return -(log_positive_weights + positive_logits[valid_rows] - log_denom).mean()


def _interest_contrastive_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    item_popularity: torch.Tensor,
    pos_item_ids: torch.Tensor,
    temperature: float,
    max_pairs: int,
) -> torch.Tensor:
    """Compute the interest contrastive loss with inverse-popularity weighting.

    Args:
        user_embeddings: Positive-pair user embeddings of shape ``(B, D)``.
        item_embeddings: Positive-pair item embeddings of shape ``(B, D)``.
        item_popularity: Item-popularity tensor used for positive-pair lookup.
        pos_item_ids: Positive item ids aligned with the batch users.
        temperature: Softmax temperature applied to the dot-product logits.
        max_pairs: Maximum number of aligned positive pairs used for the loss.

    Returns:
        Scalar interest contrastive loss.

    """
    users, items, pair_item_ids, pair_popularity = _prepare_contrastive_pairs(
        user_embeddings,
        item_embeddings,
        item_popularity,
        pos_item_ids,
        max_pairs,
    )
    distinct_item_mask = pair_item_ids.unsqueeze(1) != pair_item_ids.unsqueeze(0)
    positive_weights = torch.exp(-pair_popularity)
    return _branch_contrastive_loss(
        users,
        items,
        positive_weights,
        distinct_item_mask,
        temperature,
    )


def _conformity_contrastive_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    item_popularity: torch.Tensor,
    pos_item_ids: torch.Tensor,
    temperature: float,
    max_pairs: int,
) -> torch.Tensor:
    """Compute the conformity contrastive loss with popularity-aware negatives.

    Args:
        user_embeddings: Positive-pair user embeddings of shape ``(B, D)``.
        item_embeddings: Positive-pair item embeddings of shape ``(B, D)``.
        item_popularity: Item-popularity tensor used for positive-pair lookup.
        pos_item_ids: Positive item ids aligned with the batch users.
        temperature: Softmax temperature applied to the dot-product logits.
        max_pairs: Maximum number of aligned positive pairs used for the loss.

    Returns:
        Scalar conformity contrastive loss.

    """
    users, items, pair_item_ids, pair_popularity = _prepare_contrastive_pairs(
        user_embeddings,
        item_embeddings,
        item_popularity,
        pos_item_ids,
        max_pairs,
    )
    distinct_item_mask = pair_item_ids.unsqueeze(1) != pair_item_ids.unsqueeze(0)
    higher_popularity_mask = pair_popularity.unsqueeze(0) > pair_popularity.unsqueeze(1)
    positive_weights = 1.0 - torch.exp(-pair_popularity)
    return _branch_contrastive_loss(
        users,
        items,
        positive_weights,
        distinct_item_mask & higher_popularity_mask,
        temperature,
    )


def _directau_alignment_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """DirectAU-style alignment loss on normalized positive user-item pairs."""
    if user_embeddings.size(0) == 0:
        return user_embeddings.new_zeros(())
    user_norm = functional.normalize(user_embeddings, dim=-1)
    item_norm = functional.normalize(item_embeddings, dim=-1)
    return (user_norm - item_norm).pow(2).sum(dim=-1).mean()


def _directau_uniformity_loss(
    embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """DirectAU-style uniformity loss on normalized embeddings."""
    if embeddings.size(0) <= 1:
        return embeddings.new_zeros(())
    normalized = functional.normalize(embeddings, dim=-1)
    pairwise_dist = torch.pdist(normalized, p=2)
    if pairwise_dist.numel() == 0:
        return embeddings.new_zeros(())
    return torch.log(torch.exp(-temperature * pairwise_dist.pow(2)).mean() + 1e-8)


def _popularity_loss(
    pop_pred: torch.Tensor,
    pop_target: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between predicted and observed popularity."""
    return functional.mse_loss(pop_pred, pop_target)


def _au_branch_contrib(
    users: torch.Tensor,
    items: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DirectAU alignment and uniformity for one branch.

    Args:
        users: User embeddings for the batch, shape ``(B, D)``.
        items: Item embeddings for the batch, shape ``(B, D)``.
        temperature: Temperature for the uniformity loss kernel.

    Returns:
        Tuple of ``(alignment_loss, uniformity_loss)`` scalars.

    """
    align = _directau_alignment_loss(users, items)
    uniform = 0.5 * (
        _directau_uniformity_loss(users, temperature=temperature)
        + _directau_uniformity_loss(items, temperature=temperature)
    )
    return align, uniform


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
        pos_scores = cast("dict[str, torch.Tensor]", model_output["pos_scores"])
        neg_scores = cast("dict[str, torch.Tensor]", model_output["neg_scores"])
        propagated = cast("dict[str, torch.Tensor]", model_output["propagated"])
        ipw_weights = cast("torch.Tensor", model_output["ipw_weights"])
        loss_user_ids = cast("torch.Tensor", model_output["loss_user_ids"])

        losses: dict[str, torch.Tensor] = {}
        reference_score = pos_scores["final_score"]
        zero = reference_score.new_zeros(())

        # Curriculum: check phase thresholds.
        auxiliary_losses_active = epoch >= cfg.auxiliary_losses_start_epoch
        popularity_supervision_active = epoch >= cfg.popularity_supervision_start_epoch
        use_dual_branch = cfg.use_dual_branch

        # Fused BPR is always active from epoch 0; only auxiliary losses phase in.
        # L_rec: fused BPR on the final score
        weights = ipw_weights if cfg.use_ipw else None
        losses["rec"] = _bpr_loss(
            pos_scores["final_score"],
            neg_scores["final_score"],
            weights,
        )

        # Resolve all auxiliary weights from a single spec table.
        # Each entry: (lambda_val, ramp_rate, active_in_phased_schedule)
        aux_specs_ = [
            (cfg.lambda_interest_bpr, cfg.auxiliary_ramp_rate, True),
            (cfg.lambda_conformity_bpr, cfg.auxiliary_ramp_rate, True),
            (cfg.lambda_independence, cfg.independence_ramp_rate, auxiliary_losses_active),
            (cfg.lambda_contrastive, cfg.auxiliary_ramp_rate, auxiliary_losses_active),
            (cfg.lambda_align, cfg.auxiliary_ramp_rate, auxiliary_losses_active),
            (cfg.lambda_uniform, cfg.auxiliary_ramp_rate, auxiliary_losses_active),
            (cfg.lambda_pop, cfg.auxiliary_ramp_rate, popularity_supervision_active),
        ]
        (
            interest_weight,
            conformity_weight,
            independence_weight,
            contrastive_weight,
            align_weight,
            uniform_weight,
            popularity_weight,
        ) = [
            self._resolve_auxiliary_weight(lmax, epoch, ramp_rate=rr, active_in_phased_schedule=act)
            for lmax, rr, act in aux_specs_
        ]

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

        # DCCL-style branch-local contrastive losses on aligned positive pairs.
        if use_dual_branch and contrastive_weight > 0:
            losses["interest_contrastive"] = _interest_contrastive_loss(
                propagated["user_interest"][loss_user_ids],
                propagated["item_interest"][pos_item_ids],
                item_popularity,
                pos_item_ids,
                temperature=cfg.contrastive_temperature,
                max_pairs=cfg.contrastive_max_pairs,
            )
            losses["conformity_contrastive"] = _conformity_contrastive_loss(
                propagated["user_conformity"][loss_user_ids],
                propagated["item_conformity"][pos_item_ids],
                item_popularity,
                pos_item_ids,
                temperature=cfg.contrastive_temperature,
                max_pairs=cfg.contrastive_max_pairs,
            )
            losses["contrastive"] = (
                losses["interest_contrastive"] + losses["conformity_contrastive"]
            )
        else:
            losses["interest_contrastive"] = zero
            losses["conformity_contrastive"] = zero
            losses["contrastive"] = zero

        # DirectAU-style branch-local geometry lives on positive user-item pairs.
        if use_dual_branch and (align_weight > 0 or uniform_weight > 0):
            interest_users = propagated["user_interest"][loss_user_ids]
            interest_items = propagated["item_interest"][pos_item_ids]
            i_align, i_uniform = _au_branch_contrib(
                interest_users,
                interest_items,
                cfg.uniformity_temperature,
            )
            branch_align_losses: list[torch.Tensor] = [i_align]
            branch_uniform_losses: list[torch.Tensor] = [i_uniform]

            if cfg.use_conformity_au:
                conformity_users = propagated["user_conformity"][loss_user_ids]
                conformity_items = propagated["item_conformity"][pos_item_ids]
                c_align, c_uniform = _au_branch_contrib(
                    conformity_users,
                    conformity_items,
                    cfg.uniformity_temperature,
                )
                branch_align_losses.append(c_align)
                branch_uniform_losses.append(c_uniform)

            losses["align"] = torch.stack(branch_align_losses).mean()
            losses["uniform"] = torch.stack(branch_uniform_losses).mean()
        else:
            losses["align"] = zero
            losses["uniform"] = zero

        # Popularity regression now supervises the scorer-owned popularity head.
        if use_dual_branch and cfg.use_popularity_head and popularity_weight > 0:
            pop_target = item_popularity[pos_item_ids].to(
                device=reference_score.device,
                dtype=reference_score.dtype,
            )
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
