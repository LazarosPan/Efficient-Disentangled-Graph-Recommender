"""LossSuite: fused BPR plus branch-local auxiliaries for EDGRec."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from ..utils.config import DiceMaskReduction, EDGRecConfig

_AUXILIARY_LOSS_KEYS = (
    "interest_bpr",
    "conformity_bpr",
    "independence",
    "contrastive",
    "align",
    "uniform",
    "pop",
    "prop_calib",
)
_AUXILIARY_LOSS_LOG_NAMES = {
    "interest_bpr": "interest_bpr",
    "conformity_bpr": "conformity_bpr",
    "independence": "independence",
    "contrastive": "contrastive",
    "align": "align",
    "uniform": "uniform",
    "pop": "popularity",
    "prop_calib": "propensity_calibration",
}
_SCORE_MIX_COMPONENTS = ("interest", "conformity", "context")
_AUXILIARY_LOSS_EMA_DECAY = 0.99


def _bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the ranking loss, optionally reweighted by IPW.

    The score difference is promoted to fp32 so the loss remains stable under
    AMP. When IPW weights are provided, the result uses a self-normalized
    weighted mean so the objective scale does not depend on the absolute
    weight magnitude.
    """
    pos_scores = pos_scores.float()
    neg_scores = neg_scores.float()
    loss = -functional.logsigmoid(pos_scores - neg_scores)
    if weights is None:
        return loss.mean()
    weights = weights.detach().float()
    weights = torch.nan_to_num(weights, nan=1.0, posinf=10.0, neginf=1.0)
    weights = weights.clamp(0.1, 10.0)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def _masked_bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    mask: torch.Tensor,
    reduction: DiceMaskReduction,
) -> torch.Tensor:
    """Compute DICE's mask-weighted BPR with the configured reduction."""
    pos_scores = pos_scores.float()
    neg_scores = neg_scores.float()
    mask = mask.to(device=pos_scores.device, dtype=pos_scores.dtype)
    loss = -(mask * functional.logsigmoid(pos_scores - neg_scores))
    if reduction == "active_mean":
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def _independence_loss(
    interest: torch.Tensor,
    conformity: torch.Tensor,
) -> torch.Tensor:
    """Penalize correlation between interest and conformity embeddings."""
    cos_sim = functional.cosine_similarity(interest, conformity, dim=-1)
    return (cos_sim**2).mean()


def _distance_correlation_loss(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return positive distance correlation for DICE-style discrepancy."""
    if x.size(0) <= 1:
        return x.new_zeros(())
    x = x.float()
    y = y.float()
    a = torch.cdist(x, x, p=2)
    b = torch.cdist(y, y, p=2)
    a_centered = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    b_centered = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
    n = float(x.size(0) * x.size(0))
    dcov_xy = (a_centered * b_centered).sum() / n
    dcov_xx = (a_centered * a_centered).sum() / n
    dcov_yy = (b_centered * b_centered).sum() / n
    denom = torch.sqrt(dcov_xx.clamp_min(1e-12).sqrt() * dcov_yy.clamp_min(1e-12).sqrt())
    return torch.sqrt(dcov_xy.clamp_min(1e-12)) / denom.clamp_min(1e-12)


def _sample_quadratic_auxiliary_embeddings(
    x: torch.Tensor,
    y: torch.Tensor,
    entity_ids: torch.Tensor,
    max_pairs: int,
    salt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a bounded hash sample for quadratic auxiliary losses.

    Args:
        x: First embedding tensor with shape ``(N, D)``.
        y: Second embedding tensor aligned with ``x``.
        entity_ids: Stable IDs aligned with ``x`` and ``y``.
        max_pairs: Maximum number of rows to keep.
        salt: Epoch-dependent salt for deterministic variation across training.

    Returns:
        Pair of tensors sampled to at most ``max_pairs`` rows.

    """
    if x.size(0) <= max_pairs:
        return x, y
    modulus = 2_147_483_647
    salted_ids = (
        entity_ids.to(dtype=torch.long).remainder(modulus) + ((max(int(salt), 0) + 1) * 1_000_003)
    ).remainder(modulus)
    hash_keys = (salted_ids * 1_103_515_245 + 12_345).remainder(modulus)
    indices = torch.topk(hash_keys, k=max_pairs, largest=False, sorted=False).indices
    indices = indices.sort().values
    return x[indices], y[indices]


def _sample_quadratic_rows(
    embeddings: torch.Tensor,
    max_pairs: int,
    salt: int,
) -> torch.Tensor:
    """Return a bounded deterministic row sample for quadratic single-tensor losses."""
    if embeddings.size(0) <= max_pairs:
        return embeddings
    row_ids = torch.arange(embeddings.size(0), device=embeddings.device, dtype=torch.long)
    modulus = 2_147_483_647
    salted_ids = (row_ids.remainder(modulus) + ((max(int(salt), 0) + 1) * 1_000_003)).remainder(
        modulus
    )
    hash_keys = (salted_ids * 1_103_515_245 + 12_345).remainder(modulus)
    indices = torch.topk(hash_keys, k=max_pairs, largest=False, sorted=False).indices
    indices = indices.sort().values
    return embeddings[indices]


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

    user_embeddings = functional.normalize(user_embeddings.float(), dim=-1)
    item_embeddings = functional.normalize(item_embeddings.float(), dim=-1)
    temperature = max(float(temperature), 0.05)
    logits = (user_embeddings @ item_embeddings.t()) / temperature
    positive_logits = logits.diag()
    negative_logits = logits.masked_fill(~negative_mask, float("-inf"))
    negative_logsumexp = torch.logsumexp(negative_logits[valid_rows], dim=1)
    log_denom = torch.logaddexp(positive_logits[valid_rows], negative_logsumexp)
    loss = -(positive_logits[valid_rows] - log_denom)
    row_weights = positive_weights[valid_rows].detach().float()
    row_weights = torch.nan_to_num(row_weights, nan=1.0, posinf=1.0, neginf=0.0)
    row_weights = row_weights.clamp_min(1e-4)
    return (loss * row_weights).sum() / row_weights.sum().clamp_min(1e-8)


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
    max_pairs: int,
    salt: int,
) -> torch.Tensor:
    """DirectAU-style uniformity loss on normalized embeddings."""
    if embeddings.size(0) <= 1:
        return embeddings.new_zeros(())
    embeddings = _sample_quadratic_rows(embeddings, max_pairs, salt)
    normalized = functional.normalize(embeddings, dim=-1)
    pairwise_dist = torch.pdist(normalized, p=2)
    if pairwise_dist.numel() == 0:
        return embeddings.new_zeros(())
    return torch.log(torch.exp(-temperature * pairwise_dist.pow(2)).mean() + 1e-8)


def _popularity_loss(
    pop_pred: torch.Tensor,
    pop_target: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between predicted and observed popularity in fp32."""
    return functional.mse_loss(pop_pred.float(), pop_target.float())


def _au_branch_contrib(
    users: torch.Tensor,
    items: torch.Tensor,
    temperature: float,
    max_pairs: int,
    epoch: int,
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
        _directau_uniformity_loss(
            users,
            temperature=temperature,
            max_pairs=max_pairs,
            salt=epoch,
        )
        + _directau_uniformity_loss(
            items,
            temperature=temperature,
            max_pairs=max_pairs,
            salt=epoch + 10_000,
        )
    )
    return align, uniform


class LossSuite(nn.Module):
    """Combine fused ranking loss with branch-local auxiliary objectives."""

    def __init__(self, config: EDGRecConfig) -> None:
        super().__init__()
        self.config = config
        self._auxiliary_loss_index = {
            loss_name: index for index, loss_name in enumerate(_AUXILIARY_LOSS_KEYS)
        }
        self.register_buffer(
            "_aux_loss_ema",
            torch.ones(len(_AUXILIARY_LOSS_KEYS), dtype=torch.float32),
        )
        self.register_buffer(
            "_aux_loss_ema_initialized",
            torch.zeros(len(_AUXILIARY_LOSS_KEYS), dtype=torch.bool),
        )

    def _normalize_auxiliary_loss(
        self,
        loss_name: str,
        raw_loss: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Return an optionally EMA-normalized auxiliary loss."""
        if self.config.loss_normalization != "ema_aux" or weight <= 0:
            return raw_loss

        safe_loss = torch.nan_to_num(raw_loss, nan=0.0, posinf=0.0, neginf=0.0)
        detached_loss = torch.nan_to_num(
            safe_loss.detach().float().abs(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        loss_index = self._auxiliary_loss_index[loss_name]
        if self.training:
            current = self._aux_loss_ema[loss_index]
            initialized = self._aux_loss_ema_initialized[loss_index]
            updated = torch.where(
                initialized,
                current * _AUXILIARY_LOSS_EMA_DECAY
                + detached_loss * (1.0 - _AUXILIARY_LOSS_EMA_DECAY),
                detached_loss,
            )
            should_update = detached_loss > 0
            self._aux_loss_ema[loss_index].copy_(torch.where(should_update, updated, current))
            self._aux_loss_ema_initialized[loss_index].copy_(initialized | should_update)

        denominator = (
            self._aux_loss_ema[loss_index]
            .detach()
            .to(
                device=safe_loss.device,
                dtype=safe_loss.dtype,
            )
        )
        return safe_loss / denominator.clamp_min(1e-6)

    def _attach_loss_logging_aliases(
        self,
        losses: dict[str, torch.Tensor],
        normalized_losses: dict[str, torch.Tensor],
        aux_weights: Mapping[str, float],
    ) -> None:
        """Add stable raw/normalized/weighted scalar aliases for epoch logging."""
        losses["raw_rec_loss"] = losses["rec"]
        if "embedding_reg" in losses:
            losses["weighted_embedding_reg"] = self.config.weight_decay * losses["embedding_reg"]
        for loss_name in _AUXILIARY_LOSS_KEYS:
            raw_loss = losses[loss_name]
            log_name = _AUXILIARY_LOSS_LOG_NAMES[loss_name]
            normalized_loss = normalized_losses[loss_name]
            weight = aux_weights[loss_name]
            losses[f"raw_{log_name}"] = raw_loss
            if self.config.loss_normalization == "ema_aux":
                losses[f"normalized_{log_name}"] = normalized_loss
            losses[f"weighted_{log_name}"] = weight * normalized_loss

    def _resolve_auxiliary_weight(
        self,
        max_weight: float,
        epoch: int,
        *,
        ramp_rate: float,
        active_in_phased_schedule: bool,
    ) -> float:
        """Resolve one auxiliary-loss weight under the configured schedule.

        Args:
            max_weight: Configured maximum loss weight.
            epoch: Current zero-based epoch.
            ramp_rate: Linear ramp slope used when the schedule is
                ``linear_ramp``.
            active_in_phased_schedule: Whether the loss is active under the
                phased schedule.

        Returns:
            Effective scalar weight for the current epoch.

        """
        if max_weight <= 0:
            return 0.0
        if self.config.auxiliary_loss_schedule == "linear_ramp":
            return min(max_weight, ramp_rate * max(epoch, 0))
        return max_weight if active_in_phased_schedule else 0.0

    def forward(
        self,
        model_output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        item_popularity: torch.Tensor,
        pos_item_ids: torch.Tensor,
        epoch: int = 0,
        propensity_targets: torch.Tensor | None = None,
        branch_item_popularity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all active losses.

        Args:
            model_output: Output from EDGRec.forward().
            item_popularity: (I,) normalized popularity array.
            pos_item_ids: (B,) positive item indices (for popularity lookup).
            epoch: Current epoch (for curriculum scheduling).
            propensity_targets: Optional (I_sub,) per-item exposure proxy
                (subgraph-local).  When provided together with a positive
                ``loss_weight_propensity_calibration``, an MSE calibration loss
                is added to supervise the propensity estimator.

        Returns:
            Dict with individual losses and 'total' combined loss.

        """
        cfg = self.config
        pos_scores = cast("dict[str, torch.Tensor]", model_output["pos_scores"])
        neg_scores = cast("dict[str, torch.Tensor]", model_output["neg_scores"])
        propagated = cast("dict[str, torch.Tensor]", model_output["propagated"])
        ipw_weights = cast("torch.Tensor", model_output["ipw_weights"])
        loss_user_ids = cast("torch.Tensor", model_output["loss_user_ids"])
        neg_item_ids = cast("torch.Tensor | None", model_output.get("loss_neg_item_ids"))
        dice_negative_mask = cast(
            "torch.Tensor | None",
            model_output.get("dice_negative_mask"),
        )

        losses: dict[str, torch.Tensor] = {}
        reference_score = pos_scores["final_score"]
        zero = reference_score.new_zeros(())
        pos_interest_branch = pos_scores.get("branch_interest_score", pos_scores["interest_score"])
        neg_interest_branch = neg_scores.get("branch_interest_score", neg_scores["interest_score"])
        pos_conformity_branch = pos_scores.get(
            "branch_conformity_score",
            pos_scores["conformity_score"],
        )
        neg_conformity_branch = neg_scores.get(
            "branch_conformity_score",
            neg_scores["conformity_score"],
        )

        # Curriculum: check phase thresholds.
        auxiliary_losses_active = epoch >= cfg.auxiliary_losses_start_epoch
        popularity_supervision_active = epoch >= cfg.popularity_supervision_start_epoch
        use_dual_branch = cfg.use_dual_branch

        # Fused BPR is always active from epoch 0; only auxiliary losses phase in.
        # L_rec: fused BPR on the final score
        use_calibrated_ipw = (
            cfg.use_ipw
            and cfg.loss_weight_propensity_calibration > 0
            and propensity_targets is not None
            and model_output.get("propensity_scores") is not None
        )
        weights = ipw_weights if use_calibrated_ipw else None
        if cfg.recommendation_loss_mode == "dice_sum" and use_dual_branch:
            rec_pos = pos_interest_branch + pos_conformity_branch
            rec_neg = neg_interest_branch + neg_conformity_branch
        else:
            rec_pos = pos_scores["final_score"]
            rec_neg = neg_scores["final_score"]
        losses["rec"] = _bpr_loss(rec_pos, rec_neg, weights)

        if (
            cfg.baseline_family == "lightgcn_paper"
            and cfg.weight_decay > 0
            and neg_item_ids is not None
        ):
            embeddings = cast("dict[str, torch.Tensor]", model_output["embeddings"])
            user_ego = embeddings["user"][loss_user_ids].float()
            pos_ego = embeddings["item"][pos_item_ids].float()
            neg_ego = embeddings["item"][neg_item_ids].float()
            losses["embedding_reg"] = (
                0.5
                * (user_ego.norm(2).pow(2) + pos_ego.norm(2).pow(2) + neg_ego.norm(2).pow(2))
                / max(1, int(loss_user_ids.numel()))
            )
        else:
            losses["embedding_reg"] = zero

        interest_weight = max(float(cfg.loss_weight_interest_bpr), 0.0)
        conformity_weight = max(float(cfg.loss_weight_conformity_bpr), 0.0)

        # Resolve schedule-controlled auxiliary weights from a single spec table.
        # Branch BPR is intentionally excluded: it is primary causal branch
        # supervision, so it must be active from epoch 0.
        # Each entry: (max_weight, ramp_rate, active_in_phased_schedule)
        aux_specs_ = [
            (
                cfg.loss_weight_independence,
                cfg.independence_ramp_rate,
                auxiliary_losses_active,
            ),
            (cfg.loss_weight_contrastive, cfg.auxiliary_ramp_rate, auxiliary_losses_active),
            (cfg.loss_weight_align, cfg.auxiliary_ramp_rate, auxiliary_losses_active),
            (cfg.loss_weight_uniform, cfg.auxiliary_ramp_rate, auxiliary_losses_active),
            (
                cfg.loss_weight_popularity,
                cfg.auxiliary_ramp_rate,
                popularity_supervision_active,
            ),
            (
                cfg.loss_weight_propensity_calibration,
                cfg.auxiliary_ramp_rate,
                popularity_supervision_active,
            ),
        ]
        (
            independence_weight,
            contrastive_weight,
            align_weight,
            uniform_weight,
            popularity_weight,
            prop_calib_weight,
        ) = [
            self._resolve_auxiliary_weight(
                max_weight,
                epoch,
                ramp_rate=ramp_rate,
                active_in_phased_schedule=is_active,
            )
            for max_weight, ramp_rate, is_active in aux_specs_
        ]

        if cfg.dice_adaptive_decay:
            decay = cfg.dice_loss_decay ** max(epoch, 0)
            interest_weight *= decay
            conformity_weight *= decay

        # Branch-local BPR auxiliaries keep each branch predictive on its own.
        if (
            use_dual_branch
            and cfg.branch_loss_mode == "dice"
            and neg_item_ids is not None
            and (interest_weight > 0 or conformity_weight > 0)
        ):
            if dice_negative_mask is not None:
                popular_negative_mask = dice_negative_mask.to(
                    device=reference_score.device,
                    dtype=torch.bool,
                )
            else:
                branch_popularity = branch_item_popularity
                if branch_popularity is None:
                    branch_popularity = item_popularity
                pos_popularity = branch_popularity[pos_item_ids].to(device=reference_score.device)
                neg_popularity = branch_popularity[neg_item_ids].to(device=reference_score.device)
                branch_margin = cfg.dice_branch_margin * (
                    cfg.dice_margin_decay ** max(epoch, 0) if cfg.dice_adaptive_decay else 1.0
                )
                popular_negative_mask = neg_popularity > (pos_popularity + branch_margin)
            high_mask_rate = popular_negative_mask.float().mean()
            losses["dice_high_mask_rate"] = high_mask_rate.detach()
            losses["dice_low_mask_rate"] = (1.0 - high_mask_rate).detach()
            losses["interest_bpr"] = _masked_bpr_loss(
                pos_interest_branch,
                neg_interest_branch,
                popular_negative_mask,
                cfg.dice_mask_reduction,
            )
            losses["conformity_bpr"] = _masked_bpr_loss(
                neg_conformity_branch,
                pos_conformity_branch,
                popular_negative_mask,
                cfg.dice_mask_reduction,
            ) + _masked_bpr_loss(
                pos_conformity_branch,
                neg_conformity_branch,
                ~popular_negative_mask,
                cfg.dice_mask_reduction,
            )
        else:
            if use_dual_branch and interest_weight > 0:
                losses["interest_bpr"] = _bpr_loss(
                    pos_interest_branch,
                    neg_interest_branch,
                )
            else:
                losses["interest_bpr"] = zero

            if use_dual_branch and conformity_weight > 0:
                losses["conformity_bpr"] = _bpr_loss(
                    pos_conformity_branch,
                    neg_conformity_branch,
                )
            else:
                losses["conformity_bpr"] = zero

        # Branch independence via cosine-squared decorrelation.
        if use_dual_branch and independence_weight > 0:
            if cfg.branch_loss_mode == "dice" and neg_item_ids is not None:
                branch_user_ids = torch.unique(loss_user_ids)
                user_interest = propagated["user_interest"][branch_user_ids]
                user_conformity = propagated["user_conformity"][branch_user_ids]
                user_interest, user_conformity = _sample_quadratic_auxiliary_embeddings(
                    user_interest,
                    user_conformity,
                    branch_user_ids,
                    cfg.distance_correlation_max_pairs,
                    epoch,
                )
                branch_item_ids = torch.unique(torch.cat([pos_item_ids, neg_item_ids]))
                item_interest = propagated["item_interest"][branch_item_ids]
                item_conformity = propagated["item_conformity"][branch_item_ids]
                item_interest, item_conformity = _sample_quadratic_auxiliary_embeddings(
                    item_interest,
                    item_conformity,
                    branch_item_ids,
                    cfg.distance_correlation_max_pairs,
                    epoch,
                )
                losses["independence"] = _distance_correlation_loss(
                    user_interest,
                    user_conformity,
                ) + _distance_correlation_loss(item_interest, item_conformity)
            else:
                user_interest = propagated["user_interest"][loss_user_ids]
                user_conformity = propagated["user_conformity"][loss_user_ids]
                losses["independence"] = _independence_loss(
                    user_interest,
                    user_conformity,
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
                cfg.uniformity_max_pairs,
                epoch,
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
                    cfg.uniformity_max_pairs,
                    epoch + 20_000,
                )
                branch_align_losses.append(c_align)
                branch_uniform_losses.append(c_uniform)

            losses["align"] = torch.stack(branch_align_losses).mean()
            losses["uniform"] = torch.stack(branch_uniform_losses).mean()
        else:
            losses["align"] = zero
            losses["uniform"] = zero

        # Context regression now supervises the scorer-owned item-only context head.
        if use_dual_branch and cfg.use_popularity_head and popularity_weight > 0:
            pop_target = item_popularity[pos_item_ids].to(
                device=reference_score.device,
                dtype=reference_score.dtype,
            )
            context_for_supervision = pos_scores.get(
                "raw_context_score",
                pos_scores["context_score"],
            )
            losses["pop"] = _popularity_loss(
                context_for_supervision,
                pop_target,
            )
        else:
            losses["pop"] = zero

        # Propensity calibration: supervise the estimator with exposure-proxy targets.
        propensity_scores = cast("torch.Tensor | None", model_output.get("propensity_scores"))
        if (
            prop_calib_weight > 0
            and propensity_scores is not None
            and propensity_targets is not None
        ):
            calib_target = propensity_targets[pos_item_ids].to(
                device=reference_score.device,
                dtype=reference_score.dtype,
            )
            losses["prop_calib"] = _popularity_loss(propensity_scores, calib_target)
        else:
            losses["prop_calib"] = zero

        score_mix_weights = pos_scores.get("score_mix_weights")
        if score_mix_weights is not None:
            score_mix_means = score_mix_weights.detach().float().mean(dim=0)
            for component_index, component_name in enumerate(_SCORE_MIX_COMPONENTS):
                losses[f"score_mix_{component_name}_mean"] = score_mix_means[component_index]

        aux_weights = {
            "interest_bpr": interest_weight,
            "conformity_bpr": conformity_weight,
            "independence": independence_weight,
            "contrastive": contrastive_weight,
            "align": align_weight,
            "uniform": uniform_weight,
            "pop": popularity_weight,
            "prop_calib": prop_calib_weight,
        }
        normalized_losses = {
            loss_name: self._normalize_auxiliary_loss(
                loss_name,
                losses[loss_name],
                aux_weights[loss_name],
            )
            for loss_name in _AUXILIARY_LOSS_KEYS
        }

        total = (
            cfg.loss_weight_recommendation * losses["rec"]
            + interest_weight * normalized_losses["interest_bpr"]
            + conformity_weight * normalized_losses["conformity_bpr"]
            + independence_weight * normalized_losses["independence"]
            + contrastive_weight * normalized_losses["contrastive"]
            + align_weight * normalized_losses["align"]
            + uniform_weight * normalized_losses["uniform"]
            + popularity_weight * normalized_losses["pop"]
            + prop_calib_weight * normalized_losses["prop_calib"]
            + cfg.weight_decay * losses["embedding_reg"]
        )
        losses["total"] = total
        self._attach_loss_logging_aliases(losses, normalized_losses, aux_weights)

        return losses
