"""Shared helpers for model modules."""

from __future__ import annotations

import torch
from torch import nn


def module_parameter_dtype(module: nn.Module) -> torch.dtype:
    """Return the dtype of the first parameter owned by ``module``."""
    return next(module.parameters()).dtype


def training_output_payload(
    *,
    embeddings: dict[str, torch.Tensor],
    propagated: dict[str, torch.Tensor],
    pos_scores: dict[str, torch.Tensor],
    neg_scores: dict[str, torch.Tensor],
    user_ids: torch.Tensor,
    neg_item_ids: torch.Tensor,
    ipw_weights: torch.Tensor | None = None,
    dice_negative_mask: torch.Tensor | None = None,
    propensity_scores: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Build the model training payload consumed by ``LossSuite``."""
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "embeddings": embeddings,
        "propagated": propagated,
        "ipw_weights": (
            ipw_weights
            if ipw_weights is not None
            else torch.ones(user_ids.size(0), device=user_ids.device)
        ),
        "loss_user_ids": user_ids,
        "loss_neg_item_ids": neg_item_ids,
    }
    if dice_negative_mask is not None:
        output["dice_negative_mask"] = dice_negative_mask
    if propensity_scores is not None:
        output["propensity_scores"] = propensity_scores
    return output


__all__ = ["module_parameter_dtype", "training_output_payload"]
