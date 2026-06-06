"""Shared interaction-mask helpers for label-aware training and evaluation."""

from __future__ import annotations

import numpy as np
import torch

MaskArray = np.ndarray | torch.Tensor


def positive_interaction_mask(mask: MaskArray, labels: MaskArray | None) -> MaskArray:
    """Return ``mask`` narrowed to interactions with positive labels.

    Args:
        mask: Boolean split mask over interaction rows.
        labels: Optional label vector aligned to ``mask``. When absent, the
            original mask is returned as a boolean mask.

    Returns:
        A boolean mask of the same array family as ``mask``.

    """
    if torch.is_tensor(mask):
        bool_mask = mask.bool()
        if labels is None:
            return bool_mask
        label_tensor = (
            labels.to(device=bool_mask.device)
            if torch.is_tensor(labels)
            else torch.as_tensor(labels, device=bool_mask.device)
        )
        return bool_mask & (label_tensor > 0)

    bool_mask = np.asarray(mask, dtype=np.bool_)
    if labels is None:
        return bool_mask
    return bool_mask & (np.asarray(labels) > 0)
