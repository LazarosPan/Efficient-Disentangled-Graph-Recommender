"""Vectorized negative sampler with optional popularity-weighted hard negatives."""

from __future__ import annotations

import torch


class NegativeSampler:
    """Generates negative item IDs for a batch of (user, pos_item) pairs.

    Supports two strategies mixed via ``hard_negative_ratio``:
    1. Uniform random negatives (torch.randint — fully vectorized)
    2. Popularity-weighted hard negatives (torch.multinomial)
    """

    def __init__(
        self,
        n_items: int,
        popularity: torch.Tensor,
        n_negatives: int = 1,
        hard_negative_ratio: float = 0.0,
    ) -> None:
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.hard_negative_ratio = hard_negative_ratio

        # Pre-compute popularity sampling weights
        pop = popularity.float()
        self._pop_weights = (
            pop / pop.sum() if pop.sum() > 0 else torch.ones(n_items) / n_items
        )

    def sample(
        self,
        batch_size: int,
        positive_items: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Sample negative item IDs.

        Args:
            batch_size: Number of users in the batch.
            positive_items: (B,) tensor of positive items to avoid (best-effort).
            device: Target device.

        Returns:
            (B, n_negatives) tensor of negative item IDs.
        """
        total = batch_size * self.n_negatives
        n_hard = int(total * self.hard_negative_ratio)
        n_uniform = total - n_hard

        parts = []

        if n_uniform > 0:
            uniform_neg = torch.randint(0, self.n_items, (n_uniform,), device=device)
            parts.append(uniform_neg)

        if n_hard > 0:
            weights = self._pop_weights.to(device)
            hard_neg = torch.multinomial(weights, n_hard, replacement=True)
            parts.append(hard_neg)

        neg_items = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        # Best-effort collision avoidance: replace negatives that match positives
        if positive_items is not None:
            pos_expanded = positive_items.repeat_interleave(self.n_negatives)
            collision_mask = neg_items == pos_expanded
            if collision_mask.any():
                replacements = torch.randint(
                    0,
                    self.n_items,
                    (collision_mask.sum().item(),),
                    device=device,
                )
                neg_items[collision_mask] = replacements

        return neg_items.view(batch_size, self.n_negatives)
