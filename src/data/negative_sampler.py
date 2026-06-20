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
        positive_user_ids: torch.Tensor | None = None,
        positive_item_ids: torch.Tensor | None = None,
        max_resample_attempts: int = 4,
        strategy: str = "standard",
        dice_margin: float = 40.0,
        dice_pool: int = 40,
        dice_margin_decay: float = 1.0,
        exact_dice_pool_counts: bool = False,
    ) -> None:
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.max_resample_attempts = max_resample_attempts
        self.strategy = strategy
        self.dice_margin = dice_margin
        self.dice_pool = dice_pool
        self.dice_margin_decay = dice_margin_decay
        self.exact_dice_pool_counts = exact_dice_pool_counts

        # Pre-compute popularity sampling weights
        pop = popularity.float()
        self._pop_weights = pop / pop.sum() if pop.sum() > 0 else torch.ones(n_items) / n_items
        self._sampling_popularity = pop
        sorted_popularity, sorted_items = torch.sort(pop)
        self._sorted_popularity = sorted_popularity
        self._sorted_items = sorted_items.long()
        self._pop_weights_by_device: dict[torch.device, torch.Tensor] = {}
        self._sampling_popularity_by_device: dict[torch.device, torch.Tensor] = {}
        self._sorted_popularity_by_device: dict[torch.device, torch.Tensor] = {}
        self._sorted_items_by_device: dict[torch.device, torch.Tensor] = {}
        self._positive_keys_cpu = self._build_positive_keys(
            positive_user_ids,
            positive_item_ids,
        )
        self._positive_user_index_cpu = (
            self._build_positive_user_index() if exact_dice_pool_counts else None
        )
        self._positive_keys_by_device: dict[torch.device, torch.Tensor] = {}
        self._positive_user_index_by_device: dict[
            torch.device,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def _build_positive_keys(
        self,
        positive_user_ids: torch.Tensor | None,
        positive_item_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build sorted ``user * n_items + item`` keys for known positives.

        Args:
            positive_user_ids: User IDs for positive train interactions.
            positive_item_ids: Item IDs for positive train interactions.

        Returns:
            Sorted unique key tensor on CPU, or ``None`` when unavailable.

        """
        if positive_user_ids is None or positive_item_ids is None:
            return None
        users = positive_user_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        items = positive_item_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if users.numel() == 0 or items.numel() == 0:
            return None
        if users.numel() != items.numel():
            raise ValueError("positive_user_ids and positive_item_ids must have equal length")
        return (users * self.n_items + items).unique().sort().values

    def _build_positive_user_index(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Build CSR-style train-positive item ranges keyed by user."""
        if self._positive_keys_cpu is None or self._positive_keys_cpu.numel() == 0:
            return None
        users = torch.div(self._positive_keys_cpu, self.n_items, rounding_mode="floor")
        items = torch.remainder(self._positive_keys_cpu, self.n_items).long()
        unique_users, counts = torch.unique_consecutive(users, return_counts=True)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)])
        return unique_users.long(), offsets.long(), items

    def _popularity_weights_for(self, device: torch.device) -> torch.Tensor:
        """Return cached popularity weights on ``device``.

        Args:
            device: Target device.

        Returns:
            Popularity sampling weights on ``device``.

        """
        if device not in self._pop_weights_by_device:
            self._pop_weights_by_device[device] = self._pop_weights.to(device)
        return self._pop_weights_by_device[device]

    def _dice_popularity_for(self, device: torch.device) -> torch.Tensor:
        """Return raw popularity values used by DICE sampling on ``device``."""
        if device not in self._sampling_popularity_by_device:
            self._sampling_popularity_by_device[device] = self._sampling_popularity.to(device)
        return self._sampling_popularity_by_device[device]

    def _sorted_dice_popularity_for(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sorted popularity values and their item ids on ``device``."""
        if device not in self._sorted_popularity_by_device:
            self._sorted_popularity_by_device[device] = self._sorted_popularity.to(device)
            self._sorted_items_by_device[device] = self._sorted_items.to(device)
        return self._sorted_popularity_by_device[device], self._sorted_items_by_device[device]

    def _positive_keys_for(self, device: torch.device) -> torch.Tensor | None:
        """Return cached positive-interaction keys on ``device``.

        Args:
            device: Target device.

        Returns:
            Sorted key tensor on ``device``, or ``None`` when unavailable.

        """
        if self._positive_keys_cpu is None:
            return None
        if device not in self._positive_keys_by_device:
            self._positive_keys_by_device[device] = self._positive_keys_cpu.to(device)
        return self._positive_keys_by_device[device]

    def _positive_user_index_for(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return cached CSR-style train-positive user/item ranges on ``device``."""
        if self._positive_user_index_cpu is None:
            return None
        if device not in self._positive_user_index_by_device:
            unique_users, offsets, items = self._positive_user_index_cpu
            self._positive_user_index_by_device[device] = (
                unique_users.to(device),
                offsets.to(device),
                items.to(device),
            )
        return self._positive_user_index_by_device[device]

    def _draw_raw_negatives(
        self,
        total: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Draw raw negative item IDs before collision filtering.

        Args:
            total: Number of item IDs to draw.
            device: Target device.
            generator: Optional deterministic RNG.

        Returns:
            Tensor with shape ``(total,)`` on ``device``.

        """
        if total <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        n_hard = int(total * self.hard_negative_ratio)
        n_uniform = total - n_hard

        parts = []
        if n_uniform > 0:
            parts.append(
                torch.randint(
                    0,
                    self.n_items,
                    (n_uniform,),
                    device=device,
                    generator=generator,
                ),
            )
        if n_hard > 0:
            parts.append(
                torch.multinomial(
                    self._popularity_weights_for(device),
                    n_hard,
                    replacement=True,
                    generator=generator,
                ),
            )
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def _draw_raw_negatives_with_mask(
        self,
        total: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw standard negatives with an all-false DICE mask."""
        items = self._draw_raw_negatives(total, device, generator)
        return items, torch.zeros(total, dtype=torch.bool, device=device)

    def _draw_dice_negatives_with_mask(
        self,
        positive_items: torch.Tensor,
        user_ids: torch.Tensor | None,
        total: int,
        device: torch.device,
        generator: torch.Generator | None,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw DICE-style negative item IDs and their high-popularity mask."""
        if total <= 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )
        pos_items = positive_items.to(device=device).repeat_interleave(self.n_negatives)
        popularity = self._dice_popularity_for(device)
        sorted_popularity, sorted_items = self._sorted_dice_popularity_for(device)
        pos_popularity = popularity[pos_items]
        margin = self.dice_margin * (self.dice_margin_decay ** max(epoch, 0))
        expanded_user_ids = (
            user_ids.to(device=device).repeat_interleave(self.n_negatives)
            if user_ids is not None
            else None
        )

        high_start = torch.searchsorted(
            sorted_popularity,
            pos_popularity + margin,
            right=True,
        )
        raw_high_count = self.n_items - high_start
        high_count = raw_high_count
        low_end = torch.searchsorted(
            sorted_popularity,
            pos_popularity / 2.0,
            right=False,
        )
        raw_low_count = low_end
        low_count = raw_low_count
        if expanded_user_ids is not None and self.exact_dice_pool_counts:
            high_positive_count, low_positive_count = self._known_positive_pool_counts(
                expanded_user_ids,
                pos_popularity,
                margin,
                device,
            )
            high_count = (high_count - high_positive_count).clamp_min(0)
            low_count = (low_count - low_positive_count).clamp_min(0)

        use_high = torch.rand(total, device=device, generator=generator) < 0.5
        use_high = torch.where(low_count < self.dice_pool, torch.ones_like(use_high), use_high)
        use_high = torch.where(high_count < self.dice_pool, torch.zeros_like(use_high), use_high)
        has_high = high_count > 0
        has_low = low_count > 0
        use_high = torch.where(has_high | ~has_low, use_high, torch.zeros_like(use_high))
        use_high = torch.where(has_low | ~has_high, use_high, torch.ones_like(use_high))

        high_offsets = (
            torch.rand(total, device=device, generator=generator) * raw_high_count.clamp_min(1)
        ).long()
        low_offsets = (
            torch.rand(total, device=device, generator=generator) * raw_low_count.clamp_min(1)
        ).long()
        sorted_positions = torch.where(use_high, high_start + high_offsets, low_offsets)
        dice_items = sorted_items[sorted_positions.clamp(max=self.n_items - 1)]

        fallback, fallback_mask = self._draw_raw_negatives_with_mask(total, device, generator)
        valid_pool = has_high | has_low
        return torch.where(valid_pool, dice_items, fallback), torch.where(
            valid_pool,
            use_high,
            fallback_mask,
        )

    def _dice_high_popularity_mask(
        self,
        positive_items_expanded: torch.Tensor,
        neg_items: torch.Tensor,
        device: torch.device,
        epoch: int,
    ) -> torch.Tensor:
        """Return the DICE high-popularity mask for final sampled negatives."""
        popularity = self._dice_popularity_for(device)
        margin = self.dice_margin * (self.dice_margin_decay ** max(epoch, 0))
        return popularity[neg_items] > (popularity[positive_items_expanded] + margin)

    def _known_positive_pool_counts(
        self,
        expanded_user_ids: torch.Tensor,
        pos_popularity: torch.Tensor,
        margin: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count known user positives inside each DICE high/low candidate pool."""
        positive_index = self._positive_user_index_for(device)
        high_counts = torch.zeros_like(expanded_user_ids, dtype=torch.long)
        low_counts = torch.zeros_like(expanded_user_ids, dtype=torch.long)
        if positive_index is None:
            return high_counts, low_counts

        unique_users, offsets, positive_items = positive_index
        popularity = self._dice_popularity_for(device)
        for user_id in torch.unique(expanded_user_ids):
            index = torch.searchsorted(unique_users, user_id)
            if index >= unique_users.numel() or unique_users[index] != user_id:
                continue
            start = int(offsets[index].item())
            end = int(offsets[index + 1].item())
            user_popularity = popularity[positive_items[start:end]]
            if user_popularity.numel() == 0:
                continue
            row_mask = expanded_user_ids == user_id
            row_pos_popularity = pos_popularity[row_mask]
            high_counts[row_mask] = (
                user_popularity.unsqueeze(0) > (row_pos_popularity + margin).unsqueeze(1)
            ).sum(dim=1)
            low_counts[row_mask] = (
                user_popularity.unsqueeze(0) < (row_pos_popularity / 2.0).unsqueeze(1)
            ).sum(dim=1)
        return high_counts, low_counts

    def _draw_negatives_for_strategy_with_mask(
        self,
        total: int,
        device: torch.device,
        generator: torch.Generator | None,
        positive_items: torch.Tensor | None,
        user_ids: torch.Tensor | None,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw negatives from the configured sampling strategy with metadata."""
        if self.strategy == "dice" and positive_items is not None:
            return self._draw_dice_negatives_with_mask(
                positive_items,
                user_ids,
                total,
                device,
                generator,
                epoch,
            )
        return self._draw_raw_negatives_with_mask(total, device, generator)

    def _known_positive_collision_mask(
        self,
        expanded_user_ids: torch.Tensor,
        negative_items: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return collisions between sampled negatives and train positives.

        Args:
            expanded_user_ids: User IDs repeated to align with ``neg_items``.
            negative_items: Flattened sampled negatives.
            device: Active sampling device.

        Returns:
            Boolean mask with shape matching ``negative_items``.

        """
        positive_keys = self._positive_keys_for(device)
        if positive_keys is None or positive_keys.numel() == 0:
            return torch.zeros_like(negative_items, dtype=torch.bool)

        sampled_keys = expanded_user_ids * self.n_items + negative_items
        positions = torch.searchsorted(positive_keys, sampled_keys)
        safe_positions = positions.clamp(max=positive_keys.numel() - 1)
        return (positions < positive_keys.numel()) & (positive_keys[safe_positions] == sampled_keys)

    def _collision_mask(
        self,
        negative_items: torch.Tensor,
        positive_items_expanded: torch.Tensor | None,
        expanded_user_ids: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Return sampled negatives that violate current or known positives.

        Args:
            negative_items: Flattened sampled negatives.
            positive_items_expanded: Flattened current positive item IDs, if available.
            expanded_user_ids: User IDs repeated to align with ``neg_items``.
            device: Active sampling device.

        Returns:
            Boolean mask with shape matching ``negative_items``.

        """
        collision_mask = torch.zeros_like(negative_items, dtype=torch.bool)
        if positive_items_expanded is not None:
            collision_mask |= negative_items == positive_items_expanded
        if expanded_user_ids is not None:
            collision_mask |= self._known_positive_collision_mask(
                expanded_user_ids,
                negative_items,
                device,
            )
        return collision_mask

    def _duplicate_negative_mask(
        self,
        negative_items: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return repeated negatives within each positive row.

        The original DICE sampler rejects duplicates across the
        ``neg_sample_rate`` negatives sampled for one positive interaction. This
        vectorized equivalent marks the later duplicate columns so they can be
        redrawn without changing already-valid samples.
        """
        if self.n_negatives <= 1:
            return torch.zeros_like(negative_items, dtype=torch.bool)
        negative_item_matrix = negative_items.view(batch_size, self.n_negatives)
        duplicate_mask = torch.zeros_like(negative_item_matrix, dtype=torch.bool)
        for column in range(1, self.n_negatives):
            duplicate_mask[:, column] = (
                negative_item_matrix[:, :column]
                == negative_item_matrix[:, column : column + 1]
            ).any(dim=1)
        return duplicate_mask.reshape(-1)

    def _negative_violation_mask(
        self,
        negative_items: torch.Tensor,
        batch_size: int,
        positive_items_expanded: torch.Tensor | None,
        expanded_user_ids: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Return negatives that collide with positives or repeat within a row.

        Args:
            negative_items: Flattened sampled negatives.
            batch_size: Number of positive rows represented by ``negative_items``.
            positive_items_expanded: Current positives repeated per negative slot.
            expanded_user_ids: User IDs repeated per negative slot.
            device: Active sampling device.

        Returns:
            Boolean mask with shape matching ``negative_items``.

        """
        return self._collision_mask(
            negative_items,
            positive_items_expanded,
            expanded_user_ids,
            device,
        ) | self._duplicate_negative_mask(negative_items, batch_size)

    def sample_with_metadata(
        self,
        batch_size: int,
        positive_items: torch.Tensor | None = None,
        user_ids: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample negative item IDs plus optional DICE sampler metadata.

        Args:
            batch_size: Number of users in the batch.
            positive_items: (B,) tensor of positive items to avoid (best-effort).
            user_ids: Optional (B,) tensor used to avoid every known positive
                training item for the same user.
            device: Target device.
            generator: Optional deterministic RNG for reproducible sampling.

        Returns:
            Tuple of ``(negative_items, dice_negative_mask)``. Negative items
            have shape ``(B, n_negatives)``. The mask has the same shape only
            for DICE sampling and marks high-popularity negatives, matching
            the ``mask_type`` consumed by the original DICE loss.

        """
        if device is None:
            device = (
                positive_items.device
                if positive_items is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        device = torch.device(device)

        total_negative_count = batch_size * self.n_negatives
        negative_items, dice_negative_mask = self._draw_negatives_for_strategy_with_mask(
            total_negative_count,
            device,
            generator,
            positive_items,
            user_ids,
            epoch,
        )

        positive_items_expanded = (
            positive_items.repeat_interleave(self.n_negatives).to(device=device)
            if positive_items is not None
            else None
        )
        expanded_user_ids = (
            user_ids.to(device=device).repeat_interleave(self.n_negatives)
            if user_ids is not None
            else None
        )

        if (
            positive_items_expanded is not None
            or expanded_user_ids is not None
            or self.n_negatives > 1
        ):
            for _ in range(self.max_resample_attempts):
                violation_mask = self._negative_violation_mask(
                    negative_items,
                    batch_size,
                    positive_items_expanded,
                    expanded_user_ids,
                    device,
                )
                replacements, replacement_mask = self._draw_negatives_for_strategy_with_mask(
                    total_negative_count,
                    device,
                    generator,
                    positive_items,
                    user_ids,
                    epoch,
                )
                negative_items = torch.where(
                    violation_mask,
                    replacements,
                    negative_items,
                )
                dice_negative_mask = torch.where(
                    violation_mask,
                    replacement_mask,
                    dice_negative_mask,
                )

            violation_mask = self._negative_violation_mask(
                negative_items,
                batch_size,
                positive_items_expanded,
                expanded_user_ids,
                device,
            )
            if violation_mask.any():
                for _ in range(min(self.n_items, 64)):
                    next_items = (negative_items + 1) % self.n_items
                    negative_items = torch.where(
                        violation_mask,
                        next_items,
                        negative_items,
                    )
                    violation_mask = self._negative_violation_mask(
                        negative_items,
                        batch_size,
                        positive_items_expanded,
                        expanded_user_ids,
                        device,
                    )

        if self.strategy == "dice" and positive_items is not None:
            assert positive_items_expanded is not None
            dice_negative_mask = self._dice_high_popularity_mask(
                positive_items_expanded,
                negative_items,
                device,
                epoch,
            )
            resolved_mask: torch.Tensor | None = dice_negative_mask.view(
                batch_size,
                self.n_negatives,
            )
        else:
            resolved_mask = None

        return negative_items.view(batch_size, self.n_negatives), resolved_mask

    def sample(
        self,
        batch_size: int,
        positive_items: torch.Tensor | None = None,
        user_ids: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Sample negative item IDs.

        Args:
            batch_size: Number of users in the batch.
            positive_items: (B,) tensor of positive items to avoid (best-effort).
            user_ids: Optional (B,) tensor used to avoid every known positive
                training item for the same user.
            device: Target device.
            generator: Optional deterministic RNG for reproducible sampling.

        Returns:
            (B, n_negatives) tensor of negative item IDs.

        """
        neg_items, _dice_mask = self.sample_with_metadata(
            batch_size,
            positive_items=positive_items,
            user_ids=user_ids,
            device=device,
            generator=generator,
            epoch=epoch,
        )
        return neg_items
