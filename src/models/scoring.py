"""Scoring layer for the single refined learned scorer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from ..utils.config import EDGRecConfig
from .common import module_parameter_dtype


class ScoringModule(nn.Module):
    """Compute refined interest, conformity, context, and final scores."""

    def __init__(self, config: EDGRecConfig, *, context_feature_dim: int) -> None:
        """Initialize scorer heads and priors.

        Args:
            config: Model configuration.
            context_feature_dim: Width of the item-only context feature vector.
        """
        super().__init__()
        self.config = config
        self.component_names = ("interest", "conformity", "context")
        prior_weights = torch.tensor(
            [
                config.score_weight_interest,
                config.score_weight_conformity,
                config.score_weight_popularity,
            ],
            dtype=torch.float32,
        )
        learned_prior_weights = prior_weights.clamp_min(1e-6)
        learned_prior_weights = learned_prior_weights / learned_prior_weights.sum()
        self.register_buffer(
            "score_prior_weights",
            prior_weights,
            persistent=False,
        )
        self.register_buffer(
            "alpha_prior_logits",
            learned_prior_weights.log(),
            persistent=False,
        )
        self.interest_gate_mlp = nn.Sequential(
            nn.Linear(2 * config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, 1),
        )
        self.alpha_mlp = nn.Sequential(
            nn.Linear(2 * config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, 3),
        )
        self.context_head = (
            nn.Sequential(
                nn.Linear(context_feature_dim, config.embed_dim),
                nn.SiLU(),
                nn.Linear(config.embed_dim, 1),
            )
            if config.use_popularity_head
            else None
        )

    @staticmethod
    def _cast_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Cast a tensor to match a reference tensor.

        Args:
            value: Tensor to cast.
            ref: Reference tensor.

        Returns:
            Cast tensor.
        """
        return value.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _pairwise_or_matrix_score(
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        *,
        pairwise: bool,
    ) -> torch.Tensor:
        """Return pairwise or full-catalog dot-product scores.

        Args:
            user_emb: User embedding tensor.
            item_emb: Item embedding tensor.
            pairwise: Whether to compute aligned pairwise scores.

        Returns:
            Score tensor.
        """
        if pairwise:
            return (user_emb * item_emb).sum(dim=-1)
        return user_emb @ item_emb.t()

    @staticmethod
    def _calibrated_dot_score(
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        *,
        pairwise: bool,
    ) -> torch.Tensor:
        """Return norm-invariant component logits for final score fusion.

        Args:
            user_emb: User embedding tensor.
            item_emb: Item embedding tensor.
            pairwise: Whether to compute aligned pairwise scores.

        Returns:
            Cosine-style score tensor bounded to ``[-1, 1]``.
        """
        user_norm = functional.normalize(user_emb.float(), dim=-1)
        item_norm = functional.normalize(item_emb.float(), dim=-1)
        if pairwise:
            return (user_norm * item_norm).sum(dim=-1)
        return user_norm @ item_norm.t()

    def _get_user_interest(self, propagated: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the long-term user interest embeddings.

        Args:
            propagated: Propagated embedding dictionary.

        Returns:
            User interest embeddings.
        """
        if self.config.use_dual_branch:
            return propagated["user_interest"]
        return propagated["user"]

    def _get_item_interest(self, propagated: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the item embeddings used by the interest branch.

        Args:
            propagated: Propagated embedding dictionary.

        Returns:
            Item interest embeddings.
        """
        if self.config.use_dual_branch:
            return propagated["item_interest"]
        return propagated["item"]

    def _get_user_conformity(self, propagated: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the user embeddings used by the conformity branch.

        Args:
            propagated: Propagated embedding dictionary.

        Returns:
            User conformity embeddings.
        """
        if self.config.use_dual_branch:
            return propagated["user_conformity"]
        return self._get_user_interest(propagated)

    def _get_item_conformity(self, propagated: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the item embeddings used by the conformity branch.

        Args:
            propagated: Propagated embedding dictionary.

        Returns:
            Item conformity embeddings.
        """
        if self.config.use_dual_branch:
            return propagated["item_conformity"]
        return self._get_item_interest(propagated)

    def _select_items(
        self,
        item_embedding: torch.Tensor,
        item_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return requested item rows.

        Args:
            item_embedding: Full item embedding table.
            item_ids: Optional item ids.

        Returns:
            Selected item embeddings.
        """
        return item_embedding if item_ids is None else item_embedding[item_ids]

    def _build_short_term_interest(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        long_term_interest: torch.Tensor,
    ) -> torch.Tensor:
        """Return recent-train short-term interest embeddings.

        Args:
            propagated: Propagated embedding dictionary.
            user_ids: User ids to score.
            long_term_interest: Long-term interest embeddings for fallback.

        Returns:
            Short-term interest embeddings.
        """
        recent_items = propagated.get("recent_train_items")
        recent_mask = propagated.get("recent_train_mask")
        if recent_items is None or recent_mask is None:
            return long_term_interest

        user_recent_items = recent_items[user_ids]
        user_recent_mask = recent_mask[user_ids]
        if user_recent_items.numel() == 0:
            return long_term_interest

        recent_item_interest = propagated.get("recent_train_item_interest")
        if recent_item_interest is not None:
            recent_item_emb = recent_item_interest[user_ids]
        else:
            item_interest = self._get_item_interest(propagated)
            recent_item_emb = item_interest[user_recent_items]
        mask = user_recent_mask.unsqueeze(-1).to(dtype=recent_item_emb.dtype)
        counts = mask.sum(dim=1).clamp_min(1.0)
        short_term = (recent_item_emb * mask).sum(dim=1) / counts
        has_history = user_recent_mask.any(dim=1, keepdim=True)
        return torch.where(has_history, short_term, long_term_interest)

    def _build_interest_embedding(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the gated user interest embeddings.

        Args:
            propagated: Propagated embedding dictionary.
            user_ids: User ids to score.

        Returns:
            Refined user interest embeddings.
        """
        long_term_interest = self._get_user_interest(propagated)[user_ids]
        short_term_interest = self._build_short_term_interest(
            propagated,
            user_ids,
            long_term_interest,
        )
        gate_inputs = torch.cat([long_term_interest, short_term_interest], dim=-1)
        gate_inputs = gate_inputs.to(dtype=module_parameter_dtype(self.interest_gate_mlp))
        interest_gate = torch.sigmoid(self.interest_gate_mlp(gate_inputs)).to(
            dtype=long_term_interest.dtype,
        )
        return interest_gate * short_term_interest + (1.0 - interest_gate) * long_term_interest

    def _get_item_scalar_feature(
        self,
        propagated: dict[str, torch.Tensor],
        key: str,
        item_ids: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch an item-level scalar feature with zero fallback.

        Args:
            propagated: Propagated embedding dictionary.
            key: Metadata key.
            item_ids: Optional item ids.
            ref: Reference tensor for device and dtype.

        Returns:
            Scalar feature tensor.
        """
        value = propagated.get(key)
        if value is None:
            return torch.zeros(ref.size(0), device=ref.device, dtype=ref.dtype)
        if item_ids is not None:
            value = value[item_ids]
        return self._cast_like(value, ref)

    def _get_item_safe_features(
        self,
        propagated: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch item-safe features with zero fallback.

        Args:
            propagated: Propagated embedding dictionary.
            item_ids: Optional item ids.
            ref: Reference tensor for device and dtype.

        Returns:
            Feature matrix.
        """
        value = propagated.get("item_safe_features")
        if value is None:
            return torch.zeros((ref.size(0), 0), device=ref.device, dtype=ref.dtype)
        if item_ids is not None:
            value = value[item_ids]
        return self._cast_like(value, ref)

    def _get_item_propensity_context_feature(
        self,
        propagated: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Return exposure-proxy context input only for calibrated IPW runs."""
        if not (self.config.use_ipw and self.config.loss_weight_propensity_calibration > 0):
            return torch.zeros(ref.size(0), device=ref.device, dtype=ref.dtype)
        return self._get_item_scalar_feature(
            propagated,
            "item_propensity_targets",
            item_ids,
            ref,
        )

    def _context_scores(
        self,
        propagated: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return item-only context scores.

        Args:
            propagated: Propagated embedding dictionary.
            item_ids: Optional item ids.

        Returns:
            Context score vector for the requested items.
        """
        item_ref = self._select_items(self._get_item_interest(propagated), item_ids)
        if not self._has_context_metadata(propagated):
            return torch.zeros(item_ref.size(0), device=item_ref.device, dtype=item_ref.dtype)
        context_inputs = [
            self._get_item_scalar_feature(
                propagated,
                "item_popularity",
                item_ids,
                item_ref,
            ).unsqueeze(-1),
            self._get_item_scalar_feature(
                propagated,
                "item_recency",
                item_ids,
                item_ref,
            ).unsqueeze(-1),
            self._get_item_propensity_context_feature(
                propagated,
                item_ids,
                item_ref,
            ).unsqueeze(-1),
            self._get_item_scalar_feature(propagated, "item_age", item_ids, item_ref).unsqueeze(-1),
            self._get_item_safe_features(propagated, item_ids, item_ref),
        ]
        head_inputs = torch.cat(context_inputs, dim=-1).to(
            dtype=module_parameter_dtype(self.context_head),
        )
        context_scores = self.context_head(head_inputs).squeeze(-1).to(dtype=item_ref.dtype)
        metadata_present = head_inputs.abs().sum(dim=-1) > 0
        return torch.where(
            metadata_present,
            context_scores,
            torch.zeros_like(context_scores),
        )

    def _has_context_metadata(
        self,
        propagated: dict[str, torch.Tensor],
    ) -> bool:
        """Return whether the context head has configured item metadata to score."""
        metadata_keys = (
            "item_popularity",
            "item_recency",
            "item_age",
            "item_safe_features",
        )
        has_metadata = any(key in propagated for key in metadata_keys)
        has_calibrated_propensity = (
            self.config.use_ipw
            and self.config.loss_weight_propensity_calibration > 0
            and "item_propensity_targets" in propagated
        )
        return (
            self.config.use_popularity_head
            and self.context_head is not None
            and (has_metadata or has_calibrated_propensity)
        )

    def _active_score_components(
        self,
        propagated: dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return score components available by model/data contract, not value."""
        return torch.tensor(
            [
                True,
                self.config.use_dual_branch,
                self._has_context_metadata(propagated),
            ],
            device=device,
            dtype=torch.bool,
        )

    def _fixed_score_mix_weights(
        self,
        active_components: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        """Return preset-owned fixed score-mix weights for active components.

        Args:
            active_components: Boolean mask of active score components.
            device: Target device.
            dtype: Target dtype.
            batch_size: Number of user rows to expand.

        Returns:
            Fixed score-mix weights with inactive components zeroed out.
        """
        weights = self.score_prior_weights.to(device=device, dtype=dtype)
        active_weights = active_components.to(device=device, dtype=dtype)
        masked_weights = weights * active_weights
        total_weight = masked_weights.sum()
        fallback_weights = active_weights / active_weights.sum().clamp_min(1.0)
        normalized = torch.where(
            total_weight > 0,
            masked_weights / total_weight.clamp_min(1e-12),
            fallback_weights,
        )
        return normalized.unsqueeze(0).expand(batch_size, -1)

    def _score_mix_weights(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        interest_embedding: torch.Tensor,
        active_components: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-user score-mix weights.

        Args:
            propagated: Propagated embedding dictionary.
            user_ids: User ids to score.
            interest_embedding: Refined user interest embeddings.

        Returns:
            Per-user score-mix weights.
        """
        if not self.config.use_dual_branch:
            return torch.tensor(
                [[1.0, 0.0, 0.0]],
                device=interest_embedding.device,
                dtype=interest_embedding.dtype,
            ).expand(user_ids.size(0), -1)
        if not self.config.use_learned_score_mix:
            return self._fixed_score_mix_weights(
                active_components,
                device=interest_embedding.device,
                dtype=interest_embedding.dtype,
                batch_size=user_ids.size(0),
            )
        user_conformity = self._get_user_conformity(propagated)[user_ids]
        alpha_inputs = torch.cat([interest_embedding, user_conformity], dim=-1).to(
            dtype=module_parameter_dtype(self.alpha_mlp),
        )
        alpha_logits = self.alpha_mlp(alpha_inputs) + self.alpha_prior_logits.to(
            device=alpha_inputs.device,
            dtype=alpha_inputs.dtype,
        )
        alpha_logits = alpha_logits.masked_fill(
            ~active_components.to(device=alpha_inputs.device),
            float("-inf"),
        )
        weights = torch.softmax(alpha_logits, dim=-1)
        min_weight = float(self.config.score_mix_min_weight)
        if min_weight > 0:
            active = active_components.to(device=weights.device, dtype=weights.dtype)
            active_count = active.sum().clamp_min(1.0)
            floor = torch.full((), min_weight, device=weights.device, dtype=weights.dtype)
            floor = torch.minimum(floor, 0.95 / active_count)
            weights = weights * (1.0 - floor * active_count) + floor * active
        return weights.to(dtype=interest_embedding.dtype)

    def _build_score_dict(
        self,
        interest_score: torch.Tensor,
        conformity_score: torch.Tensor,
        context_score: torch.Tensor,
        fusion_interest_score: torch.Tensor,
        fusion_conformity_score: torch.Tensor,
        fusion_context_score: torch.Tensor,
        score_mix_weights: torch.Tensor,
        *,
        pairwise: bool,
    ) -> dict[str, torch.Tensor]:
        """Assemble the scorer outputs.

        Args:
            interest_score: Raw interest dot-product score tensor.
            conformity_score: Raw conformity dot-product score tensor.
            context_score: Raw context score tensor.
            fusion_interest_score: Calibrated interest score used in final fusion.
            fusion_conformity_score: Calibrated conformity score used in final fusion.
            fusion_context_score: Calibrated context score used in final fusion.
            score_mix_weights: Per-user score-mix weights.
            pairwise: Whether the scores are pairwise or full-catalog.

        Returns:
            Score dictionary.
        """
        if pairwise:
            final_score = (
                score_mix_weights[:, 0] * fusion_interest_score
                + score_mix_weights[:, 1] * fusion_conformity_score
                + score_mix_weights[:, 2] * fusion_context_score
            )
            context_for_dict = fusion_context_score
            raw_context_for_dict = context_score
        else:
            final_score = (
                score_mix_weights[:, 0:1] * fusion_interest_score
                + score_mix_weights[:, 1:2] * fusion_conformity_score
                + score_mix_weights[:, 2:3] * fusion_context_score.unsqueeze(0)
            )
            context_for_dict = fusion_context_score.unsqueeze(0).expand_as(
                fusion_interest_score,
            )
            raw_context_for_dict = context_score.unsqueeze(0).expand_as(interest_score)
        return {
            "interest_score": fusion_interest_score,
            "conformity_score": fusion_conformity_score,
            "context_score": context_for_dict,
            "branch_interest_score": interest_score,
            "branch_conformity_score": conformity_score,
            "raw_context_score": raw_context_for_dict,
            "score_mix_weights": score_mix_weights,
            "final_score": final_score,
        }

    def forward(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score user-item pairs.

        Args:
            propagated: Propagated embedding dictionary.
            user_ids: User ids.
            item_ids: Item ids.

        Returns:
            Pairwise scorer outputs.
        """
        user_interest = self._build_interest_embedding(propagated, user_ids)
        item_interest = self._select_items(self._get_item_interest(propagated), item_ids)
        interest_score = self._pairwise_or_matrix_score(
            user_interest,
            item_interest,
            pairwise=True,
        )
        fusion_interest_score = self._calibrated_dot_score(
            user_interest,
            item_interest,
            pairwise=True,
        )
        user_conformity = self._get_user_conformity(propagated)[user_ids]
        item_conformity = self._select_items(self._get_item_conformity(propagated), item_ids)
        conformity_score = self._pairwise_or_matrix_score(
            user_conformity,
            item_conformity,
            pairwise=True,
        )
        fusion_conformity_score = self._calibrated_dot_score(
            user_conformity,
            item_conformity,
            pairwise=True,
        )
        context_score = self._context_scores(propagated, item_ids)
        fusion_context_score = torch.tanh(context_score.float())
        active_components = self._active_score_components(
            propagated,
            device=user_interest.device,
        )
        score_mix_weights = self._score_mix_weights(
            propagated,
            user_ids,
            user_interest,
            active_components,
        )
        return self._build_score_dict(
            interest_score,
            conformity_score,
            context_score,
            fusion_interest_score,
            fusion_conformity_score,
            fusion_context_score,
            score_mix_weights,
            pairwise=True,
        )

    def score_all_items(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return full-catalog scorer outputs.

        Args:
            propagated: Propagated embedding dictionary.
            user_ids: User ids.

        Returns:
            Full-catalog scorer outputs.
        """
        user_interest = self._build_interest_embedding(propagated, user_ids)
        item_interest = self._get_item_interest(propagated)
        interest_score = self._pairwise_or_matrix_score(
            user_interest,
            item_interest,
            pairwise=False,
        )
        fusion_interest_score = self._calibrated_dot_score(
            user_interest,
            item_interest,
            pairwise=False,
        )
        user_conformity = self._get_user_conformity(propagated)[user_ids]
        item_conformity = self._get_item_conformity(propagated)
        conformity_score = self._pairwise_or_matrix_score(
            user_conformity,
            item_conformity,
            pairwise=False,
        )
        fusion_conformity_score = self._calibrated_dot_score(
            user_conformity,
            item_conformity,
            pairwise=False,
        )
        context_score = self._context_scores(propagated, None)
        fusion_context_score = torch.tanh(context_score.float())
        active_components = self._active_score_components(
            propagated,
            device=user_interest.device,
        )
        score_mix_weights = self._score_mix_weights(
            propagated,
            user_ids,
            user_interest,
            active_components,
        )
        return self._build_score_dict(
            interest_score,
            conformity_score,
            context_score,
            fusion_interest_score,
            fusion_conformity_score,
            fusion_context_score,
            score_mix_weights,
            pairwise=False,
        )
