"""Module C: fused scoring with a popularity head and adaptive score mixing.

Scoring modes and the per-preset train/eval contract are documented in
``UCaGNNConfig`` (``src/utils/config.py``), above the preset methods.
"""

from __future__ import annotations

import torch
from torch import nn

from ..utils.config import UCaGNNConfig


class ScoringModule(nn.Module):
    """Compute fused recommendation scores from propagated embeddings.

    The mainline path combines interest, conformity, and popularity scores into a
    fused ranking score. Diagnostics expose the raw component scores and gate
    weights without introducing an extra derived causal quantity.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        super().__init__()
        self.config = config
        self.component_names = ("interest", "conformity", "popularity")

        # Pre-compute mode masks as non-persistent buffers (auto-move with .to())
        use_conformity = config.use_dual_branch
        use_popularity = config.use_popularity_head
        for name, mask in {
            "default": torch.tensor([True, use_conformity, use_popularity]),
            "interest_only": torch.tensor([True, False, False]),
            "conformity_only": torch.tensor([False, use_conformity, False]),
            "conformity_suppressed": torch.tensor([True, False, use_popularity]),
        }.items():
            self.register_buffer(f"_mask_{name}", mask, persistent=False)

        if config.scoring_weight_mode == "learned" and config.use_dual_branch:
            initial_weights = torch.tensor(
                [
                    config.alpha_interest,
                    config.beta_conformity,
                    config.gamma_popularity,
                ],
                dtype=torch.bfloat16,
            )
            initial_weights = initial_weights.clamp_min(1e-6)
            initial_weights = initial_weights / initial_weights.sum()
            self.score_weight_logits = nn.Parameter(initial_weights.log())
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * config.embed_dim, config.embed_dim),
                nn.SiLU(),
                nn.Linear(config.embed_dim, 3),
            )
        else:
            self.register_parameter("score_weight_logits", None)
            self.gate_mlp = None

        popularity_input_dim = config.embed_dim + 2
        if config.use_popularity_emb:
            popularity_input_dim += config.pop_embed_dim
        self.popularity_head = (
            nn.Sequential(
                nn.Linear(popularity_input_dim, config.embed_dim),
                nn.SiLU(),
                nn.Linear(config.embed_dim, 1),
            )
            if config.use_popularity_head
            else None
        )

        # Register fixed weights as a buffer (moves with .to(device) automatically)
        self.register_buffer(
            "_fixed_weights",
            torch.tensor(
                [
                    config.alpha_interest,
                    config.beta_conformity,
                    config.gamma_popularity,
                ],
                dtype=torch.bfloat16,
            ),
        )

    def _mode_mask(
        self,
        scoring_mode: str,
    ) -> torch.Tensor:
        """Return the pre-registered mode mask (already on the module's device)."""
        mask = getattr(self, f"_mask_{scoring_mode}", None)
        if mask is None:
            raise ValueError(f"Unknown scoring_mode: {scoring_mode}")
        return mask

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        """Return the dtype of the first parameter owned by ``module``."""
        return next(module.parameters()).dtype

    @staticmethod
    def _cast_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Return ``value`` cast to the device and dtype of ``ref``."""
        return value.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _select_item_embeddings(
        item_embedding: torch.Tensor,
        item_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return full-catalog or indexed item embeddings for scoring."""
        if item_ids is None:
            return item_embedding
        return item_embedding[item_ids]

    @property
    def _conformity_item_key(self) -> str:
        """Return the propagated embedding key for the popularity head anchor."""
        return "item_conformity" if self.config.use_dual_branch else "item"

    def _get_scalar_feature(
        self,
        propagated: dict[str, torch.Tensor],
        key: str,
        item_ids: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch an optional per-item scalar from *propagated*, slice, and cast.

        Args:
            propagated: Dict of propagated embeddings.
            key: Key to look up in *propagated*.
            item_ids: Optional item index subset; ``None`` means full catalog.
            ref: Reference tensor whose device and dtype the result is cast to.

        Returns:
            1-D tensor of shape ``(ref.size(0),)`` on *ref*'s device and dtype.
            Falls back to zeros when *key* is absent from *propagated*.

        """
        value = propagated.get(key)
        if value is None:
            return torch.zeros(ref.size(0), device=ref.device, dtype=ref.dtype)
        if item_ids is not None:
            value = value[item_ids]
        return self._cast_like(value, ref)

    def _score_components(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        item_ids: torch.Tensor | None,
        *,
        pairwise: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return interest and conformity scores for one scoring path."""
        score_fn = (lambda u, i: (u * i).sum(dim=-1)) if pairwise else (lambda u, i: u @ i.t())

        if self.config.use_dual_branch:
            interest_score = score_fn(
                propagated["user_interest"][user_ids],
                self._select_item_embeddings(propagated["item_interest"], item_ids),
            )
            conformity_score = score_fn(
                propagated["user_conformity"][user_ids],
                self._select_item_embeddings(propagated["item_conformity"], item_ids),
            )
            return interest_score, conformity_score

        interest_score = score_fn(
            propagated["user"][user_ids],
            self._select_item_embeddings(propagated["item"], item_ids),
        )
        return interest_score, torch.zeros_like(interest_score)

    def _popularity_item_inputs(
        self,
        propagated: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the popularity-head input features for the requested item rows."""
        item_anchor = self._select_item_embeddings(propagated[self._conformity_item_key], item_ids)
        popularity = self._get_scalar_feature(propagated, "item_popularity", item_ids, item_anchor)
        recency = self._get_scalar_feature(propagated, "item_recency", item_ids, item_anchor)

        features = [item_anchor, popularity.unsqueeze(-1), recency.unsqueeze(-1)]
        item_pop = propagated.get("item_pop")
        if item_pop is not None:
            if item_ids is not None:
                item_pop = item_pop[item_ids]
            features.append(self._cast_like(item_pop, item_anchor))
        elif self.config.use_popularity_emb:
            features.append(
                torch.zeros(
                    item_anchor.size(0),
                    self.config.pop_embed_dim,
                    device=item_anchor.device,
                    dtype=item_anchor.dtype,
                ),
            )
        return torch.cat(features, dim=-1)

    def _popularity_scores(
        self,
        propagated: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return item-level popularity scores for pairwise or full-catalog paths."""
        item_anchor = self._select_item_embeddings(propagated[self._conformity_item_key], item_ids)
        if not self.config.use_popularity_head or self.popularity_head is None:
            return torch.zeros(
                item_anchor.size(0),
                device=item_anchor.device,
                dtype=item_anchor.dtype,
            )
        pop_inputs = self._popularity_item_inputs(propagated, item_ids)
        pop_inputs = pop_inputs.to(dtype=self._module_dtype(self.popularity_head))
        return self.popularity_head(pop_inputs).squeeze(-1)

    def _resolve_gate_weights(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        *,
        scoring_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return per-user gate weights for the active score components."""
        batch_size = user_ids.size(0)
        fixed_or_prior = self.get_score_weight_tensor(
            scoring_mode=scoring_mode,
            device=device,
            dtype=dtype,
        )
        if not self.config.use_dual_branch:
            return fixed_or_prior.unsqueeze(0).expand(batch_size, -1)

        if self.config.scoring_weight_mode != "learned" or self.gate_mlp is None:
            return fixed_or_prior.unsqueeze(0).expand(batch_size, -1)

        if self.score_weight_logits is None:
            raise RuntimeError("Adaptive fusion requested without score_weight_logits")

        gate_inputs = torch.cat(
            [
                propagated["user_interest"][user_ids],
                propagated["user_conformity"][user_ids],
            ],
            dim=-1,
        )
        gate_inputs = gate_inputs.to(dtype=self._module_dtype(self.gate_mlp))
        logits = self.gate_mlp(gate_inputs) + self.score_weight_logits.to(
            device=gate_inputs.device,
            dtype=gate_inputs.dtype,
        )
        mask = self._mode_mask(scoring_mode).to(logits.device)
        active_logits = logits[:, mask]
        if active_logits.numel() == 0:
            return torch.zeros(batch_size, 3, device=device, dtype=dtype)
        active_weights = torch.softmax(active_logits, dim=-1).to(dtype=dtype)
        weights = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        weights[:, mask] = active_weights
        return weights

    def get_score_weight_tensor(
        self,
        scoring_mode: str = "default",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        resolved_device = device or torch.device("cpu")
        resolved_dtype = dtype or torch.bfloat16
        if not self.config.use_dual_branch:
            return torch.tensor(
                [1.0, 0.0, 0.0],
                device=resolved_device,
                dtype=resolved_dtype,
            )

        if self.config.scoring_weight_mode == "fixed":
            base_weights = self._fixed_weights.to(
                device=resolved_device,
                dtype=resolved_dtype,
            )
        else:
            if self.score_weight_logits is None:
                raise RuntimeError(
                    "Learned scoring weights requested without score_weight_logits",
                )
            base_weights = self.score_weight_logits
            if device is not None or dtype is not None:
                base_weights = base_weights.to(
                    device=device or base_weights.device,
                    dtype=dtype or base_weights.dtype,
                )

        mask = self._mode_mask(scoring_mode).to(base_weights.device)
        if self.config.scoring_weight_mode == "fixed":
            weights = torch.where(mask, base_weights, torch.zeros_like(base_weights))
            if int(mask.sum().item()) == 1:
                active_total = weights.sum()
                if active_total > 0:
                    weights = weights / active_total
            return weights

        active_logits = base_weights[mask]
        if active_logits.numel() == 0:
            return torch.zeros_like(base_weights)

        active_weights = torch.softmax(active_logits, dim=0).to(
            dtype=base_weights.dtype,
        )
        weights = torch.zeros_like(base_weights)
        weights[mask] = active_weights
        return weights

    def get_score_weight_summary(
        self,
        scoring_mode: str = "default",
    ) -> dict[str, float]:
        weights = self.get_score_weight_tensor(scoring_mode=scoring_mode).detach().cpu().tolist()
        return {
            f"score_weight_{name}": float(weight)
            for name, weight in zip(self.component_names, weights, strict=True)
        }

    def _build_score_dict(
        self,
        interest_score: torch.Tensor,
        conformity_score: torch.Tensor,
        popularity: torch.Tensor,
        gate_weights: torch.Tensor,
        *,
        scoring_mode: str,
        pairwise: bool,
    ) -> dict[str, torch.Tensor]:
        """Build the complete score dict for both pairwise and full-catalog paths.

        Args:
            interest_score: Interest component — (B,) pairwise or (B, I) matrix.
            conformity_score: Conformity component — same shape as interest_score.
            popularity: (B,) per-pair scores for pairwise; (I,) per-item for matrix.
            gate_weights: (B, 3) per-user component weights.
            scoring_mode: Active score view (used for logging/diagnostics only here).
            pairwise: ``True`` → pairwise (B,) path; ``False`` → matrix (B, I) path.

        Returns:
            Dict with interest, conformity, popularity, gate weights, and fused
            final scores.

        """
        if pairwise:
            final_score = (
                gate_weights[:, 0] * interest_score
                + gate_weights[:, 1] * conformity_score
                + gate_weights[:, 2] * popularity
            )
            pop_for_dict = popularity
        else:
            final_score = (
                gate_weights[:, 0:1] * interest_score
                + gate_weights[:, 1:2] * conformity_score
                + gate_weights[:, 2:3] * popularity.unsqueeze(0)
            )
            pop_for_dict = popularity.unsqueeze(0).expand_as(interest_score)
        return {
            "interest_score": interest_score,
            "conformity_score": conformity_score,
            "popularity_score": pop_for_dict,
            "gate_weights": gate_weights,
            "final_score": final_score,
        }

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
            scoring_mode: Active score view. Training uses
                ``config.train_scoring_mode`` through ``UCaGNN``, while
                evaluation can reuse the same checkpoint with a different score
                mode.

        Returns:
            Dict with interest, conformity, popularity, gate, and fused final
            scores.

        """
        interest_score, conformity_score = self._score_components(
            propagated,
            user_ids,
            item_ids,
            pairwise=True,
        )
        popularity_score = self._popularity_scores(propagated, item_ids)
        gate_weights = self._resolve_gate_weights(
            propagated,
            user_ids,
            scoring_mode=scoring_mode,
            device=interest_score.device,
            dtype=interest_score.dtype,
        )
        return self._build_score_dict(
            interest_score,
            conformity_score,
            popularity_score,
            gate_weights,
            scoring_mode=scoring_mode,
            pairwise=True,
        )

    def score_all_items(
        self,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
        scoring_mode: str = "default",
    ) -> dict[str, torch.Tensor]:
        """Return full-catalog score components for evaluation and diagnostics.

        ``scoring_mode`` lets callers reuse one propagated checkpoint state
        while selecting a different evaluation-time score contract.

        """
        interest_score, conformity_score = self._score_components(
            propagated,
            user_ids,
            None,
            pairwise=False,
        )
        popularity_items = self._popularity_scores(propagated, None)
        gate_weights = self._resolve_gate_weights(
            propagated,
            user_ids,
            scoring_mode=scoring_mode,
            device=interest_score.device,
            dtype=interest_score.dtype,
        )
        return self._build_score_dict(
            interest_score,
            conformity_score,
            popularity_items,
            gate_weights,
            scoring_mode=scoring_mode,
            pairwise=False,
        )
