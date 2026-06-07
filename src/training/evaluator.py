"""GPU-vectorized link-prediction evaluation using PyG metrics."""

from __future__ import annotations

from typing import Final

import torch
from torch_geometric.metrics import (
    LinkPredAveragePopularity,
    LinkPredHitRatio,
    LinkPredMetricCollection,
    LinkPredNDCG,
    LinkPredPersonalization,
    LinkPredRecall,
)

from ..data.interaction_masks import positive_interaction_mask
from ..utils.config import UCaGNNConfig
from ..utils.trainer_runtime import (
    autocast_context,
    model_device,
    move_optional_tensor_to_device,
    move_tensor_to_device,
    stage_graph_tensors_for_device,
)

THESIS_PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "NDCG@20",
    "Recall@20",
    "AveragePopularity@20",
    "HitRatio@20",
    "Personalization@20",
    "NDCG@40",
    "Recall@40",
    "AveragePopularity@40",
    "HitRatio@40",
    "Personalization@40",
)

# Keep the thesis metric contract limited to PyG-standard metrics plus
# diagnostics that are easy to interpret and defend. Do not reintroduce custom
# Novelty, TrainPop Avoidance, or item-pop IoU surrogates unless a paper-faithful
# definition is implemented and explicitly justified.
LOWER_IS_BETTER_METRICS: Final[frozenset[str]] = frozenset(
    {"AveragePopularity@20", "AveragePopularity@40"},
)
THESIS_EVAL_KS: Final[tuple[int, ...]] = (20, 40)
_SCORE_MIX_COMPONENTS: Final[tuple[str, ...]] = ("interest", "conformity", "context")
_BRANCH_RANKING_SCORE_KEYS: Final[dict[str, tuple[str, str]]] = {
    "interest_branch": ("branch_interest_score", "interest_score"),
    "conformity_branch": ("branch_conformity_score", "conformity_score"),
}


class _SafeLinkPredPersonalization(LinkPredPersonalization):
    """Return a finite personalization score for degenerate tiny-user splits.

    PyG's base implementation computes the average dissimilarity across user
    pairs. When fewer than two users are evaluated, the pair count is zero and
    the upstream metric returns ``0 / 0 -> nan``. For tiny smoke-validation
    runs, treating that case as ``0.0`` keeps the metric defined without
    changing the behavior on real multi-user evaluation splits.

    """

    def compute(self) -> torch.Tensor:
        """Compute personalization, returning ``0.0`` when no user pairs exist."""
        if int(self.total) < 2:
            return torch.zeros((), device=self.total.device, dtype=torch.get_default_dtype())
        return super().compute()


def _rowwise_rank(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based average ranks for each row, handling ties explicitly."""
    n_rows, n_cols = values.shape
    ranks = torch.empty((n_rows, n_cols), device=values.device, dtype=torch.float32)
    base_ranks = torch.arange(n_cols, device=values.device, dtype=torch.float32)

    for row_index in range(n_rows):
        sort_order = torch.argsort(values[row_index], stable=True)
        sorted_values = values[row_index][sort_order]
        sorted_ranks = base_ranks.clone()
        tie_starts = torch.ones(n_cols, device=values.device, dtype=torch.bool)
        tie_starts[1:] = sorted_values[1:] != sorted_values[:-1]
        group_starts = torch.nonzero(tie_starts, as_tuple=False).flatten()
        group_ends = torch.cat(
            [group_starts[1:], group_starts.new_tensor([n_cols])],
        )
        for start, end in zip(group_starts.tolist(), group_ends.tolist(), strict=True):
            sorted_ranks[start:end] = 0.5 * float(start + end - 1)
        ranks[row_index, sort_order] = sorted_ranks
    return ranks


def _rowwise_spearman(values: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
    """Compute a finite Spearman-style correlation for each row."""
    if values.size(-1) < 2:
        return torch.zeros(values.size(0), device=values.device, dtype=torch.float32)
    value_ranks = _rowwise_rank(values.float())
    popularity_ranks = _rowwise_rank(popularity.float())
    value_centered = value_ranks - value_ranks.mean(dim=-1, keepdim=True)
    popularity_centered = popularity_ranks - popularity_ranks.mean(dim=-1, keepdim=True)
    numerator = (value_centered * popularity_centered).sum(dim=-1)
    denominator = torch.sqrt(
        value_centered.square().sum(dim=-1) * popularity_centered.square().sum(dim=-1)
    )
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


class _EvaluatorDiagnosticsAccumulator:
    """Accumulate refined scorer diagnostics across evaluation batches."""

    def __init__(self, top_ks: tuple[int, ...]) -> None:
        """Initialize running sums for score-mix, top-k, and cosine diagnostics."""
        self._top_ks = top_ks
        self._score_mix_sum: dict[str, float] = {name: 0.0 for name in _SCORE_MIX_COMPONENTS}
        self._score_mix_sum_sq: dict[str, float] = {name: 0.0 for name in _SCORE_MIX_COMPONENTS}
        self._score_mix_count = 0
        self._contribution_sum: dict[tuple[str, int], float] = {}
        self._contribution_count: dict[tuple[str, int], int] = {}
        self._popularity_sum: dict[tuple[str, int], float] = {}
        self._popularity_count: dict[tuple[str, int], int] = {}
        self._cosine_sum = 0.0
        self._cosine_sum_sq = 0.0
        self._cosine_count = 0

    def update(
        self,
        score_components: dict[str, torch.Tensor],
        pred_index_mat: torch.Tensor,
        popularity: torch.Tensor,
    ) -> None:
        """Update diagnostics from one evaluated batch."""
        score_mix_weights = score_components.get("score_mix_weights")
        if score_mix_weights is not None:
            weights = score_mix_weights.float()
            self._score_mix_count += weights.size(0)
            for component_index, component_name in enumerate(_SCORE_MIX_COMPONENTS):
                component_weights = weights[:, component_index]
                self._score_mix_sum[component_name] += float(component_weights.sum().item())
                self._score_mix_sum_sq[component_name] += float(
                    component_weights.square().sum().item()
                )

        component_scores = {
            "interest": score_components.get("interest_score"),
            "conformity": score_components.get("conformity_score"),
            "context": score_components.get("context_score"),
            "final": score_components.get("final_score"),
        }

        for top_k in self._top_ks:
            top_indices = pred_index_mat[:, :top_k]
            top_popularity = popularity.index_select(
                0,
                top_indices.reshape(-1),
            ).reshape_as(top_indices)
            for component_name, component_score in component_scores.items():
                if component_score is None:
                    continue
                gathered_scores = component_score.gather(1, top_indices).float()
                if component_name in _SCORE_MIX_COMPONENTS and score_mix_weights is not None:
                    weight_index = _SCORE_MIX_COMPONENTS.index(component_name)
                    gathered_scores = gathered_scores * score_mix_weights[
                        :,
                        weight_index,
                    ].float().unsqueeze(1)
                key = (component_name, top_k)
                self._contribution_sum[key] = self._contribution_sum.get(key, 0.0) + float(
                    gathered_scores.sum().item()
                )
                self._contribution_count[key] = self._contribution_count.get(key, 0) + int(
                    gathered_scores.numel()
                )
                correlations = _rowwise_spearman(gathered_scores, top_popularity)
                self._popularity_sum[key] = self._popularity_sum.get(key, 0.0) + float(
                    correlations.sum().item()
                )
                self._popularity_count[key] = self._popularity_count.get(key, 0) + int(
                    correlations.numel()
                )

        user_interest_emb = score_components.get("user_interest_emb")
        user_conformity_emb = score_components.get("user_conformity_emb")
        if user_interest_emb is not None and user_conformity_emb is not None:
            cosine = torch.nn.functional.cosine_similarity(
                user_interest_emb.float(),
                user_conformity_emb.float(),
                dim=-1,
            )
            self._cosine_sum += float(cosine.sum().item())
            self._cosine_sum_sq += float(cosine.square().sum().item())
            self._cosine_count += int(cosine.numel())

    @staticmethod
    def _population_std(total: float, total_sq: float, count: int) -> float:
        """Return the population standard deviation from running sums."""
        if count == 0:
            return 0.0
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        return variance**0.5

    def compute(self) -> dict[str, float]:
        """Materialize averaged diagnostics with explicit metric names."""
        diagnostics: dict[str, float] = {}
        if self._score_mix_count > 0:
            for component_name in _SCORE_MIX_COMPONENTS:
                total = self._score_mix_sum[component_name]
                total_sq = self._score_mix_sum_sq[component_name]
                diagnostics[f"score_mix_{component_name}_mean"] = total / self._score_mix_count
                diagnostics[f"score_mix_{component_name}_std"] = self._population_std(
                    total,
                    total_sq,
                    self._score_mix_count,
                )

        for component_name in ("interest", "conformity", "context"):
            for top_k in self._top_ks:
                key = (component_name, top_k)
                count = self._contribution_count.get(key, 0)
                if count == 0:
                    continue
                diagnostics[f"{component_name}_contribution@{top_k}"] = (
                    self._contribution_sum[key] / count
                )

        for component_name in ("interest", "conformity", "context", "final"):
            for top_k in self._top_ks:
                key = (component_name, top_k)
                count = self._popularity_count.get(key, 0)
                if count == 0:
                    continue
                diagnostics[f"{component_name}_popularity_spearman@{top_k}"] = (
                    self._popularity_sum[key] / count
                )

        if self._cosine_count > 0:
            diagnostics["interest_conformity_cosine_mean"] = self._cosine_sum / self._cosine_count
            diagnostics["interest_conformity_cosine_std"] = self._population_std(
                self._cosine_sum,
                self._cosine_sum_sq,
                self._cosine_count,
            )
        return diagnostics


class Evaluator:
    """Batched GPU evaluation for the PyG link-prediction metric suite.

    Runtime evaluation is intentionally restricted to the thesis-primary
    metric set so reporting, validation, and saved results stay aligned.

    Ground-truth and exclusion dicts are built on first access per split mask
    and cached for subsequent epochs, avoiding repeated O(N) Python loops and
    D2H transfers.
    """

    def __init__(self, config: UCaGNNConfig) -> None:
        self.config = config
        # Cache keyed by mask tensor identity (id()) to avoid rebuilding per epoch.
        self._split_cache: dict[int, dict] = {}

    @staticmethod
    def _effective_eval_batch_size(
        requested_batch_size: int,
        n_items: int,
        export_score_components: bool,
    ) -> int:
        """Cap eval batch size so full-catalog score tensors stay under the budget."""
        score_matrix_budget_bytes = 512 * 1024**2
        full_score_matrices = 5 if export_score_components else 1
        safe_batch = max(
            1,
            score_matrix_budget_bytes // (n_items * 4 * full_score_matrices),
        )
        return min(requested_batch_size, int(safe_batch))

    def _build_metrics(
        self,
        n_items: int,
        popularity: torch.Tensor,
    ) -> LinkPredMetricCollection:
        """Build the thesis-primary PyG metric bundle.

        ``LinkPredMetricCollection`` still needs one metric instance per metric
        family and cutoff, but runtime updates happen through a single shared
        collection call.
        """
        metrics: dict[str, object] = {}
        for k in THESIS_EVAL_KS:
            metrics[f"NDCG@{k}"] = LinkPredNDCG(k=k)
            metrics[f"Recall@{k}"] = LinkPredRecall(k=k)
            metrics[f"AveragePopularity@{k}"] = LinkPredAveragePopularity(
                k=k,
                popularity=popularity,
            )
            metrics[f"HitRatio@{k}"] = LinkPredHitRatio(k=k)
            metrics[f"Personalization@{k}"] = _SafeLinkPredPersonalization(k=k)
        return LinkPredMetricCollection(metrics)

    @staticmethod
    def _matches_split_mask(target_mask: torch.Tensor, split_mask: torch.Tensor) -> bool:
        """Return whether ``target_mask`` identifies the same split as ``split_mask``."""
        if target_mask is split_mask:
            return True
        if target_mask.shape != split_mask.shape:
            return False
        split_mask_on_target_device = move_tensor_to_device(
            split_mask.bool(),
            target_mask.device,
        )
        return torch.equal(target_mask.bool(), split_mask_on_target_device)

    @staticmethod
    def _observed_non_target_mask(data, target_mask: torch.Tensor) -> torch.Tensor:
        """Return interactions to exclude from the recommendation scoring pool.

        Applies split-aware exclusion so that test-set labels never influence
        model selection through the validation metric used for early stopping:

        - Val evaluation: only training interactions are excluded.  Test items
          remain in the scoring pool so no test-set knowledge reaches early
          stopping or best-model selection.
        - Test evaluation: training + validation interactions are excluded,
          matching the standard temporal evaluation protocol.
        """
        exclude_mask = torch.zeros_like(target_mask, dtype=torch.bool)

        train_mask = getattr(data, "train_mask", None)
        if train_mask is not None:
            exclude_mask |= move_tensor_to_device(
                train_mask.bool(),
                exclude_mask.device,
            )

        # When the caller evaluates the test split, also exclude validation
        # interactions so the test pool contains only items the model has never
        # encountered during training or validation. Accept equivalent copied
        # masks as test masks to keep downstream evaluation scripts safe.
        test_mask = getattr(data, "test_mask", None)
        if test_mask is not None and Evaluator._matches_split_mask(target_mask, test_mask):
            val_mask = getattr(data, "val_mask", None)
            if val_mask is not None:
                exclude_mask |= move_tensor_to_device(
                    val_mask.bool(),
                    exclude_mask.device,
                )

        return exclude_mask

    @staticmethod
    def _group_items_by_user(
        user_nodes: torch.Tensor,
        item_nodes: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Group CPU item IDs by CPU user ID with one pass over unique users.

        Args:
            user_nodes: CPU user IDs for one interaction slice.
            item_nodes: CPU item IDs aligned to ``user_nodes``.

        Returns:
            Mapping from user ID to a CPU ``torch.long`` tensor of item IDs.

        """
        if user_nodes.numel() == 0:
            return {}

        sort_order = user_nodes.argsort(stable=True)
        sorted_users = user_nodes[sort_order]
        sorted_items = item_nodes[sort_order].long()
        unique_users, counts = torch.unique_consecutive(sorted_users, return_counts=True)
        item_groups = sorted_items.split(counts.tolist())
        return {
            int(user_id.item()): items
            for user_id, items in zip(unique_users, item_groups, strict=True)
        }

    def _get_or_build_split_cache(
        self,
        data,
        mask: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """Return (user_gt, user_seen_items, unique_users) for the split mask.

        Built once on first call and stored; subsequent epoch calls for the
        same split (same mask object) return instantly with no Python loops or
        D2H transfers.
        """
        key = id(mask)
        if key not in self._split_cache:
            target_mask = positive_interaction_mask(
                mask.bool(),
                getattr(data, "labels", None),
            )
            if target_mask.device.type != "cpu":
                target_mask = target_mask.cpu()
            exclude_mask = self._observed_non_target_mask(data, mask)
            if exclude_mask.device.type != "cpu":
                exclude_mask = exclude_mask.cpu()

            user_nodes_cpu = data.user_nodes[target_mask]
            item_nodes_cpu = data.item_nodes[target_mask] - data.n_users
            unique_users = user_nodes_cpu.unique()
            user_gt = self._group_items_by_user(user_nodes_cpu, item_nodes_cpu)

            exclude_user_nodes = data.user_nodes[exclude_mask]
            exclude_item_nodes = data.item_nodes[exclude_mask] - data.n_users
            user_seen_items = self._group_items_by_user(
                exclude_user_nodes,
                exclude_item_nodes,
            )

            self._split_cache[key] = {
                "user_gt": user_gt,
                "user_seen_items": user_seen_items,
                "unique_users": unique_users,
            }
        entry = self._split_cache[key]
        return entry["user_gt"], entry["user_seen_items"], entry["unique_users"]

    @staticmethod
    def _get_score_components_from_propagated(
        model,
        propagated: dict[str, torch.Tensor],
        user_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        """Return refined score components when the model exports them."""
        get_components = getattr(model, "get_score_components_from_propagated", None)
        if get_components is None:
            return None
        return get_components(propagated, user_ids)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data,
        mask: torch.Tensor,
        batch_size: int = 512,
        include_refined_diagnostics: bool = False,
    ) -> dict[str, float]:
        """Evaluate model on users present in mask.

        Args:
            model: Model exposing the propagated-score evaluation surface.
            data: Runtime graph data and split masks.
            mask: Split mask to evaluate.
            batch_size: Requested user batch size for ranking metrics.
            include_refined_diagnostics: Whether to append the expensive
                refined scorer diagnostics such as score-mix, Spearman, and
                cosine stats. Keep this opt-in so epoch validation remains
                cheap; the training runner enables it for the final test pass.

        Returns:
            Dict of metric name to scalar value.

        """
        model.eval()
        device = model_device(model)

        user_gt, user_seen_items, unique_users = self._get_or_build_split_cache(
            data,
            mask,
        )
        n_items = data.n_items
        if unique_users.numel() == 0:
            return {}
        unique_users_cpu = unique_users
        unique_users = move_tensor_to_device(unique_users_cpu, device)
        use_eval_amp = device.type == "cuda" and self.config.use_amp

        edge_index, edge_sign, edge_norm = stage_graph_tensors_for_device(
            data,
            device,
        )
        popularity = move_optional_tensor_to_device(
            data.popularity,
            device,
            dtype=torch.bfloat16,
        )
        assert popularity is not None
        metrics = self._build_metrics(n_items=n_items, popularity=popularity)
        metrics = metrics.to(device)
        diagnostics = (
            _EvaluatorDiagnosticsAccumulator(THESIS_EVAL_KS)
            if include_refined_diagnostics
            else None
        )
        export_score_components = (
            include_refined_diagnostics
            and getattr(model, "get_score_components_from_propagated", None) is not None
        )
        branch_ranking_metrics = (
            {
                branch_name: self._build_metrics(n_items=n_items, popularity=popularity).to(
                    device,
                )
                for branch_name in _BRANCH_RANKING_SCORE_KEYS
            }
            if export_score_components and self.config.use_dual_branch
            else {}
        )

        effective_batch = self._effective_eval_batch_size(
            requested_batch_size=batch_size,
            n_items=n_items,
            export_score_components=export_score_components,
        )

        # Propagate once over the full graph; reuse across all user batches.
        with autocast_context(use_amp=use_eval_amp):
            propagated = model.get_propagated_for_eval(
                edge_index,
                edge_sign,
                edge_norm,
                embedding_dtype=torch.bfloat16 if use_eval_amp else None,
            )

        # CAGRA candidate pre-filtering (opt-in, GPU-native ANN via cuVS).
        # When cagra_candidate_k > 0 we restrict per-user scoring to the top-K
        # nearest-neighbor items, drastically reducing eval VRAM.
        cagra_candidate_k = self.config.cagra_candidate_k
        cagra_index = None
        if cagra_candidate_k > 0 and cagra_candidate_k < n_items and device.type == "cuda":
            try:
                import cupy as cp
                from cuvs.neighbors import cagra as cuvs_cagra

                item_key = "item_interest" if "item_interest" in propagated else "item"
                item_embs = propagated[item_key].float().contiguous()
                item_cp = cp.asarray(item_embs.detach())
                index_params = cuvs_cagra.IndexParams(
                    metric="inner_product",
                    graph_degree=self.config.cagra_out_degree,
                    intermediate_graph_degree=self.config.cagra_initial_degree,
                )
                cagra_index = cuvs_cagra.build(index_params, item_cp)
            except Exception as exc:
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "CAGRA candidate pre-filtering disabled: %s", exc
                )
                cagra_index = None

        max_k = max(THESIS_EVAL_KS)
        for start in range(0, unique_users.size(0), effective_batch):
            batch_users = unique_users[start : start + effective_batch]
            batch_user_ids = unique_users_cpu[start : start + effective_batch].tolist()
            score_components: dict[str, torch.Tensor] | None = None
            with autocast_context(use_amp=use_eval_amp):
                if include_refined_diagnostics:
                    score_components = self._get_score_components_from_propagated(
                        model,
                        propagated,
                        batch_users,
                    )
                if score_components is None:
                    scores = model.score_users_from_propagated(
                        propagated,
                        batch_users,
                    )
                else:
                    scores = score_components["final_score"]
            scores = scores.float()

            # Mask out non-candidate items from CAGRA index (if built).
            if cagra_index is not None:
                try:
                    import cupy as cp
                    from cuvs.neighbors import cagra as cuvs_cagra

                    user_key = "user_interest" if "user_interest" in propagated else "user"
                    user_embs = propagated[user_key][batch_users].float().contiguous()
                    user_cp = cp.asarray(user_embs.detach())
                    search_params = cuvs_cagra.SearchParams(
                        itopk_size=max(cagra_candidate_k, self.config.cagra_itopk_size),
                    )
                    _, neighbors_cp = cuvs_cagra.search(
                        search_params, cagra_index, user_cp, cagra_candidate_k
                    )
                    # neighbors_cp: (B, cagra_candidate_k) cupy int64
                    neighbors_t = torch.as_tensor(neighbors_cp, device=device)
                    candidate_mask = torch.zeros(
                        scores.size(0), n_items, dtype=torch.bool, device=device
                    )
                    candidate_mask.scatter_(1, neighbors_t, True)
                    scores[~candidate_mask] = float("-inf")
                except Exception as exc:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "CAGRA search failed, using full-catalog scores: %s", exc
                    )

            seen_row_parts: list[torch.Tensor] = []
            seen_col_parts: list[torch.Tensor] = []
            for row_index, user_id in enumerate(batch_user_ids):
                seen_items = user_seen_items.get(user_id)
                if seen_items is None or seen_items.numel() == 0:
                    continue
                seen_row_parts.append(
                    torch.full((seen_items.numel(),), row_index, dtype=torch.long),
                )
                seen_col_parts.append(seen_items)
            if seen_row_parts:
                seen_rows = move_tensor_to_device(torch.cat(seen_row_parts), device)
                seen_cols = move_tensor_to_device(torch.cat(seen_col_parts), device)
                scores[seen_rows, seen_cols] = float("-inf")
                if score_components is not None:
                    for component_score in score_components.values():
                        if component_score.ndim == 2 and component_score.shape == scores.shape:
                            component_score[seen_rows, seen_cols] = float("-inf")

            # Build batch ground truth on-the-fly (small: batch_size * n_items)
            gt_rows: list[torch.Tensor] = []
            keep_rows: list[int] = []
            for row_index, uid in enumerate(batch_user_ids):
                items = user_gt.get(uid)
                if items is None or items.numel() == 0:
                    continue
                keep_rows.append(row_index)
                gt_rows.append(items)
            if not gt_rows:
                continue

            keep_indices = torch.tensor(keep_rows, device=device, dtype=torch.long)
            scores = scores.index_select(0, keep_indices)
            if scores.size(0) == 0:
                continue
            if score_components is not None:
                score_components = {
                    name: value.index_select(0, keep_indices)
                    for name, value in score_components.items()
                }

            _, pred_index_mat = torch.topk(scores, max_k, dim=-1)
            if diagnostics is not None and score_components is not None:
                diagnostics.update(score_components, pred_index_mat, popularity)

            # Build edge_label_index from sparse ground truth
            gt_counts = torch.tensor([items.numel() for items in gt_rows], device=device)
            gt_user_index = torch.repeat_interleave(
                torch.arange(len(gt_rows), device=device),
                gt_counts,
            )
            gt_item_index = move_tensor_to_device(torch.cat(gt_rows), device)
            edge_label_index = (
                gt_user_index,
                gt_item_index,
            )
            if score_components is not None and branch_ranking_metrics:
                for branch_name, score_keys in _BRANCH_RANKING_SCORE_KEYS.items():
                    branch_scores = next(
                        (
                            score_components[score_key]
                            for score_key in score_keys
                            if score_key in score_components
                        ),
                        None,
                    )
                    if branch_scores is None:
                        continue
                    _, branch_pred_index_mat = torch.topk(
                        branch_scores.float(),
                        max_k,
                        dim=-1,
                    )
                    branch_ranking_metrics[branch_name].update(
                        branch_pred_index_mat,
                        edge_label_index,
                    )
            metrics.update(pred_index_mat, edge_label_index)

        results = {name: value.item() for name, value in metrics.compute().items()}
        for branch_name, metric_collection in branch_ranking_metrics.items():
            results.update(
                {
                    f"{branch_name}_{name}": value.item()
                    for name, value in metric_collection.compute().items()
                }
            )
        if diagnostics is not None:
            results.update(diagnostics.compute())
        return results
