# U-CaGNN Architecture Skill

Use this skill when working on model architecture, embeddings, GCN layers, or module design.

## Key Files
- `.github/skills/ucagnn-implementation/ucagnn-architecture.md` - Routed architecture summary for the current implementation
- `src/models/ucagnn.py` - Main orchestrator
- `src/models/embeddings.py` - Module A: EmbeddingModule
- `src/models/lightgcn.py` - Module B: DualBranchGCN
- `src/models/scoring.py` - Module C: ScoringModule
- `src/models/propensity.py` - Module F: PropensityEstimator

## Paper Sources
| Decision | Source |
|----------|--------|
| Embedding init Uniform(-1,1) | DDCE |
| LightGCN backbone (no W, no activation) | He et al. 2020 |
| 2 GCN layers default | LightGCN, MCLN, CaDSI |
| Asymmetric branch depth option | MGCE |
| No self-loops | LightGCN section 3.1 |
| Sign-aware alpha_pos/alpha_neg | SIGformer |

## Current Architecture Notes
- `DualBranchGCN` now uses explicit branch-specific propagation depths via `interest_gnn_layers` and `conformity_gnn_layers`, while the LightGCN single-branch path uses its own `single_branch_gnn_layers` field.
- `ScoringModule` now supports both fixed fusion priors and a learnable user-conditioned gate via `scoring_weight_mode`, mixing interest, conformity, and popularity scores while exposing the raw component scores and gate weights for diagnostics without deriving an extra pseudo-causal score.
- `EmbeddingModule` now supports optional item-feature fusion when `use_features=True` and canonical item features are available, producing branch-specific item inputs for interest and conformity propagation.
- `EmbeddingModule` now routes branch-aware user selection, raw item popularity, train-split item recency, and optional popularity embeddings through shared private helpers so the full-graph, subgraph, and stacked-embedding paths stay aligned.
- `ScoringModule` now owns both pairwise batch scoring and full-catalog component assembly, with a scorer-owned popularity head that consumes item popularity and train-split recency while preserving the item-level fast path used by evaluation.
- `UCaGNN` now routes subgraph training and full-graph evaluation through shared propagation and training-payload helpers, so pairwise scoring, loss-suite payload assembly, and IPW item selection stay aligned across the runtime paths that still exist.
- Cached evaluation keeps the narrow public model surface: `UCaGNN.get_propagated_for_eval()` caches one propagated full-graph state, `UCaGNN.score_users_from_propagated()` owns the evaluator-facing full-score path, and `UCaGNN.get_all_score_components()` remains the diagnostic full-catalog entry point.
- `LightGCNBranch` now uses repeated sparse adjacency matmuls instead of per-layer `LGConv` gather-scatter calls. `DualBranchGCN.forward()` accepts `edge_norm` (pre-computed full-graph `1/sqrt(deg_u * deg_v)`), combines it with optional sign-aware weights, builds one sparse adjacency matrix per forward pass, and reuses it across all propagation layers. This preserves the LightGCN update exactly while cutting the edge-message materialization cost that was inflating memory on large graphs.
- `DualBranchGCN` keeps sign-aware edge weighting opt-in. On one-sided graphs it keeps the unit LightGCN baseline, and on mixed-sign graphs it preserves that baseline for positive/neutral edges while down-weighting negative edges relative to the learned `alpha_neg / alpha_pos` ratio.
- The default ranking contract is the fused score `alpha * interest + beta * conformity + gamma * popularity`. Alternate score views (`interest_only`, `conformity_only`, `conformity_suppressed`) are explicit train/eval modes rather than hidden changes to the underlying model.
- `PropensityEstimator` is a standalone two-layer MLP reused by `UCaGNN` for inverse propensity weighting. It consumes propagated item embeddings, applies a sigmoid, and clamps to `[propensity_clip_min, propensity_clip_max]`; in theory-facing writeups this should be described as an item-side propensity proxy rather than a fully identified treatment/exposure model.

## Quick Reference
```python
from src.utils.config import UCaGNNConfig
from src.models.ucagnn import UCaGNN

# Create model variant via config preset
config = UCaGNNConfig().preset_full()  # or preset_lightgcn() / preset_dice_like()
model = UCaGNN(
    n_users,
    n_items,
    config,
    item_features=item_features,
    item_popularity=item_popularity,
    item_recency=item_recency,
)
```
