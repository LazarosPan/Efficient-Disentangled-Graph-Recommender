# U-CaGNN Architecture Skill

Use this skill when working on model architecture, embeddings, GCN layers, or module design.

## Key Files
- `docs/ucagnn_implementation/architecture.md` - System overview with paper cross-references
- `docs/ucagnn_implementation/models.md` - Module A-F implementation details
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
- `ScoringModule` now supports both fixed fusion priors and a learnable user-conditioned gate via `scoring_weight_mode`, mixing interest, conformity, and popularity scores while keeping the counterfactual score diagnostic-only.
- `EmbeddingModule` now supports optional item-feature fusion when `use_features=True` and canonical item features are available, producing branch-specific item inputs for interest and conformity propagation.
- `EmbeddingModule` now routes branch-aware user selection, raw item popularity, train-split item recency, and optional popularity embeddings through shared private helpers so the full-graph, subgraph, and stacked-embedding paths stay aligned.
- `ScoringModule` now owns both pairwise batch scoring and full-catalog component assembly, with a scorer-owned popularity head that consumes item popularity and train-split recency while preserving the item-level fast path used by evaluation.
- `UCaGNN` now routes subgraph training and full-graph evaluation through shared propagation and training-payload helpers, so pairwise scoring, loss-suite payload assembly, and IPW item selection stay aligned across the runtime paths that still exist.
- `LightGCNBranch` now uses `LGConv(normalize=False)` — degree normalization is no longer computed internally from the passed `edge_index`.  Instead, `DualBranchGCN.forward()` accepts `edge_norm` (pre-computed full-graph `1/sqrt(deg_u * deg_v)`) and combines it with optional sign-aware weights before passing the combined `edge_weight` to each `LGConv` layer.  This ensures training subgraph passes and full-graph evaluation passes use identical normalization factors.
- `DualBranchGCN` keeps sign-aware edge weighting opt-in. On one-sided graphs it keeps the unit LightGCN baseline, and on mixed-sign graphs it preserves that baseline for positive/neutral edges while down-weighting negative edges relative to the learned `alpha_neg / alpha_pos` ratio.

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
