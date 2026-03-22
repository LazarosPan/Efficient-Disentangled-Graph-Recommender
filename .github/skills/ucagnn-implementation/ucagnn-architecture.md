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
- `DualBranchGCN` now supports branch-specific propagation depth via `interest_gnn_layers` and `conformity_gnn_layers`; when omitted, both fall back to `n_gnn_layers`.
- `ScoringModule` now supports evaluation-time intervention modes through `eval_scoring_mode` while preserving the existing training-time default score mixture.
- `EmbeddingModule` now supports optional item-feature fusion when `use_features=True` and canonical item features are available, producing branch-specific item inputs for interest and conformity propagation.

## Quick Reference
```python
from src.utils.config import UCaGNNConfig
from src.models.ucagnn import UCaGNN

# Create model variant via config preset
config = UCaGNNConfig().preset_full()  # or preset_lightgcn() / preset_dice_like()
model = UCaGNN(n_users, n_items, config, item_features=item_features, item_popularity=item_popularity)
```
