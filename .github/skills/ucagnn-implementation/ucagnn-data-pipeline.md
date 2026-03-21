# U-CaGNN Data Pipeline Skill

Use this skill when working on data loading, graph construction, negative sampling, or dataset handling.

## Key Files
- `docs/ucagnn_implementation/data-pipeline.md` - Data flow with paper cross-references
- `src/data/loaders.py` - Dataset loading (load_dataset)
- `src/data/canonical.py` - CanonicalInteractions format
- `src/data/graph_builder.py` - Graph construction (build_graph)
- `src/data/negative_sampler.py` - NegativeSampler

## Graph Construction Methods
| Method | Function | When to Use |
|--------|----------|-------------|
| `"dense"` | Bipartite edges from training | Default, no embeddings needed |
| `"knn"` | Bipartite + kNN edges | After first epoch (needs embeddings) |
| `"cagra"` | Bipartite + CAGRA ANN | Large-scale, needs RAPIDS cuVS |

## Paper Sources
| Decision | Source |
|----------|--------|
| No self-loops (loop=False) | LightGCN section 3.1 |
| 80/10/10 temporal split | FMMRec, DICE |
| CAGRA ANN acceleration | NVIDIA CAGRA 2024 |

## Quick Reference
```python
from src.data.loaders import load_dataset
from src.data.graph_builder import build_graph

canonical = load_dataset(config.dataset, config.data_dir)
data = build_graph(canonical, config, embeddings=None)  # dense
data = build_graph(canonical, config, embeddings=model.get_stacked_embeddings())  # knn/cagra
```
