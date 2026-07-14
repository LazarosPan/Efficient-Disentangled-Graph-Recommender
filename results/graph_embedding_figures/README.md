# Graph and Embedding Figures

This directory contains the consolidated thesis-facing visualization set. The old
per-dataset `*_umap_checkpoint/` and `*_feature_umap/` folders were intentionally
replaced because they repeated the same topology and degree plots.

These figures are qualitative evidence. They explain the EDGRec data and model
geometry, but ranking, feature-selection, and method-comparison claims still need
the validation/full-data result tables.

## Artifact Inventory

| File | What it shows | How to read it |
| --- | --- | --- |
| `core_gcn_topology_networkx_spring.png` | Four NetworkX spring-layout views of fixed-budget train-row samples from the GCN training view. | Blue nodes are users, yellow nodes are items, and green edges are displayed `label > 0` interactions used by GCN propagation. Gray/red edges show displayed `sign = 0`/`sign < 0` feedback rows. `sign > 0` rows are counted in the panel notes and `core_graph_embedding_figures.json` rather than overdrawn because they usually coincide with label-positive propagation rows. |
| `core_train_degree_distributions.png` | Full positive-train user/item degree distributions. | Histograms are the GCN scale check; label/sign row summaries are kept in `core_graph_embedding_figures.json` instead of crowding the plot. |
| `core_learned_interest_umap.png` | UMAP projections of propagated learned EDGRec interest embeddings from reference checkpoints. | Blue/yellow fill separates users/items; gray/red outlines mark sampled nodes with `sign = 0` or `sign < 0` feedback. Use local neighborhoods only. |
| `core_item_feature_umap.png` | UMAP projections of encoded thesis-policy item features. | Points are items, colored by `log1p(train degree)` and outlined when `sign = 0` or `sign < 0` train feedback exists. AmazonBook is blank because it is graph-only in this feature policy. |
| `core_graph_embedding_figures.json` | Machine-readable counts, checkpoint names, projection settings, and trustworthiness values. | Use this for exact captions and appendix tables. |

## Dataset Scale

| Dataset | Reference view | Full positive train graph | Full sign distribution | Topology core sample shown |
| --- | --- | ---: | ---: | ---: |
| AmazonBook | `default preprocessing; observed_interaction_items` | 2,093,875 edges, 52,643 users, 90,781 items | sign > 0 0; sign = 0 2,093,875; sign < 0 0 | 560 users, 174 items, 1,200 displayed rows; label > 0 1,200; sign rows omitted (constant sign = 0) |
| MovieLens 1M | `movielens_explicit; observed_interaction_items` | 469,232 edges, 6,038 users, 3,465 items | sign > 0 469,232; sign = 0 204,578; sign < 0 123,948 | 389 users, 168 items, 1,200 displayed rows; label > 0 840; sign > 0 / = 0 / < 0 840/224/136 |
| KuaiRec v2 | `kuairec_watchratio; observed_interaction_items` | 5,538,839 edges, 7,176 users, 9,669 items | sign > 0 3,008,406; sign = 0 563; sign < 0 5,228,971 | 424 users, 168 items, 1,200 displayed rows; label > 0 1,163; sign > 0 / = 0 / < 0 488/1/711 |
| KuaiRand 1K | `kuairand_causal; random_exposure_items_only` | 5,986 edges, 812 users, 3,963 items | sign > 0 2,986; sign = 0 31,014; sign < 0 25 | 345 users, 294 items, 1,200 displayed rows; label > 0 763; sign > 0 / = 0 / < 0 387/811/2 |

## Interpretation Notes

- `label` and `sign` are deliberately separate. `label > 0` defines the observed positive train graph used by LightGCN/EDGRec propagation and by BPR positives. `sign` is the graded feedback descriptor retained for diagnostics and signed-feedback analysis.
- If a dataset has no graded sign variation, as in AmazonBook where the canonical sign is stored as zero for every observed interaction, the topology does not draw a separate `sign = 0` overlay. Otherwise the gray overlay would simply cover the same positive graph and make the propagation edges harder to read.
- The GCN topology figure is a NetworkX visualization of a fixed displayed-row budget, not the full graph and not a uniform random sample. Plotting millions of edges would collapse into an unreadable mass, while uniform row sampling mostly shows isolated one-edge nodes in sparse datasets. The sampler therefore prioritizes a degree-aware `label > 0` propagation core and reserves a smaller budget for `sign = 0` and `sign < 0` diagnostics. The exact displayed counts are printed in each panel and recorded in JSON.
- The degree-distribution figure is the main answer to whether the data scale is real: it is computed over the complete positive training graph, so it should be cited when explaining popularity skew, negative sampling, and why train-only popularity is a controlled input. The binary label and graded sign counts are kept in the metadata table instead of being drawn on top of the histogram.
- The learned-interest UMAP figure can support a qualitative discussion of whether trained user and item embeddings occupy coherent local neighborhoods. Gray/red outlines show whether sampled nodes also receive `sign = 0` or `sign < 0` observed feedback. Do not use it to claim that one method ranks better than another.
- The feature UMAP figure helps explain what the safe item features look like after encoding and whether feature-space neighborhoods align with popularity and signed-feedback exposure. It supports the feature-engineering narrative, while usefulness still comes from `results/feature_analysis/` and matched full-data rows.

## Projection Metadata

| Dataset | Learned UMAP | Feature UMAP |
| --- | --- | --- |
| AmazonBook | 500 points, 64 dims, trust 0.933 | not available |
| MovieLens 1M | 500 points, 64 dims, trust 0.821 | 250 points, 18 dims, trust 0.956 |
| KuaiRec v2 | 500 points, 64 dims, trust 0.759 | 250 points, 44 dims, trust 0.943 |
| KuaiRand 1K | 500 points, 64 dims, trust 0.821 | 250 points, 68 dims, trust 0.941 |

## Thesis Caption Starters

- GCN topology: "NetworkX spring-layout visualizations of fixed-budget readable samples from the observed training view. Green edges are `label > 0` user-item rows used for EDGRec/LightGCN propagation. Gray and red edges show displayed `sign = 0` and `sign < 0` observed training feedback where available. `sign > 0` rows are counted in the panel notes and metadata rather than overdrawn because they usually coincide with the green propagation graph. Sign-colored rows are diagnostic feedback rows, not extra message-passing edges. Node size is log-scaled positive train degree; validation and test interactions are excluded."
- Degree distribution: "Full positive-training-degree distributions show dataset sparsity and popularity skew for the actual propagation graph; aligned label/sign row summaries are reported separately so non-positive rows are not visually confused with GCN edges."
- Learned UMAP: "UMAP projections of propagated EDGRec interest embeddings provide qualitative geometry diagnostics; gray/red outlines indicate sampled nodes with `sign = 0` or `sign < 0` feedback, and the figure is read with the metric tables rather than as standalone evidence."
- Feature UMAP: "Encoded item-feature projections show the structure of the safe thesis-policy descriptors, train-degree popularity, and where signed non-positive feedback occurs in feature space."
