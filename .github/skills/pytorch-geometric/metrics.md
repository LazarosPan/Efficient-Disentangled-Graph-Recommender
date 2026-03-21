# torch_geometric.metrics

Advanced performance evaluation metrics for link prediction and recommendation tasks.

## Contents

- [Link Prediction Metrics](#link-prediction-metrics)

## Link Prediction Metrics

Link prediction metrics for evaluating recommendation and graph link systems.

| Metric | Description |
|--------|-------------|
| `LinkPredMetric` | An abstract class for computing link prediction retrieval metrics. |
| `LinkPredMetricCollection` | A collection of metrics to reduce and speed-up computation of link prediction metrics. |
| `LinkPredPrecision` | A link prediction metric to compute Precision @ k, i.e. the proportion of recommendations within the top-k that are actually relevant. |
| `LinkPredRecall` | A link prediction metric to compute Recall @ k, i.e. the proportion of relevant items that appear within the top-k. |
| `LinkPredF1` | A link prediction metric to compute F1 @ k. |
| `LinkPredMAP` | A link prediction metric to compute MAP @ k (Mean Average Precision), considering the order of relevant items within the top-k. |
| `LinkPredNDCG` | A link prediction metric to compute the NDCG @ k (Normalized Discounted Cumulative Gain). |
| `LinkPredMRR` | A link prediction metric to compute the MRR @ k (Mean Reciprocal Rank), i.e. the mean reciprocal rank of the first correct prediction. |
| `LinkPredHitRatio` | A link prediction metric to compute the hit ratio @ k, i.e. the percentage of users for whom at least one relevant item is present within the top-k recommendations. |
| `LinkPredCoverage` | A link prediction metric to compute the Coverage @ k of predictions, i.e. the percentage of unique items recommended across all users within the top-k. |
| `LinkPredDiversity` | A link prediction metric to compute the Diversity @ k of predictions according to item categories. |
| `LinkPredPersonalization` | A link prediction metric to compute the Personalization @ k, i.e. the dissimilarity of recommendations across different users. |
| `LinkPredAveragePopularity` | A link prediction metric to compute the Average Recommendation Popularity (ARP) @ k, which averages the popularity scores of items within the top-k recommendations. |