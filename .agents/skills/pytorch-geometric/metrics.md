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


# PyTorch Geometric Link Prediction Metrics

## torch_geometric.metrics.LinkPredMetric

### class LinkPredMetric(k: int)[source]

Bases: `_LinkPredMetric`

An abstract class for computing link prediction retrieval metrics.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None[source]

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor[source]

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredMetricCollection

### class LinkPredMetricCollection(metrics: Union[List[LinkPredMetric], Dict[str, LinkPredMetric]])[source]

Bases: `ModuleDict`

A collection of metrics to reduce and speed-up computation of link prediction metrics.

```python
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredMetricCollection,
    LinkPredPrecision,
    LinkPredRecall,
)

metrics = LinkPredMetricCollection([
    LinkPredMAP(k=10),
    LinkPredPrecision(k=100),
    LinkPredRecall(k=50),
])

metrics.update(pred_index_mat, edge_label_index)
out = metrics.compute()
metrics.reset()

print(out)
>>> {'LinkPredMAP@10': tensor(0.375),
...  'LinkPredPrecision@100': tensor(0.127),
...  'LinkPredRecall@50': tensor(0.483)}
```

**Parameters:**

- **metrics** (`Union[List[LinkPredMetric], Dict[str, LinkPredMetric]]`) – The link prediction metrics.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None[source]

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Dict[str, Tensor][source]

Computes the final metric values.

**Return type:**

`Dict[str, Tensor]`

### reset() → None[source]

Reset metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredPrecision

### class LinkPredPrecision(k: int)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute Precision @ $k$, i.e. the proportion of recommendations within the top-$k$ that are actually relevant.

A higher precision indicates the model's ability to surface relevant items early in the ranking.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredRecall

### class LinkPredRecall(k: int, weighted: bool = False)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute Recall @ $k$, i.e. the proportion of relevant items that appear within the top-$k$.

A higher recall indicates the model's ability to retrieve a larger proportion of relevant items.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.
- **weighted** (`bool`, optional) – If set to `True`, computes weighted recall. (default: `False`)

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredF1

### class LinkPredF1(k: int)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute F1 @ $k$.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredMAP

### class LinkPredMAP(k: int)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute MAP @ $k$ (Mean Average Precision), considering the order of relevant items within the top-$k$.

MAP @ $k$ can provide a more comprehensive view of ranking quality than precision alone.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredNDCG

### class LinkPredNDCG(k: int, weighted: bool = False)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute the NDCG @ $k$ (Normalized Discounted Cumulative Gain).

In particular, can account for the position of relevant items by considering relevance scores, giving higher weight to more relevant items appearing at the top.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.
- **weighted** (`bool`, optional) – If set to `True`, assumes sorted lists of ground-truth items according to a relevance score as given by `edge_label_weight`. (default: `False`)

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredMRR

### class LinkPredMRR(k: int)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute the MRR @ $k$ (Mean Reciprocal Rank), i.e. the mean reciprocal rank of the first correct prediction (or zero otherwise).

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredHitRatio

### class LinkPredHitRatio(k: int)[source]

Bases: `LinkPredMetric`

A link prediction metric to compute the hit ratio @ $k$, i.e. the percentage of users for whom at least one relevant item is present within the top-$k$ recommendations.

A high ratio signifies the model's effectiveness in satisfying a broad range of user preferences.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredCoverage

### class LinkPredCoverage(k: int, num_dst_nodes: int)[source]

Bases: `_LinkPredMetric`

A link prediction metric to compute the Coverage @ $k$ of predictions, i.e. the percentage of unique items recommended across all users within the top-$k$.

Higher coverage indicates a wider exploration of the item catalog.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.
- **num_dst_nodes** (`int`) – The total number of destination nodes.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None[source]

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor[source]

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredDiversity

### class LinkPredDiversity(k: int, category: Tensor)[source]

Bases: `_LinkPredMetric`

A link prediction metric to compute the Diversity @ $k$ of predictions according to item categories.

Diversity is computed as:

$$div_{u@k} = 1 - \left( \frac{1}{k \cdot (k-1)} \right) \sum_{i \neq j} sim(i, j)$$

where:

$$sim(i,j) = \begin{cases}
    1 & \quad \text{if } i,j \text{ share category,}\\
    0 & \quad \text{otherwise.}
\end{cases}$$

This measures the pair-wise inequality of recommendations according to item categories.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.
- **category** (`torch.Tensor`) – A vector that assigns each destination node to a specific category.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None[source]

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor[source]

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredPersonalization

### class LinkPredPersonalization(k: int, max_src_nodes: Optional[int] = 4096, batch_size: int = 65536)[source]

Bases: `_LinkPredMetric`

A link prediction metric to compute the Personalization @ $k$, i.e. the dissimilarity of recommendations across different users.

Higher personalization suggests that the model tailors recommendations to individual user preferences rather than providing generic results.

Dissimilarity is defined by the average inverse cosine similarity between users' lists of recommendations.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.
- **max_src_nodes** (`int`, optional) – The maximum source nodes to consider to compute pair-wise dissimilarity. If specified, Personalization @ $k$ is approximated to avoid computation blowup due to quadratic complexity. (default: `2**12`)
- **batch_size** (`int`, optional) – The batch size to determine how many pairs of user recommendations should be processed at once. (default: `2**16`)

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None[source]

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor[source]

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`

---

## torch_geometric.metrics.LinkPredAveragePopularity

### class LinkPredAveragePopularity(k: int, popularity: Tensor)[source]

Bases: `_LinkPredMetric`

A link prediction metric to compute the Average Recommendation Popularity (ARP) @ $k$, which provides insights into the model's tendency to recommend popular items by averaging the popularity scores of items within the top-$k$ recommendations.

**Parameters:**

- **k** (`int`) – The number of top-$k$ predictions to evaluate against.
- **popularity** (`torch.Tensor`) – The popularity of every item in the training set, e.g., the number of times an item has been rated.

### update(pred_index_mat: Tensor, edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]], edge_label_weight: Optional[Tensor] = None) → None[source]

Updates the state variables based on the current mini-batch prediction.

`update()` can be repeated multiple times to accumulate the results of successive predictions, e.g., inside a mini-batch training or evaluation loop.

**Parameters:**

- **pred_index_mat** (`torch.Tensor`) – The top-$k$ predictions of every example in the mini-batch with shape `[batch_size, k]`.
- **edge_label_index** (`torch.Tensor`) – The ground-truth indices for every example in the mini-batch, given in COO format of shape `[2, num_ground_truth_indices]`.
- **edge_label_weight** (`torch.Tensor`, optional) – The weight of the ground-truth indices for every example in the mini-batch of shape `[num_ground_truth_indices]`. If given, needs to be a vector of positive values. Required for weighted metrics, ignored otherwise. (default: `None`)

**Return type:**

`None`

### compute() → Tensor[source]

Computes the final metric value.

**Return type:**

`Tensor`

### reset() → None

Resets metric state variables to their default value.

**Return type:**

`None`