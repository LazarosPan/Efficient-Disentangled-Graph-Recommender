# K-Means (cuVS)

K-Means clustering utilities and helpers.

---

## K-Means Parameters

### `cuvs.cluster.kmeans.KMeansParams(metric=None, *, n_clusters=None, init_method=None, max_iter=None, tol=None, n_init=None, oversampling_factor=None, batch_samples=None, batch_centroids=None, inertia_check=None, streaming_batch_size=None, hierarchical=None, hierarchical_n_iters=None)`

Hyper-parameters for the k-means algorithm.

#### Parameters

- `metric` (`str`): String denoting the metric type.
- `n_clusters` (`int`): The number of clusters to form and centroids to generate.
- `init_method` (`str`): Initialization method. One of: `KMeansPlusPlus`, `Random`, `Array`.
- `max_iter` (`int`): Maximum number of iterations.
- `tol` (`float`): Relative tolerance regarding inertia to declare convergence.
- `n_init` (`int`): Number of times the algorithm will be run with different seeds.
- `oversampling_factor` (`float`): Oversampling factor for k-means||.
- `batch_samples` (`int`): Number of samples per batch for tiled computation.
- `batch_centroids` (`int`): Number of centroids per batch (0 uses `n_clusters`).
- `inertia_check` (`bool`): If True, check inertia during iterations for early convergence.
- `streaming_batch_size` (`int`): Number of samples per GPU batch when fitting host data (0 = process all at once).
- `hierarchical` (`bool`): Whether to use hierarchical k-means.
- `hierarchical_n_iters` (`int`): Iterations for hierarchical k-means.

#### Attributes

- `batch_centroids`, `batch_samples`, `hierarchical`, `hierarchical_n_iters`, `inertia_check`, `init_method`, `max_iter`, `metric`, `n_clusters`, `n_init`, `oversampling_factor`, `streaming_batch_size`, `tol`.

---

## K-Means fit

### `cuvs.cluster.kmeans.fit(params, X, centroids=None, sample_weights=None, resources=None)`

Find clusters with the k-means algorithm. When `X` is a device array (CUDA array interface) on-device k-means is used. For host arrays, data is streamed to the GPU in batches controlled by `params.streaming_batch_size`.

#### Parameters

- `params` (`KMeansParams`): Parameters for fitting.
- `X` (array-like): Training instances, shape `(m, k)`.
- `centroids` (optional): Initial centroids, shape `(n_clusters, k)`.
- `sample_weights` (optional): Weights per observation; must reside in same memory space as `X`.
- `resources` (optional): cuVS Resource handle.

#### Returns

- `centroids` (`raft.device_ndarray`): Computed centroids for each cluster.
- `inertia` (`float`): Sum of squared distances of samples to their closest cluster center.
- `n_iter` (`int`): Number of iterations used.

#### Examples

```python
import cupy as cp
from cuvs.cluster.kmeans import fit, KMeansParams

n_samples = 5000
n_features = 50
n_clusters = 3

X = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
params = KMeansParams(n_clusters=n_clusters)
centroids, inertia, n_iter = fit(params, X)

# Host-data (batched) example
import numpy as np
X_host = np.random.random((10_000_000, 128)).astype(np.float32)
params = KMeansParams(n_clusters=1000, streaming_batch_size=1_000_000)
centroids, inertia, n_iter = fit(params, X_host)
```

---

## K-Means predict

### `cuvs.cluster.kmeans.predict(params, X, centroids, sample_weights=None, labels=None, normalize_weight=True, resources=None)`

Predict cluster labels for input data.

#### Parameters

- `params` (`KMeansParams`)
- `X` (array-like): Input matrix shape `(m, k)`.
- `centroids` (array-like): Centroids computed by `fit`, shape `(n_clusters, k)`.
- `sample_weights` (optional)
- `labels` (optional): Preallocated output array to hold labels.
- `normalize_weight` (`bool`): Whether to normalize weights.
- `resources` (optional)

#### Returns

- `labels` (`raft.device_ndarray`) — label for each datapoint in `X`.
- `inertia` (`float`) — sum of squared distances to nearest centroid.

#### Examples

```python
import cupy as cp
from cuvs.cluster.kmeans import fit, predict, KMeansParams

n_samples = 5000
n_features = 50
n_clusters = 3

X = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
params = KMeansParams(n_clusters=n_clusters)
centroids, inertia, n_iter = fit(params, X)
labels, inertia = predict(params, X, centroids)
```

---

## Cluster cost

### `cuvs.cluster.kmeans.cluster_cost(X, centroids, resources=None)`

Compute cluster cost (inertia) given inputs and centroids.

#### Parameters

- `X` (array-like): Input matrix shape `(m, k)`.
- `centroids` (array-like): Centroids shape `(n_clusters, k)`.
- `resources` (optional)

#### Returns

- `inertia` (`float`): The cluster cost between the input matrix and centroids.

#### Examples

```python
import cupy as cp
from cuvs.cluster.kmeans import cluster_cost

n_samples = 5000
n_features = 50
n_clusters = 3

X = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
centroids = cp.random.random_sample((n_clusters, n_features), dtype=cp.float32)

inertia = cluster_cost(X, centroids)
```
