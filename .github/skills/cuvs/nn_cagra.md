# CAGRA (cuVS)

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. It provides state-of-the-art index build and query performance for both small- and large-batch sized search.

---

## Index build parameters

### `cuvs.neighbors.cagra.IndexParams`

Parameters to build an index for CAGRA nearest neighbor search.

#### Constructor signature

```python
class cuvs.neighbors.cagra.IndexParams(
    metric='sqeuclidean',
    *,
    intermediate_graph_degree=128,
    graph_degree=64,
    build_algo='ivf_pq',
    nn_descent_niter=20,
    compression=None,
    ivf_pq_build_params: ivf_pq.IndexParams = None,
    ivf_pq_search_params: ivf_pq.SearchParams = None,
    ace_params: AceParams = None,
    refinement_rate: float = 1.0,
)
```

#### Parameters

- **metric** (`str`, default `'sqeuclidean'`)
  - Distance metric: `sqeuclidean`, `inner_product`, or `cosine`.
  - `sqeuclidean` is Euclidean distance squared (no sqrt).
  - `inner_product` is defined as `sum(a_i * b_i)`.
  - `cosine` is defined as `1 - dot(a, b) / (||a|| * ||b||)`.
- **intermediate_graph_degree** (`int`, default `128`)
- **graph_degree** (`int`, default `64`)
- **build_algo** (`str`, default `'ivf_pq'`)
  - `ivf_pq`: IVF-PQ algorithm.
  - `nn_descent`: experimental NN-Descent algorithm (generally faster than IVF-PQ).
  - `iterative_cagra_search`: iteratively builds the kNN graph using `search()` + `optimize()`.
  - `ace`: ACE (Augmented Core Extraction) for datasets that do not fit in GPU memory.
- **compression** (`CompressionParams`, optional): Enables compression if provided.
- **ivf_pq_build_params** (`cuvs.neighbors.ivf_pq.IndexParams`, optional): Parameters for the IVF-PQ build phase.
- **ivf_pq_search_params** (`cuvs.neighbors.ivf_pq.SearchParams`, optional): Parameters for the IVF-PQ search phase.
- **ace_params** (`AceParams`, optional): Parameters for the ACE build algorithm.
- **refinement_rate** (`float`, default `1.0`)

#### Attributes

- `ace_params`
- `build_algo`
- `compression`
- `graph_degree`
- `intermediate_graph_degree`
- `ivf_pq_build_params`
- `ivf_pq_search_params`
- `metric`
- `nn_descent_niter`
- `refinement_rate`

#### Methods

- `get_handle(self)`

---

## Index search parameters

### `cuvs.neighbors.cagra.SearchParams`

CAGRA search parameters.

#### Constructor signature

```python
class cuvs.neighbors.cagra.SearchParams(
    max_queries=0,
    *,
    itopk_size=64,
    max_iterations=0,
    algo='auto',
    team_size=0,
    search_width=1,
    min_iterations=0,
    thread_block_size=0,
    hashmap_mode='auto',
    hashmap_min_bitlen=0,
    hashmap_max_fill_rate=0.5,
    num_random_samplings=1,
    rand_xor_mask=0x128394,
    persistent=False,
    persistent_lifetime=None,
    persistent_device_usage=None,
)
```

#### Parameters

- **max_queries** (`int`, default `0`): Maximum number of queries to search at once (auto when 0).
- **itopk_size** (`int`, default `64`): Number of intermediate search results retained. Larger values increase accuracy.
- **max_iterations** (`int`, default `0`): Upper limit of search iterations (auto when 0).
- **algo** (`str`, default `'auto'`): Search algorithm.
  - `auto`: select best based on query size.
  - `single_cta`: better when query count is large (>10).
  - `multi_cta`: better when query count is small.
- **team_size** (`int`, default `0`): Number of threads used for a single distance calculation (4, 8, 16, 32).
- **search_width** (`int`, default `1`): Number of graph nodes used as starting points each iteration.
- **min_iterations** (`int`, default `0`): Lower limit of search iterations.
- **thread_block_size** (`int`, default `0`): CUDA thread block size. Values: 0, 64, 128, 256, 512, 1024.
- **hashmap_mode** (`str`, default `'auto'`): Type of hash map to use. Options: `auto`, `small`, `hash`.
- **hashmap_min_bitlen** (`int`, default `0`): Minimum hash map bit length.
- **hashmap_max_fill_rate** (`float`, default `0.5`): Maximum hash map fill rate.
- **num_random_samplings** (`int`, default `1`): Number of random seed selections.
- **rand_xor_mask** (`int`, default `0x128394`): Bit mask for random seed node selection.
- **persistent** (`bool`, default `False`): Whether to use the persistent kernel.
- **persistent_lifetime** (`float`): Time (seconds) before persistent kernel stops when idle.
- **persistent_device_usage** (`float`): Fraction of maximum grid size used by the persistent kernel.

#### Attributes

- `algo`
- `hashmap_max_fill_rate`
- `hashmap_min_bitlen`
- `hashmap_mode`
- `itopk_size`
- `max_iterations`
- `max_queries`
- `min_iterations`
- `num_random_samplings`
- `rand_xor_mask`
- `search_width`
- `team_size`
- `thread_block_size`

#### Methods

- `get_handle(self)`

---

## Index

### `cuvs.neighbors.cagra.Index`

CAGRA index object that stores the trained index state used for nearest neighbor search.

#### Attributes

- `dataset`
- `dim`
- `dtype`
- `graph`
- `graph_degree`
- `trained`

---

## Index build

### `cuvs.neighbors.cagra.build(index_params, dataset, resources=None)`

Build the CAGRA index from a dataset.

- The build process first constructs an intermediate kNN graph, then optimizes it into the final graph.
- Both the dataset and the optimized graph must fit in GPU memory.
- When using the `ace` build algorithm, the dataset must be in host memory (NumPy array or CuPy `.get()`).

#### Supported distance metrics

- `L2`
- `InnerProduct`
- `Cosine`

#### Parameters

- `index_params` (`IndexParams`): Build parameters.
- `dataset`: CUDA array interface compliant matrix `(n_samples, dim)` with dtype in `[float, half, int8, uint8]`.
- `resources`: Optional cuVS resources handle.

#### Returns

- `index`: `cuvs.cagra.Index`

#### Example

```python
import cupy as cp
from cuvs.neighbors import cagra

n_samples = 50000
n_features = 50
k = 10

dataset = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
build_params = cagra.IndexParams(metric="sqeuclidean")
index = cagra.build(build_params, dataset)

distances, neighbors = cagra.search(cagra.SearchParams(), index, dataset, k)
distances = cp.asarray(distances)
neighbors = cp.asarray(neighbors)
```

#### ACE example (host data)

```python
import numpy as np
import cupy as cp
from cuvs.neighbors import cagra

n_samples = 50000
n_features = 50

dataset_host = np.random.random_sample((n_samples, n_features)).astype(np.float32)
ace_params = cagra.AceParams(npartitions=4, use_disk=True, build_dir="/tmp/ace")
build_params = cagra.IndexParams(metric="sqeuclidean", build_algo="ace", ace_params=ace_params)
idx = cagra.build(build_params, dataset_host)
```

---

## Index search

### `cuvs.neighbors.cagra.search(search_params, index, queries, k, neighbors=None, distances=None, resources=None, filter=None)`

Find the k nearest neighbors for each query.

#### Parameters

- `search_params` (`SearchParams`): Search parameters.
- `index` (`Index`): Trained CAGRA index.
- `queries`: CUDA array interface compliant matrix `(n_queries, dim)` with dtype in `[float, int8, uint8]`.
- `k` (`int`): Number of neighbors.
- `neighbors` (optional): Optional output array `(n_queries, k)` of dtype `int64`.
- `distances` (optional): Optional output array `(n_queries, k)` for distances.
- `filter` (optional): `cuvs.neighbors.cuvsFilter` to filter neighbors based on a bitset.
- `resources` (optional): cuVS resource handle.

#### Example

```python
import cupy as cp
from cuvs.neighbors import cagra

n_samples = 50000
n_features = 50
n_queries = 1000
k = 10

dataset = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
index = cagra.build(cagra.IndexParams(), dataset)

queries = cp.random.random_sample((n_queries, n_features), dtype=cp.float32)
search_params = cagra.SearchParams(max_queries=100, itopk_size=64)

distances, neighbors = cagra.search(search_params, index, queries, k)
neighbors = cp.asarray(neighbors)
distances = cp.asarray(distances)
```

---

## Index save

### `cuvs.neighbors.cagra.save(filename, index, include_dataset=True, resources=None)`

Save a CAGRA index to a file.

> **Note:** Saving/loading is experimental and the serialization format may change.

#### Parameters

- `filename` (`str`): File path.
- `index` (`Index`): Trained CAGRA index.
- `include_dataset` (`bool`, default `True`): Whether to write the dataset into the serialized file. If false, you must call `index.update_dataset(dataset)` after loading.
- `resources` (optional): cuVS resource handle.

#### Example

```python
import cupy as cp
from cuvs.neighbors import cagra

n_samples = 50000
n_features = 50

dataset = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
index = cagra.build(cagra.IndexParams(), dataset)

cagra.save("my_index.bin", index)
index_loaded = cagra.load("my_index.bin")
```

---

## Index load

### `cuvs.neighbors.cagra.load(filename, resources=None)`

Load a previously saved index from disk.

> **Note:** The serialization format is experimental, and loading an index saved with a previous version is not guaranteed.

#### Parameters

- `filename` (`str`): File path.
- `resources` (optional): cuVS resource handle.

#### Returns

- `index` (`Index`)

---

## Index extend

### `cuvs.neighbors.cagra.extend(params, index, additional_dataset, resources=None)`

Extend an existing CAGRA index with additional vectors.

#### Parameters

- `params` (`ExtendParams`): Parameters for extension.
- `index` (`Index`): Existing CAGRA index to extend.
- `additional_dataset`: CUDA array interface compliant matrix with dtype in `[float, half, int8, uint8]`.
- `resources` (optional): cuVS resource handle.
