# Distance (cuVS)

Utilities for pairwise distance computations.

---

## Pairwise distance

### `cuvs.distance.pairwise_distance(X, Y, out=None, metric='euclidean', p=2.0, resources=None)`

Compute pairwise distances between X and Y.

#### Parameters

- `X` (CUDA array interface compliant matrix, shape `(m, k)`).
- `Y` (CUDA array interface compliant matrix, shape `(n, k)`).
- `out` (optional): Writable CUDA array interface compliant matrix, shape `(m, n)`.
- `metric` (`str`, default `'euclidean'`): Distance metric. Supported values: `euclidean`, `l2`, `l1`, `cityblock`, `inner_product`, `chebyshev`, `canberra`, `lp`, `hellinger`, `jensenshannon`, `kl_divergence`, `russellrao`, `minkowski`, `correlation`, `cosine`.
- `p` (`float`, default `2.0`): Parameter used by the Minkowski (`minkowski`/`lp`) metric.
- `resources` (optional): cuVS Resource handle for reusing CUDA resources. If supplied, callers must call `resources.sync()` before reading outputs.

#### Returns

- A CUDA array interface compliant matrix of shape `(m, n)` containing pairwise distances.

#### Examples

```python
import cupy as cp
from cuvs.distance import pairwise_distance

n_samples = 5000
n_features = 50

in1 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
in2 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)

output = pairwise_distance(in1, in2, metric='euclidean')
```
