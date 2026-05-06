# LayerGCN - Standalone Developer Blueprint (Code-First, Repository-Derived)

## Execution Map

| Path | Entry script | Runtime chain | Instantiated symbol | Parameter-owning `nn.Module` | Initialization style |
|---|---|---|---|---|---|
| Canonical repo path | `external/LayerGCN/main.py` | `main.py` -> `utils.quick_start.quick_start()` -> `utils.utils.get_model('LayerGCN')` -> `models.layergcn.LayerGCN(config, train_data)` | `LayerGCN` | `LayerGCN` itself | `nn.init.xavier_uniform_` on raw `nn.Parameter` tensors |

The executable path is single-stack and direct. There is no wrapper module that owns the learnable embeddings; `LayerGCN` allocates `self.user_embeddings` and `self.item_embeddings` itself.

**Evidence**: `external/LayerGCN/main.py::__main__`, `external/LayerGCN/utils/quick_start.py::quick_start`, `external/LayerGCN/utils/utils.py::get_model`, `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `LayerGCN.user_embeddings`, `LayerGCN.item_embeddings`.

Two skeptical execution notes matter:

1. The real training graph is built from the **training split only**, because `quick_start()` instantiates the model with `train_data`, and `LayerGCN.__init__` calls `dataset.inter_matrix(...)` on that loader.
2. The README says extra parameters may be specified on the command line, but `main.py` only parses `--model` and `--dataset`; `gpu_id` is hard-coded in `config_dict`, and all other settings come from YAML plus the hyper-parameter loop.

**Evidence**: `external/LayerGCN/utils/quick_start.py::quick_start`, `external/LayerGCN/models/common/abstract_recommender.py::GeneralRecommender.__init__`, `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `external/LayerGCN/utils/configurator.py::Config.__init__`, `Config._load_dataset_model_config`.

## Mechanical Summary

The implementation is mechanically simple. Its distinctive behavior is not a new graph normalization formula, but a specific sequence of sparse propagation, cosine-based node reweighting, and sum-only layer pooling.

1. **Adjacency normalization**: the base graph is still standard symmetric bipartite normalization `D^{-1/2} A D^{-1/2}` with no self-loops.
2. **Propagation refinement**: after each `torch.sparse.mm`, the layer output is compared against the original embedding table with `F.cosine_similarity`, and each node embedding is rescaled by that scalar.
3. **Layer combination**: final embeddings are the **sum of refined propagated layers only**. The raw layer-0 embeddings are excluded, and there are no learned layer weights.
4. **Graph dropout path**: training can replace the full normalized adjacency with a pruned adjacency rebuilt every epoch; evaluation always uses the full normalized graph.

For the requested forensic questions:

- **Is adjacency normalization different from vanilla LightGCN?** No at the base graph level. The square adjacency in `get_norm_adj_mat()` is standard symmetric normalization. The change is the per-epoch pruned re-normalization plus cosine refinement.
- **Is there an extra normalization step inside propagation?** Not an explicit L2 normalize/re-project step. The extra operation is cosine similarity against the ego embedding, which internally uses normalized vectors to produce a scalar gate.
- **Is layer combination a mean or learned gate?** Neither. It is a plain sum across refined propagated layers. Learned `alpha`, `beta`, `gamma`, or gating parameters are **NOT FOUND** in the active LayerGCN code path.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_norm_adj_mat`, `LayerGCN.pre_epoch_processing`, `LayerGCN._normalize_adj_m`, `LayerGCN.forward`.

---

## 1) Core Class Structure and Initialization

### 1.1 Canonical class

The runtime instantiates `models.layergcn.LayerGCN`, and that same class both owns parameters and implements training/evaluation behavior.

```python
# external/LayerGCN/utils/quick_start.py
model = get_model(config['model'])(config, train_data).to(config['device'])

# external/LayerGCN/models/layergcn.py
class LayerGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LayerGCN, self).__init__(config, dataset)
```

`GeneralRecommender` supplies `self.n_users`, `self.n_items`, `self.batch_size`, and `self.device` from the training dataloader context.

**Evidence**: `external/LayerGCN/utils/quick_start.py::quick_start`, `external/LayerGCN/models/layergcn.py::LayerGCN`, `external/LayerGCN/models/common/abstract_recommender.py::GeneralRecommender.__init__`.

### 1.2 Learnable parameters

The only learnable tensors in the LayerGCN class are the user and item embedding tables:

```python
self.user_embeddings = nn.Parameter(
    nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim))
)
self.item_embeddings = nn.Parameter(
    nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim))
)
```

There is no separate projection layer, no MLP, no attention block, and no learnable layer-combination vector.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `LayerGCN.user_embeddings`, `LayerGCN.item_embeddings`.

### 1.3 Fixed runtime state

The main fixed tensors and scalars are:

- `self.interaction_matrix`: SciPy COO user-item matrix from the train split.
- `self.norm_adj_matrix`: full symmetric normalized square adjacency.
- `self.masked_adj`: per-epoch pruned adjacency or full adjacency when dropout is disabled.
- `self.forward_adj`: the adjacency actually consumed by `forward()`.
- `self.edge_indices`, `self.edge_values`: rectangular bipartite edge list plus normalized edge weights.
- `self.dropout`, `self.n_layers`, `self.reg_weight`, `self.pruning_random`: configuration/runtime control values.

`self.mf_loss = BPRLoss()` is allocated but not used by the active loss path; `calculate_loss()` calls the local `bpr_loss()` implementation instead.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `LayerGCN.get_edge_info`, `LayerGCN.calculate_loss`, `LayerGCN.bpr_loss`.

### 1.4 Configuration surface

The active model YAML exposes:

- `embedding_size: 64`
- `n_layers: [4]`
- `reg_weight: [1e-02, 1e-03, 1e-04, 1e-05]`
- `dropout: [0.0, 0.1, 0.2]`
- `hyper_parameters: ["n_layers", "dropout", "reg_weight"]`

`quick_start()` prepends `seed` from `configs/overall.yaml`, then iterates over the Cartesian product. Under the shipped defaults, that yields `1 * 1 * 3 * 4 = 12` runs per invocation.

**Evidence**: `external/LayerGCN/configs/model/LayerGCN.yaml`, `external/LayerGCN/configs/overall.yaml::seed`, `external/LayerGCN/utils/quick_start.py::quick_start`, `quick_start.hyper_parameters`, `quick_start.combinators`.

---

## 2) The Propagation Kernel

### 2.1 Full adjacency precomputation

`get_norm_adj_mat()` constructs a square bipartite adjacency over `n_users + n_items` nodes:

1. Insert user->item edges into a DOK sparse matrix.
2. Insert mirrored item->user edges.
3. Compute per-node degree from the binary adjacency.
4. Apply symmetric degree normalization `D^{-1/2} A D^{-1/2}`.
5. Convert the result to a `torch.sparse.FloatTensor`.

There are no self-connections in this construction.

```python
A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items))
...
L = D * A * D
return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
```

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_norm_adj_mat`, `LayerGCN.n_nodes`.

### 2.2 Rectangular edge weights for pruning

`get_edge_info()` separately stores the original user-item edges in rectangular form:

- `edge_indices`: shape `[2, E]`, where row 0 is user IDs and row 1 is item IDs.
- `edge_values`: shape `[E]`, computed by `_normalize_adj_m()` as `deg_u^{-1/2} * deg_i^{-1/2}` on the rectangular bipartite graph.

Those weights are later reused as sampling probabilities for the weighted pruning path.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_edge_info`, `LayerGCN._normalize_adj_m`, `LayerGCN.edge_indices`, `LayerGCN.edge_values`.

### 2.3 Pre-epoch edge pruning

`pre_epoch_processing()` decides which adjacency to use for the next epoch:

- If `dropout <= 0`, `self.masked_adj = self.norm_adj_matrix`.
- Otherwise, it keeps `int(E * (1 - dropout))` edges.

The edge selection alternates by epoch:

1. **Weighted pruning path** (`self.pruning_random == False` initially): `torch.multinomial(self.edge_values, keep_len)` samples edges with probability proportional to the stored normalized edge weights. Because those weights shrink on high-degree endpoints, the sampling is biased away from hub-heavy edges.
2. **Random pruning path**: `random.sample(...)` selects edges uniformly.

After selecting edges, the code re-normalizes the retained rectangular bipartite graph with `_normalize_adj_m()`, mirrors the edges into a square adjacency, and stores the result as `self.masked_adj`.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.pre_epoch_processing`, `LayerGCN._normalize_adj_m`.

### 2.4 Forward pass: actual operator sequence

The forward pass uses raw `torch.sparse.mm`, not PyG, DGL, or a custom `MessagePassing` subclass.

For each layer:

1. Concatenate user and item embeddings into `ego_embeddings`.
2. Multiply by the currently active sparse adjacency: `all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)`.
3. Compute node-wise cosine similarity to the original table: `_weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)`.
4. Scale every node row by that scalar: `all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)`.
5. Append the refined layer output to `embeddings_layers`.

The cosine weights are signed. Negative cosine similarity will flip the sign of that node's layer output.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_ego_embeddings`, `LayerGCN.forward`.

### 2.5 Hidden/nonstandard behaviors

- **Self-connections**: **NOT FOUND** in the propagation graph.
- **Graph-library kernel**: **NOT FOUND**; the implementation uses only `torch.sparse.mm`.
- **Learned per-layer scalars/gates**: **NOT FOUND**.
- **Extra normalization inside the loop**: present indirectly as cosine similarity, not as a separate explicit `F.normalize(...)` or norm division.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_norm_adj_mat`, `LayerGCN.forward`.

---

## 3) Layer-Wise Pooling and Final Representation

### 3.1 Exact combination rule

LayerGCN does **not** average `[ego, layer1, layer2, ...]` as vanilla LightGCN does in this repo's `models/lightgcn.py`.

Instead, it computes:

1. `H^(0)` = concatenated user/item embedding table.
2. For each layer `l`, `Z^(l) = A Z^(l-1)`.
3. `w^(l) = cosine(Z^(l), H^(0))` as a node-wise scalar vector.
4. `R^(l) = w^(l) ⊙ Z^(l)`.
5. `H_final = sum_l R^(l)`.

The code never appends `H^(0)` to `embeddings_layers`, so the final embedding excludes raw embeddings.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.forward`.

### 3.2 Splitting users and items

After summing the refined layer outputs, the code splits the combined tensor by fixed offsets:

```python
user_all_embeddings, item_all_embeddings = torch.split(
    ui_all_embeddings, [self.n_users, self.n_items]
)
```

So user rows occupy `[0 : n_users)` and item rows occupy `[n_users : n_users + n_items)`.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.forward`, `LayerGCN.n_users`, `LayerGCN.n_items`.

### 3.3 Ranking scores

Training scores are plain dot products:

```python
pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
```

Evaluation scores are dense user-by-all-item matrix multiplies:

```python
scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
```

There is no margin network, bias term, temperature parameter, or additional projector in the active LayerGCN path.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.bpr_loss`, `LayerGCN.full_sort_predict`.

### 3.4 Loss implementation

The active loss is:

```python
loss = mf_loss + self.reg_weight * reg_loss
```

with:

- `mf_loss = sum(-log(sigmoid(pos - neg)))`
- `reg_loss = 0.5 * sum(||u_ego||^2 + ||i_pos_ego||^2 + ||i_neg_ego||^2)`

The regularizer uses the **raw** embedding tables, not the propagated embeddings.

One important detail: `bpr_loss()` uses `torch.sum`, not batch mean, so the ranking-loss magnitude scales with batch size.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.bpr_loss`, `LayerGCN.emb_loss`, `LayerGCN.calculate_loss`, `external/LayerGCN/models/common/loss.py::L2Loss.forward`.

---

## 4) Data Flow and Tensor Shapes

Let:

- `U` = number of train users after filtering and remapping
- `I` = number of train items after filtering and remapping
- `E` = number of train interactions
- `D` = embedding size (`64` by default)
- `L` = number of GCN layers (`4` under the shipped LayerGCN config)
- `B` = train batch size
- `B_u` = number of evaluation users in one eval batch

### 4.1 Dataset ingestion and split

For the provided `food`/`games` dataset configs:

1. Load `[item, user, timestamp]` from CSV.
2. Apply 5-core user/item filtering.
3. Split globally by timestamp quantiles using ratios `[0.7, 0.1, 0.2]`.
4. Remap IDs using **training users/items only**.
5. Drop validation/test rows containing unseen users/items after remapping.

This means the model graph and ID space are train-split-only.

**Evidence**: `external/LayerGCN/configs/dataset/food.yaml`, `external/LayerGCN/configs/dataset/games.yaml`, `external/LayerGCN/configs/overall.yaml::split_ratio`, `external/LayerGCN/utils/dataset.py::RecDataset._from_scratch`, `RecDataset._filter_by_k_core`, `RecDataset.split`.

### 4.2 Graph tensors

The graph-related shapes are:

- `interaction_matrix`: SciPy COO, shape `[U, I]`, `nnz = E`
- `edge_indices`: torch long tensor, shape `[2, E]`
- `edge_values`: torch float tensor, shape `[E]`
- `norm_adj_matrix`: torch sparse tensor, shape `[U + I, U + I]`, approximately `2E` nonzeros
- `masked_adj`: same shape as `norm_adj_matrix`, with approximately `2 * E_keep` nonzeros when dropout is active

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `LayerGCN.get_edge_info`, `LayerGCN.get_norm_adj_mat`, `LayerGCN.pre_epoch_processing`.

### 4.3 Training batch path

`TrainDataLoader._get_neg_sample()` returns a tensor with shape `[3, B]`:

- row 0: users
- row 1: positive items
- row 2: sampled negative items

Inside `calculate_loss()` those become three tensors of shape `[B]`.

**Evidence**: `external/LayerGCN/utils/dataloader.py::TrainDataLoader._get_neg_sample`, `external/LayerGCN/models/layergcn.py::LayerGCN.calculate_loss`.

### 4.4 Forward-pass shapes

For one full graph forward pass:

1. `ego_embeddings = cat([user_embeddings, item_embeddings], 0)` -> `[U + I, D]`
2. After each sparse propagation -> `[U + I, D]`
3. `_weights = cosine_similarity(...)` -> `[U + I]`
4. Refined layer output after `einsum` -> `[U + I, D]`
5. `stack(embeddings_layers, dim=0)` -> `[L, U + I, D]`
6. `sum(..., dim=0)` -> `[U + I, D]`
7. Split -> user tensor `[U, D]`, item tensor `[I, D]`

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_ego_embeddings`, `LayerGCN.forward`.

### 4.5 Loss-selection shapes

For a training batch:

- `u_embeddings[user]` -> `[B, D]`
- `i_embeddings[pos_item]` -> `[B, D]`
- `i_embeddings[neg_item]` -> `[B, D]`
- `pos_scores`, `neg_scores` -> `[B]`
- `loss` -> scalar

Regularization uses the raw tables with the same `[B, D]` gather pattern.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.bpr_loss`, `LayerGCN.emb_loss`.

### 4.6 Evaluation path

`EvalDataLoader` yields:

- `batch_users`: shape `[B_u]`
- `batch_mask_matrix`: shape `[2, M]`, where `M` is the number of train positives for those batch users

`full_sort_predict()` then computes:

1. Full user/item propagated embeddings for the entire graph.
2. `u_embeddings = restore_user_e[user]` -> `[B_u, D]`
3. `scores = u_embeddings @ restore_item_e.T` -> `[B_u, I]`

The trainer masks train positives with `scores[masked_items[0], masked_items[1]] = -(1 << 10)` and runs `torch.topk` over the dense score matrix.

**Evidence**: `external/LayerGCN/utils/dataloader.py::EvalDataLoader._next_batch_data`, `EvalDataLoader._get_pos_items_per_u`, `external/LayerGCN/models/layergcn.py::LayerGCN.full_sort_predict`, `external/LayerGCN/models/common/trainer.py::Trainer.evaluate`.

---

## 5) Hardware Adaptation and Performance Profiling

### 5.1 What hardware adaptation exists

Hardware adaptation is minimal:

- `Config._init_device()` chooses CUDA vs CPU using `use_gpu` and `gpu_id`.
- `norm_adj_matrix` is moved to the chosen device during model construction.
- Train/eval batches are created directly on `config['device']`.

Mixed precision, distributed training, graph-kernel libraries, and explicit memory profilers are **NOT FOUND** in this repo path.

**Evidence**: `external/LayerGCN/utils/configurator.py::Config._init_device`, `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `external/LayerGCN/utils/dataloader.py::TrainDataLoader._get_neg_sample`, `EvalDataLoader.__init__`.

### 5.2 Main bottlenecks

| Bottleneck | Why it is expensive | Where it happens |
|---|---|---|
| Repeated full-graph evaluation | `Trainer.evaluate()` iterates user batches, but each batch calls `LayerGCN.full_sort_predict()`, which reruns the entire graph `forward()` before slicing users | `external/LayerGCN/models/common/trainer.py::Trainer.evaluate`, `external/LayerGCN/models/layergcn.py::LayerGCN.full_sort_predict` |
| Layer-history retention | The model stores every refined layer output in `embeddings_layers`, then allocates another stacked tensor with `torch.stack(...)` before summing | `external/LayerGCN/models/layergcn.py::LayerGCN.forward` |
| Dense full-sort score matrix | Each eval batch materializes `[B_u, I]` scores with `torch.matmul` | `external/LayerGCN/models/layergcn.py::LayerGCN.full_sort_predict` |
| Epoch-wise sparse graph rebuild | With dropout active, every epoch samples edges, renormalizes the retained graph, and rebuilds `masked_adj` | `external/LayerGCN/models/layergcn.py::LayerGCN.pre_epoch_processing` |
| Python negative sampling loop | Negative IDs are sampled in Python with rejection sampling per user | `external/LayerGCN/utils/dataloader.py::TrainDataLoader._sample_neg_ids` |
| Python evaluation membership loop | Top-k correctness is computed with Python `in` checks for every recommended item per user | `external/LayerGCN/utils/topk_evaluator.py::TopKEvaluator.evaluate` |

### 5.3 VRAM and memory implications

- The full-graph dense state is `[U + I, D]` per layer.
- `embeddings_layers` keeps `L` such tensors alive until pooling.
- `torch.stack(embeddings_layers, dim=0)` transiently creates an additional `[L, U + I, D]` tensor.
- Evaluation adds a dense `[B_u, I]` score matrix on top of the propagated embeddings.

For large `U + I`, this is the dominant memory pattern. There is no activation checkpointing, sparse user batching inside `forward()`, or cached propagated embedding reuse in evaluation.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.forward`, `LayerGCN.full_sort_predict`, `external/LayerGCN/models/common/trainer.py::Trainer.evaluate`.

### 5.4 Reproducibility and runtime notes

- The outer loop resets the random seed per hyper-parameter combination.
- Edge pruning behavior depends on both the seed and epoch parity because `self.pruning_random` toggles between weighted and random pruning every epoch.
- The runtime contains no explicit throughput or VRAM profiler. Profiling support is **NOT FOUND**.

**Evidence**: `external/LayerGCN/utils/quick_start.py::quick_start`, `external/LayerGCN/utils/utils.py::init_seed`, `external/LayerGCN/models/layergcn.py::LayerGCN.pre_epoch_processing`.

---

## 6) Minimal Viable Code for Replication

The snippets below keep the active repository mechanics and drop only framework glue.

### 6.1 `build_layergcn_graph`

```python
from __future__ import annotations

import torch


def _normalize_bipartite_edges(
    edge_index: torch.Tensor,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """Return LayerGCN's per-edge symmetric weights for a user-item graph."""
    users = edge_index[0]
    items = edge_index[1]
    user_deg = torch.bincount(users, minlength=num_users).float() + 1e-7
    item_deg = torch.bincount(items, minlength=num_items).float() + 1e-7
    return user_deg[users].pow(-0.5) * item_deg[items].pow(-0.5)


def build_layergcn_graph(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the full normalized square adjacency and the rectangular edge cache."""
    edge_index = torch.stack([user_ids.long(), item_ids.long()], dim=0)
    edge_weight = _normalize_bipartite_edges(edge_index, num_users, num_items)

    square_src = torch.cat([edge_index[0], edge_index[1] + num_users], dim=0)
    square_dst = torch.cat([edge_index[1] + num_users, edge_index[0]], dim=0)
    square_index = torch.stack([square_src, square_dst], dim=0)
    square_weight = torch.cat([edge_weight, edge_weight], dim=0)

    full_adj = torch.sparse_coo_tensor(
        square_index,
        square_weight,
        size=(num_users + num_items, num_users + num_items),
        device=device,
    ).coalesce()
    return full_adj, edge_index, edge_weight
```

Repository correspondence: `LayerGCN.get_norm_adj_mat()` plus `LayerGCN.get_edge_info()` and `LayerGCN._normalize_adj_m()`.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.get_norm_adj_mat`, `LayerGCN.get_edge_info`, `LayerGCN._normalize_adj_m`.

### 6.2 `pool_layergcn`

```python
from __future__ import annotations

import torch


def pool_layergcn(
    refined_layers: list[torch.Tensor],
    num_users: int,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum refined propagated layers and split back into user/item blocks."""
    ui_embeddings = torch.sum(torch.stack(refined_layers, dim=0), dim=0)
    return torch.split(ui_embeddings, [num_users, num_items], dim=0)
```

This is intentionally a sum, not a mean, and it assumes `refined_layers` excludes the raw layer-0 table.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.forward`.

### 6.3 `LayerGCNCore`

```python
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerGCNCore(nn.Module):
    """Minimal LayerGCN core: graph pruning, propagation, cosine refinement, pooling."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        n_layers: int,
        dropout: float,
        full_adj: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.user_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_users, embed_dim))
        )
        self.item_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_items, embed_dim))
        )

        self.full_adj = full_adj.coalesce().to(device)
        self.edge_index = edge_index.long()
        self.edge_weight = edge_weight.float()
        self.pruning_random = False
        self.forward_adj = self.full_adj

    def _renormalize_kept_edges(self, kept_edge_index: torch.Tensor) -> torch.Tensor:
        """Recompute symmetric rectangular weights on the retained user-item edges."""
        users = kept_edge_index[0]
        items = kept_edge_index[1]
        user_deg = torch.bincount(users, minlength=self.num_users).float() + 1e-7
        item_deg = torch.bincount(items, minlength=self.num_items).float() + 1e-7
        return user_deg[users].pow(-0.5) * item_deg[items].pow(-0.5)

    def compute_masked_adj(self) -> None:
        """Match the repo's alternating weighted/random edge pruning."""
        if self.dropout <= 0.0:
            self.forward_adj = self.full_adj
            return

        keep_len = int(self.edge_weight.numel() * (1.0 - self.dropout))
        if self.pruning_random:
            keep_idx = torch.tensor(
                random.sample(range(self.edge_weight.numel()), keep_len),
                dtype=torch.long,
            )
        else:
            keep_idx = torch.multinomial(self.edge_weight, keep_len, replacement=False)
        self.pruning_random = not self.pruning_random

        kept_edge_index = self.edge_index[:, keep_idx]
        kept_weight = self._renormalize_kept_edges(kept_edge_index)

        src = torch.cat([kept_edge_index[0], kept_edge_index[1] + self.num_users], dim=0)
        dst = torch.cat([kept_edge_index[1] + self.num_users, kept_edge_index[0]], dim=0)
        values = torch.cat([kept_weight, kept_weight], dim=0).to(self.device)
        indices = torch.stack([src, dst], dim=0).to(self.device)

        self.forward_adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=self.full_adj.shape,
            device=self.device,
        ).coalesce()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run sparse propagation, cosine refinement, and sum-only pooling."""
        ego = torch.cat([self.user_embeddings, self.item_embeddings], dim=0)
        hidden = ego
        refined_layers: list[torch.Tensor] = []

        for _ in range(self.n_layers):
            hidden = torch.sparse.mm(self.forward_adj, hidden)
            weight = F.cosine_similarity(hidden, ego, dim=-1)
            hidden = torch.einsum("n,nd->nd", weight, hidden)
            refined_layers.append(hidden)

        return pool_layergcn(refined_layers, self.num_users, self.num_items)
```

Repository correspondence: `LayerGCN.__init__()`, `LayerGCN.pre_epoch_processing()`, `LayerGCN.forward()`.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.__init__`, `LayerGCN.pre_epoch_processing`, `LayerGCN.forward`.

### 6.4 `loss_function`

```python
from __future__ import annotations

import torch


def loss_function(
    user_all: torch.Tensor,
    item_all: torch.Tensor,
    user_ego: torch.Tensor,
    item_ego: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    reg_weight: float,
) -> torch.Tensor:
    """Reproduce the active LayerGCN ranking loss plus raw-embedding L2 regularizer."""
    u = user_all[users]
    pos = item_all[pos_items]
    neg = item_all[neg_items]

    pos_scores = (u * pos).sum(dim=1)
    neg_scores = (u * neg).sum(dim=1)
    mf_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).sum()

    u0 = user_ego[users]
    pos0 = item_ego[pos_items]
    neg0 = item_ego[neg_items]
    reg_loss = 0.5 * (u0.square().sum() + pos0.square().sum() + neg0.square().sum())

    return mf_loss + reg_weight * reg_loss
```

This matches the repository's active loss semantics more closely than the unused `BPRLoss` helper because it keeps the sum-reduced ranking term and the raw-table L2 penalty.

**Evidence**: `external/LayerGCN/models/layergcn.py::LayerGCN.bpr_loss`, `LayerGCN.emb_loss`, `LayerGCN.calculate_loss`, `external/LayerGCN/models/common/loss.py::L2Loss.forward`.
