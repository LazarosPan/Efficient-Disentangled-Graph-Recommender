# LightGCN++ - Standalone Developer Blueprint (Code-First, Repository-Derived)

## Execution Map

The repository ships two LightGCN++ paths.

| Path | Entry script | Instantiated symbol | Object that actually owns parameters | Initialization style |
|---|---|---|---|---|
| Main repo path | `external/LightGCNpp/code/main.py` | `model.LightGCN` via `register.MODELS['lgn']` | `LightGCN(nn.Module)` | `nn.init.normal_(..., std=0.1)` |
| SELFRec path | `external/LightGCNpp/SELFRec/main.py` | `model.graph.LightGCNpp.LightGCNpp` | `LGCN_Encoder(nn.Module)` inside the wrapper | `nn.init.xavier_uniform_` |

For independent replication, the main repo path is the canonical one. The top-level README explicitly points to `external/LightGCNpp/code` as the implementation to run, and `main.py` wires the training job directly into `model.LightGCN`. The SELFRec folder re-expresses the same kernel behind a framework wrapper and changes some initialization details.

## Mechanical Summary

The shipped LightGCN++ delta is small and mechanical:

1. The bipartite adjacency is reweighted with configurable `alpha` and `beta` instead of the standard symmetric square-root normalization.
2. Node embeddings are L2-normalized inside every propagation step before sparse aggregation.
3. Final embeddings are not a plain mean over all layers. The code averages only propagated layers, then mixes that average with the raw embedding table using a fixed scalar `gamma`.

There is no PyG `MessagePassing`, no learned pooling gate, and no learned neighbor weight in the active LightGCN++ files.

---

## 1) Core Class Structure and Initialization

### 1.1 Canonical class in the main code path

The main runtime does not instantiate a class literally named `LightGCNpp`. It instantiates `LightGCN`, and that class contains the LightGCN++ mechanics.

```python
# external/LightGCNpp/code/register.py
MODELS = {
	'mf': model.PureMF,
	'lgn': model.LightGCN,
}

# external/LightGCNpp/code/main.py
Recmodel = register.MODELS[world.model_name](world.config, dataset)
```

The parameter-owning class is:

```python
class LightGCN(BasicModel):
	def __init__(self, config: dict, dataset: BasicDataset):
		super(LightGCN, self).__init__()
		self.config = config
		self.dataset: BasicDataset = dataset
		self.__init_weight()
```

### 1.2 SELFRec class structure

The SELFRec path uses a wrapper named `LightGCNpp`, but that wrapper is not an `nn.Module`. The learnable module lives in `LGCN_Encoder`.

```python
class LightGCNpp(GraphRecommender):
	def __init__(self, conf, training_set, valid_set, test_set):
		super(LightGCNpp, self).__init__(conf, training_set, valid_set, test_set)
		self.n_layers = conf.n_layer
		self.alpha = conf.alpha
		self.beta = conf.beta
		self.gamma = conf.gamma
		self.model = LGCN_Encoder(
			self.data,
			self.emb_size,
			self.n_layers,
			self.alpha,
			self.beta,
			self.gamma,
		)
```

### 1.3 Embedding initialization

#### Main code path

```python
def __init_weight(self):
	self.num_users = self.dataset.n_users
	self.num_items = self.dataset.m_items
	self.latent_dim = self.config['latent_dim_rec']
	self.n_layers = self.config['lightGCN_n_layers']
	self.keep_prob = self.config['keep_prob']
	self.A_split = self.config['A_split']
	self.gamma = self.config['gamma']
	self.embedding_user = torch.nn.Embedding(
		num_embeddings=self.num_users,
		embedding_dim=self.latent_dim,
	)
	self.embedding_item = torch.nn.Embedding(
		num_embeddings=self.num_items,
		embedding_dim=self.latent_dim,
	)

	if self.config['pretrain'] == 0:
		nn.init.normal_(self.embedding_user.weight, std=0.1)
		nn.init.normal_(self.embedding_item.weight, std=0.1)
	else:
		self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
		self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
```

Main-path initialization details:

- **User embedding matrix**: `nn.Embedding(num_users, latent_dim)`
- **Item embedding matrix**: `nn.Embedding(num_items, latent_dim)`
- **Default distribution**: normal with default mean `0.0` and `std=0.1`
- **Alternative path**: direct copy from pretrained NumPy arrays when `pretrain != 0`

#### SELFRec path

```python
def _init_model(self):
	initializer = nn.init.xavier_uniform_
	embedding_dict = nn.ParameterDict({
		'user_emb': nn.Parameter(
			initializer(torch.empty(self.data.user_num, self.latent_size))
		),
		'item_emb': nn.Parameter(
			initializer(torch.empty(self.data.item_num, self.latent_size))
		),
	})
	return embedding_dict
```

SELFRec-path initialization details:

- **User embedding matrix**: `embedding_dict['user_emb']`
- **Item embedding matrix**: `embedding_dict['item_emb']`
- **Distribution**: Xavier uniform

### 1.4 Learnable parameters vs fixed state

#### Main code path: learnable

- **`embedding_user.weight`**
- **`embedding_item.weight`**

#### Main code path: fixed runtime state

- **`self.Graph`**: prebuilt sparse adjacency tensor
- **`self.gamma`**: plain config scalar, not an `nn.Parameter`
- **`self.keep_prob`** and **`self.A_split`**: config values
- **`self.f`**: `nn.Sigmoid()` module with no learnable weights

#### SELFRec path: learnable

- **`embedding_dict['user_emb']`**
- **`embedding_dict['item_emb']`**

#### SELFRec path: fixed runtime state

- **`self.alpha`**, **`self.beta`**, **`self.gamma`**: wrapper scalars copied from CLI args
- **`self.norm_adj`**: SciPy sparse matrix
- **`self.sparse_norm_adj`**: torch sparse tensor placed on CUDA at construction time

### 1.5 Are scaling factors learnable?

No.

- **`alpha`** and **`beta`** are fixed hyperparameters used only while building the normalized adjacency.
- **`gamma`** is a fixed scalar used in the final residual mix.
- There is no learned layer weight vector and no learned edge coefficient in the active LightGCN++ files.

Repository footnote: notebook checkpoint files inside `.ipynb_checkpoints` contain experimental variants where `alpha`, `beta`, and `gamma` become `nn.Parameter`, but the shipped launchers do not import those files.

---

## 2) The Propagation Kernel

### 2.1 Where neighbor weighting is actually implemented

The repo does not compute edge weights inside the forward pass. It precomputes a weighted sparse adjacency and then reuses it in `torch.sparse.mm`.

#### Main code path

```python
rowsum_left = np.array(adj_mat.sum(axis=1)) ** -self.alpha
rowsum_right = np.array(adj_mat.sum(axis=1)) ** -self.beta

d_inv_left = rowsum_left.flatten()
d_inv_left[np.isinf(d_inv_left)] = 0.

d_inv_right = rowsum_right.flatten()
d_inv_right[np.isinf(d_inv_right)] = 0.

d_mat_left = sp.diags(d_inv_left)
d_mat_right = sp.diags(d_inv_right)

norm_adj = d_mat_left.dot(adj_mat)
norm_adj = norm_adj.dot(d_mat_right)
norm_adj = norm_adj.tocsr()
```

This is called inside `Loader.getSparseGraph()` after building the bipartite adjacency.

#### SELFRec path

```python
@staticmethod
def normalize_graph_mat(adj_mat, alpha=0.5, beta=0.5):
	rowsum = np.array(adj_mat.sum(1))
	d_inv_left = np.power(rowsum, -alpha).flatten()
	d_inv_left[np.isinf(d_inv_left)] = 0.
	d_mat_inv_left = sp.diags(d_inv_left)

	d_inv_right = np.power(rowsum, -beta).flatten()
	d_inv_right[np.isinf(d_inv_right)] = 0.
	d_mat_inv_right = sp.diags(d_inv_right)

	norm_adj_tmp = d_mat_inv_left.dot(adj_mat)
	norm_adj_mat = norm_adj_tmp.dot(d_mat_inv_right)
	return norm_adj_mat
```

### 2.2 What each edge weight becomes

Each user-item edge starts with value `1.0` in the bipartite adjacency. After normalization, its stored value is scaled by:

- the source node degree raised to `-alpha`
- the destination node degree raised to `-beta`

That scaling is static for the whole training run unless the graph is rebuilt.

### 2.3 Where embedding norm scaling happens

The extra embedding normalization happens inside the propagation loop, before each sparse matrix multiply.

```python
for layer in range(self.n_layers):
	norm = torch.norm(all_emb, dim=1) + 1e-12
	all_emb = all_emb / norm[:, None]
	all_emb = torch.sparse.mm(g_droped, all_emb)
	embs.append(all_emb)
```

SELFRec uses the same sequence:

```python
for k in range(self.layers):
	norm = torch.norm(ego_embeddings, dim=1) + 1e-12
	ego_embeddings = ego_embeddings / norm[:, None]
	ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
	all_embeddings += [ego_embeddings]
```

There is no post-processing normalization block after all layers finish. The normalization is part of every propagation step.

### 2.4 Message-passing operator sequence

The active kernel is raw sparse matrix multiplication, not a graph library message-passing API.

Per layer, the main code path does this:

1. Read full user embedding table.
2. Read full item embedding table.
3. Concatenate into one dense matrix `all_emb`.
4. L2-normalize each row of `all_emb`.
5. Multiply by the preweighted sparse adjacency using `torch.sparse.mm`.
6. Append the result to a Python list for later pooling.

Optional branch in the main code path:

```python
if self.A_split:
	temp_emb = []
	for f in range(len(g_droped)):
		temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
	side_emb = torch.cat(temp_emb, dim=0)
	all_emb = side_emb
else:
	all_emb = torch.sparse.mm(g_droped, all_emb)
```

Implications:

- No `edge_index`
- No `scatter_add`
- No PyG `MessagePassing`
- No per-edge function call inside the forward loop

---

## 3) Layer-Wise Pooling and Final Representation

### 3.1 Exact LightGCN++ pooling rule

The repo uses a two-stage pooling rule:

1. Average only the propagated states.
2. Mix that average with the original embedding table using fixed scalar `gamma`.

Main code path:

```python
embs_zero = embs[0]
embs_prop = torch.mean(torch.stack(embs[1:], dim=1), dim=1)

light_out = (self.gamma * embs_zero) + ((1 - self.gamma) * embs_prop)

_users, _items = torch.split(torch.stack(embs, dim=1), [self.num_users, self.num_items])
users, items = torch.split(light_out, [self.num_users, self.num_items])

return users, items, _users, _items
```

SELFRec path:

```python
embs_zero = all_embeddings[0]
embs_prop = torch.mean(torch.stack(all_embeddings[1:], dim=1), dim=1)

light_out = (self.gamma * embs_zero) + ((1 - self.gamma) * embs_prop)

user_all_embeddings = light_out[:self.data.user_num]
item_all_embeddings = light_out[self.data.user_num:]
return user_all_embeddings, item_all_embeddings
```

### 3.2 Weighting logic

- Propagated layers are combined with a uniform average.
- The raw embedding table receives fixed weight `gamma`.
- The propagated average receives fixed weight `1 - gamma`.
- There is no learned gate, no attention score, and no layer-index-dependent decay.

### 3.3 Is there separate consistency logic?

Not as a standalone module.

The code has no extra consistency loss and no explicit cross-layer constraint block inside LightGCN++. The practical consistency mechanism is just this combination of:

- per-layer row normalization before aggregation
- retaining every propagated layer output
- uniform averaging over the propagated stack
- residual mixing with the initial embedding table

### 3.4 From pooled tensor to ranking score

#### Training path

```python
all_users, all_items, _, _ = self.computer()
users_emb = all_users[users]
pos_emb = all_items[pos_items]
neg_emb = all_items[neg_items]

pos_scores = torch.sum(users_emb * pos_emb, dim=1)
neg_scores = torch.sum(users_emb * neg_emb, dim=1)
loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
```

#### Evaluation path

```python
rating = self.f(torch.matmul(users_emb, items_emb.t()))
```

So the final pooled user matrix and final pooled item matrix are the only tensors used for both BPR training and full-catalog ranking.

---

## 4) Data Flow and Tensor Shapes

### 4.1 Important correction: there is no active `edge_index`

The repo does not operate on a PyG-style `edge_index`. It stores the graph directly as a sparse adjacency tensor.

The closest equivalent to `edge_index` is the COO index tensor used when building the sparse adjacency:

```python
index = torch.stack([row, col])
graph = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
```

That `index` tensor has shape `(2, nnz)`.

### 4.2 Graph build trace

Assume:

- **`U`** = number of users
- **`I`** = number of items
- **`N`** = `U + I`
- **`R`** = number of observed training interactions
- **`D`** = embedding width
- **`L`** = number of propagation layers
- **`B`** = sampled BPR batch size

Graph construction shapes:

| Tensor or structure | Shape | Notes |
|---|---|---|
| `trainUser` | `(R,)` | user ids from train file |
| `trainItem` | `(R,)` | item ids from train file |
| `UserItemNet` | `(U, I)` | SciPy CSR interaction matrix |
| bipartite adjacency `adj_mat` | `(N, N)` | user block plus item block |
| COO index before torch sparse conversion | `(2, 2 * R)` in the simple unweighted case | both directions are inserted |
| normalized sparse graph `Graph` | `(N, N)` | stored as sparse tensor on device |

### 4.3 Single forward trace in the main code path

#### Inputs

| Input | Shape |
|---|---|
| `batch_users` | `(B,)` |
| `batch_pos` | `(B,)` |
| `batch_neg` | `(B,)` |
| `self.Graph` | sparse tensor of size `(N, N)` |

#### Parameter tensors

| Tensor | Shape |
|---|---|
| `embedding_user.weight` | `(U, D)` |
| `embedding_item.weight` | `(I, D)` |

#### Propagation trace

| Step | Tensor | Shape |
|---|---|---|
| concatenate user and item tables | `all_emb` | `(N, D)` |
| row norms | `norm` | `(N,)` |
| normalized embedding matrix | `all_emb / norm[:, None]` | `(N, D)` |
| propagated layer output | `torch.sparse.mm(Graph, all_emb)` | `(N, D)` |
| stored history list length | `embs` | `L + 1` tensors, each `(N, D)` |
| stacked propagated history | `torch.stack(embs[1:], dim=1)` | `(N, L, D)` |
| propagated average | `embs_prop` | `(N, D)` |
| final mixed output | `light_out` | `(N, D)` |
| split user output | `users` | `(U, D)` |
| split item output | `items` | `(I, D)` |

#### BPR tensors

| Tensor | Shape |
|---|---|
| `users_emb = all_users[batch_users]` | `(B, D)` |
| `pos_emb = all_items[batch_pos]` | `(B, D)` |
| `neg_emb = all_items[batch_neg]` | `(B, D)` |
| `pos_scores` | `(B,)` |
| `neg_scores` | `(B,)` |

#### Evaluation tensor

| Tensor | Shape |
|---|---|
| `rating = sigmoid(users_emb @ items_emb.T)` | `(B_eval, I)` |

### 4.4 SELFRec forward trace

The SELFRec path uses the same core shape evolution with two naming differences:

- the dense concatenated matrix is named **`ego_embeddings`**
- the parameter store is **`embedding_dict`** instead of `nn.Embedding`

Return value:

```python
return user_all_embeddings, item_all_embeddings
```

So the SELFRec wrapper exposes only the final pooled user and item matrices, not the per-layer stack.

---

## 5) Hardware Adaptation and Performance Profiling

### 5.1 CUDA path that matters for a Linux RTX 5080

For a Linux machine with an NVIDIA GPU, the relevant runtime is:

- `external/LightGCNpp/code/main.py`
- `external/LightGCNpp/code/model.py::LightGCN`

The Gaudi path in `main_gaudi.py` and `LightGCNGaudi` is not the correct path for CUDA hardware.

### 5.2 Memory management behavior

There is no explicit memory cleanup inside propagation.

- No `detach()` between layers
- No `torch.cuda.empty_cache()`
- No activation checkpointing
- No manual gradient checkpointing

The only visible `detach()` calls are around checkpoint/export logic, for example when saving best embeddings:

```python
all_users, all_items = all_users.detach().cpu(), all_items.detach().cpu()
```

Those do not reduce training-time activation memory inside the forward pass.

### 5.3 Main VRAM and runtime bottlenecks

#### Bottleneck 1: full layer history is retained

Both implementations keep every layer output in a Python list because final pooling depends on all propagated states.

```python
embs = [all_emb]
...
embs.append(all_emb)
```

That means backward has to retain all sparse-matmul outputs for all `L` layers.

#### Bottleneck 2: the main code path stacks history twice

In `code/model.py::computer`, the history is materialized twice:

```python
embs_prop = torch.mean(torch.stack(embs[1:], dim=1), dim=1)
_users, _items = torch.split(torch.stack(embs, dim=1), [self.num_users, self.num_items])
```

Implication:

- first stack allocates `(N, L, D)`
- second stack allocates `(N, L + 1, D)`

The second stack is created even when the caller does not need per-layer outputs.

#### Bottleneck 3: repeated full-graph propagation during evaluation in the main code path

`Procedure.Test` and `Procedure.Valid` call:

```python
rating = Recmodel.getUsersRating(batch_users_gpu)
```

and `getUsersRating` does this when embeddings are not passed in:

```python
if all_users is None or all_items is None:
	all_users, all_items, _, _ = self.computer()
```

So the main code path recomputes the entire graph propagation once per evaluation user batch. That is a large avoidable runtime cost.

The SELFRec path is better here because it computes `self.user_emb` and `self.item_emb` once per epoch, then uses those cached tensors during ranking.

#### Bottleneck 4: Python-side sampling and masking

CPU-side loops remain in several hot paths:

- `UniformSample_original_python()`
- `next_batch_pairwise()`
- exclusion-index construction in `Procedure.Test` and `Procedure.Valid`
- fold loop when `A_split=True`

The main repo can optionally load a C++ negative sampler through `cppimport`, but the fallback path is pure Python.

### 5.4 What is already vectorized well

The heavy numerical parts are already vectorized:

- user/item table concatenation
- row-wise L2 normalization
- sparse aggregation with `torch.sparse.mm`
- batched dot products for BPR
- full-catalog matrix multiply for ranking

### 5.5 Dense graph warning

The Gaudi-only class converts the full sparse graph to dense memory:

```python
self.Graph_dense = self.Graph.to_dense().to(world_gaudi.device)
```

That is a major memory expansion and not appropriate for the CUDA path on larger datasets.

### 5.6 Practical replication note for RTX 5080

To match the repository while staying on the CUDA-friendly path:

1. Use sparse adjacency tensors.
2. Do not densify the graph.
3. Expect activation memory to scale with the number of stored layer outputs.
4. If reproducing the main code path exactly, keep the duplicated `torch.stack` behavior and the per-batch evaluation recomputation, even though both are inefficient.

---

## 6) Minimal Viable Code for Replication

The snippets below reproduce the actual repository mechanics, using the main `code/` path as the baseline. They are cleaned up for readability but preserve the tensor flow and hyperparameter usage.

### 6.1 Custom aggregator: adjacency builder with `alpha` and `beta`

```python
import numpy as np
import scipy.sparse as sp
import torch


def build_lightgcnpp_graph(
	user_item_csr: sp.csr_matrix,
	alpha: float,
	beta: float,
	device: torch.device,
) -> torch.Tensor:
	num_users, num_items = user_item_csr.shape
	num_nodes = num_users + num_items

	adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
	adj = adj.tolil()
	user_item = user_item_csr.tolil()
	adj[:num_users, num_users:] = user_item
	adj[num_users:, :num_users] = user_item.T
	adj = adj.todok()

	rowsum_left = np.array(adj.sum(axis=1)) ** -alpha
	rowsum_right = np.array(adj.sum(axis=1)) ** -beta

	d_inv_left = rowsum_left.flatten()
	d_inv_left[np.isinf(d_inv_left)] = 0.0

	d_inv_right = rowsum_right.flatten()
	d_inv_right[np.isinf(d_inv_right)] = 0.0

	norm_adj = sp.diags(d_inv_left).dot(adj).dot(sp.diags(d_inv_right)).tocsr()
	coo = norm_adj.tocoo().astype(np.float32)

	index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
	value = torch.tensor(coo.data, dtype=torch.float32, device=device)
	graph = torch.sparse_coo_tensor(index, value, size=coo.shape, device=device)
	return graph.coalesce()
```

### 6.2 Pooling module: propagated mean plus `gamma` residual

```python
import torch


def pool_lightgcnpp(layer_outputs: list[torch.Tensor], gamma: float) -> torch.Tensor:
	base_embedding = layer_outputs[0]
	propagated_average = torch.mean(torch.stack(layer_outputs[1:], dim=1), dim=1)
	return (gamma * base_embedding) + ((1.0 - gamma) * propagated_average)
```

### 6.3 Forward core: from user/item parameters to pooled embeddings

```python
import torch
from torch import nn


class LightGCNppCore(nn.Module):
	def __init__(
		self,
		num_users: int,
		num_items: int,
		latent_dim: int,
		num_layers: int,
		gamma: float,
		graph: torch.Tensor,
	):
		super().__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.num_layers = num_layers
		self.gamma = gamma
		self.graph = graph

		self.embedding_user = nn.Embedding(num_users, latent_dim)
		self.embedding_item = nn.Embedding(num_items, latent_dim)

		nn.init.normal_(self.embedding_user.weight, std=0.1)
		nn.init.normal_(self.embedding_item.weight, std=0.1)

	def computer(self) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
		all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
		layer_outputs = [all_emb]

		for _ in range(self.num_layers):
			norm = torch.norm(all_emb, dim=1) + 1e-12
			all_emb = all_emb / norm[:, None]
			all_emb = torch.sparse.mm(self.graph, all_emb)
			layer_outputs.append(all_emb)

		light_out = pool_lightgcnpp(layer_outputs, self.gamma)
		users, items = torch.split(light_out, [self.num_users, self.num_items], dim=0)
		return users, items, layer_outputs

	def get_embedding_triplet(
		self,
		users: torch.Tensor,
		pos_items: torch.Tensor,
		neg_items: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		all_users, all_items, _ = self.computer()
		return (
			all_users[users],
			all_items[pos_items],
			all_items[neg_items],
			self.embedding_user(users),
			self.embedding_item(pos_items),
			self.embedding_item(neg_items),
		)

	def get_users_rating(
		self,
		users: torch.Tensor,
		all_users: torch.Tensor | None = None,
		all_items: torch.Tensor | None = None,
	) -> torch.Tensor:
		if all_users is None or all_items is None:
			all_users, all_items, _ = self.computer()
		users_emb = all_users[users.long()]
		return torch.sigmoid(users_emb @ all_items.t())

	def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
		all_users, all_items, _ = self.computer()
		users_emb = all_users[users]
		items_emb = all_items[items]
		return torch.sum(users_emb * items_emb, dim=1)
```

### 6.4 BPR head exactly as used by the main code path

```python
import torch
import torch.nn.functional as F


def bpr_loss_with_repo_regularization(
	model: LightGCNppCore,
	users: torch.Tensor,
	pos_items: torch.Tensor,
	neg_items: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	(
		users_emb,
		pos_emb,
		neg_emb,
		user0,
		pos0,
		neg0,
	) = model.get_embedding_triplet(users.long(), pos_items.long(), neg_items.long())

	pos_scores = torch.sum(users_emb * pos_emb, dim=1)
	neg_scores = torch.sum(users_emb * neg_emb, dim=1)
	ranking_loss = torch.mean(F.softplus(neg_scores - pos_scores))

	reg_loss = 0.5 * (
		user0.norm(2).pow(2)
		+ pos0.norm(2).pow(2)
		+ neg0.norm(2).pow(2)
	) / float(len(users))

	return ranking_loss, reg_loss
```

### 6.5 SELFRec replication note

If exact SELFRec behavior is required instead of the main code path:

- replace `nn.Embedding` tables with a `nn.ParameterDict`
- initialize with `nn.init.xavier_uniform_`
- build `norm_adj` inside the encoder from `data.ui_adj`
- return only final user and item matrices from the encoder

The propagation kernel and pooling rule stay the same.

---

## Replication Checklist

To reproduce repository behavior faithfully:

1. Build a full bipartite adjacency over users plus items.
2. Left- and right-scale that adjacency with degree powers controlled by `alpha` and `beta`.
3. Concatenate user and item embedding tables before propagation.
4. Normalize every node embedding row before each sparse multiply.
5. Store every layer output.
6. Average only the propagated layers.
7. Mix the propagated average with the raw embedding table using fixed scalar `gamma`.
8. Split the mixed output back into user and item blocks.
9. Train with dot-product BPR and L2 regularization on the raw lookup embeddings.
10. For the main code path, keep in mind that evaluation recomputes the full graph embeddings per user batch unless you explicitly cache them.
