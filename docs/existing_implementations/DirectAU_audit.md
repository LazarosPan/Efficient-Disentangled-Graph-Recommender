# DirectAU Developer Blueprint

## Scope

This audit only uses the DirectAU-specific execution path:

- `external/DirectAU/recbole/model/general_recommender/directau.py`
- `external/DirectAU/recbole/model/init.py`
- `external/DirectAU/recbole/properties/model/DirectAU.yaml`
- `external/DirectAU/recbole/properties/overall.yaml`
- `external/DirectAU/recbole/config/configurator.py`
- `external/DirectAU/recbole/model/abstract_recommender.py`
- `external/DirectAU/recbole/trainer/trainer.py`

Everything else in the repository is ignored unless it directly affects this call path.

## Default Configuration Snapshot

From `recbole/properties/model/DirectAU.yaml` and `recbole/properties/overall.yaml`:

- **embedding_size**: `64`
- **encoder**: `MF`
- **gamma**: `1`
- **training_neg_sample_num**: `0`
- **train_batch_size**: `256`
- **weight_decay**: `1e-6`
- **eval_setting**: `RO_RS,full`
- **eval_batch_size**: `4096`

This means the default training path uses positive `(user, item)` pairs only, and the default evaluation path is full-sort ranking over all items.

## 1. Embedding Initialization And Hypersphere Mapping

### **DirectAU** encoder selection

```python
if self.encoder_name == 'MF':
	self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
elif self.encoder_name == 'LightGCN':
	self.n_layers = config['n_layers']
	self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
	self.norm_adj = self.get_norm_adj_mat().to(self.device)
	self.encoder = LGCNEncoder(
		self.n_users,
		self.n_items,
		self.embedding_size,
		self.norm_adj,
		self.n_layers,
	)
else:
	raise ValueError('Non-implemented Encoder.')
```

### **MFEncoder** allocation

```python
class MFEncoder(nn.Module):
	def __init__(self, user_num, item_num, emb_size):
		super(MFEncoder, self).__init__()
		self.user_embedding = nn.Embedding(user_num, emb_size)
		self.item_embedding = nn.Embedding(item_num, emb_size)
```

### **LGCNEncoder** allocation

```python
class LGCNEncoder(nn.Module):
	def __init__(self, user_num, item_num, emb_size, norm_adj, n_layers=3):
		super(LGCNEncoder, self).__init__()
		self.n_users = user_num
		self.n_items = item_num
		self.n_layers = n_layers
		self.norm_adj = norm_adj

		self.user_embedding = torch.nn.Embedding(user_num, emb_size)
		self.item_embedding = torch.nn.Embedding(item_num, emb_size)
```

### Initialization logic

The encoder constructors only allocate embedding tables. Actual initialization happens once in the top-level **DirectAU** constructor:

```python
self.apply(xavier_normal_initialization)
```

The helper is:

```python
def xavier_normal_initialization(module):
	if isinstance(module, nn.Embedding):
		xavier_normal_(module.weight.data)
	elif isinstance(module, nn.Linear):
		xavier_normal_(module.weight.data)
		if module.bias is not None:
			constant_(module.bias.data, 0)
```

Mechanical facts:

- **user_embedding.weight** and **item_embedding.weight** are initialized with `torch.nn.init.xavier_normal_`.
- There is no custom variance, gain, standard deviation, scale multiplier, or manual seed inside the encoder constructors.
- **DirectAU** uses Xavier normal initialization, while the standalone `LightGCN` baseline in this repository uses a different initializer.

### Hypersphere projection

The actual repository code normalizes inside **DirectAU.forward**, not inside the loss functions:

```python
def forward(self, user, item):
	user_e, item_e = self.encoder(user, item)
	return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
```

Mechanical facts:

- `F.normalize(..., dim=-1)` is the only hypersphere mapping in the training objective.
- The normalization runs on every call to **DirectAU.forward**.
- **alignment** and **uniformity** consume already-normalized tensors.
- **predict** and **full_sort_predict** do not normalize embeddings before scoring.

## 2. The DirectAU Loss Kernel

### **alignment** implementation

```python
@staticmethod
def alignment(x, y, alpha=2):
	return (x - y).norm(p=2, dim=1).pow(alpha).mean()
```

Mechanical meaning:

- **x** and **y** are batch-aligned user and positive-item embeddings.
- The code subtracts them row by row.
- It computes row-wise Euclidean norm along `dim=1`.
- It raises each distance to `alpha`.
- With the default `alpha=2`, the term is squared Euclidean distance averaged across the batch.

### **uniformity** implementation

```python
@staticmethod
def uniformity(x, t=2):
	return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
```

Mechanical meaning:

- `torch.pdist(x, p=2)` computes the condensed list of all unique pairwise Euclidean distances between rows of **x**.
- The output is not a full square matrix. It is a flat vector containing one entry per unique pair.
- The code squares the distances, multiplies by `-t`, applies `exp`, averages all values, then applies `log`.

Mechanical facts:

- There is no sample-based approximation inside **uniformity**.
- There is no deduplication of repeated users or repeated items within a batch.
- There is no chunking, `torch.cdist`, manual broadcasting, clamp, epsilon, or `logsumexp` stabilization.
- With batch size `1`, `torch.pdist` returns an empty tensor and this function has no guard for that case.

### Loss assembly

```python
def calculate_loss(self, interaction):
	if self.restore_user_e is not None or self.restore_item_e is not None:
		self.restore_user_e, self.restore_item_e = None, None

	user = interaction[self.USER_ID]
	item = interaction[self.ITEM_ID]

	user_e, item_e = self.forward(user, item)
	align = self.alignment(user_e, item_e)
	uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

	return align + uniform
```

Mechanical facts:

- The loss uses only **user** and **item** tensors.
- **NEG_ITEM_ID** is defined by the RecBole base class, but **DirectAU.calculate_loss** never reads it.
- `training_neg_sample_num: 0` is converted by the config layer into `train_neg_sample_args = {'strategy': 'none'}`.
- User-side uniformity and item-side uniformity are computed separately and then averaged.
- **gamma** is the only configurable loss weight.
- `alpha=2` in **alignment** and `t=2` in **uniformity** are hardcoded function defaults, not YAML-configured hyperparameters.
- The model does not import or call **BPRLoss** from `recbole/model/loss.py`.

## 3. Backbone Integration

### **MFEncoder** forward path

```python
def forward(self, user_id, item_id):
	u_embed = self.user_embedding(user_id)
	i_embed = self.item_embedding(item_id)
	return u_embed, i_embed
```

This is plain matrix factorization lookup. No graph propagation, no MLP, no extra projector.

### **LGCNEncoder** forward path

```python
def get_ego_embeddings(self):
	user_embeddings = self.user_embedding.weight
	item_embeddings = self.item_embedding.weight
	ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
	return ego_embeddings

def get_all_embeddings(self):
	all_embeddings = self.get_ego_embeddings()
	embeddings_list = [all_embeddings]

	for layer_idx in range(self.n_layers):
		all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
		embeddings_list.append(all_embeddings)

	lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
	lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

	user_all_embeddings, item_all_embeddings = torch.split(
		lightgcn_all_embeddings,
		[self.n_users, self.n_items],
	)
	return user_all_embeddings, item_all_embeddings

def forward(self, user_id, item_id):
	user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
	u_embed = user_all_embeddings[user_id]
	i_embed = item_all_embeddings[item_id]
	return u_embed, i_embed
```

Mechanical facts:

- **LGCNEncoder** builds one full `user + item` embedding table.
- It propagates through `torch.sparse.mm(self.norm_adj, all_embeddings)` for each graph layer.
- It stacks all layer outputs and averages them across layers.
- It returns gathered user/item rows only after the full graph pass has completed.

### Inference scoring path

Default evaluation is full-sort because `eval_setting` contains `full`, and **Trainer._full_sort_batch_eval** calls **model.full_sort_predict** first.

The DirectAU full-sort scorer is:

```python
def full_sort_predict(self, interaction):
	user = interaction[self.USER_ID]
	if self.encoder_name == 'LightGCN':
		if self.restore_user_e is None or self.restore_item_e is None:
			self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
		user_e = self.restore_user_e[user]
		all_item_e = self.restore_item_e
	else:
		user_e = self.encoder.user_embedding(user)
		all_item_e = self.encoder.item_embedding.weight
	score = torch.matmul(user_e, all_item_e.transpose(0, 1))
	return score.view(-1)
```

Mechanical facts:

- Final ranking scores are raw dot products.
- The inference path does not apply `F.normalize` before `torch.matmul`.
- For **LightGCN**, full-graph embeddings are cached in **restore_user_e** and **restore_item_e**.
- For **MF**, the scorer directly uses the raw embedding tables.

### Important caveat in **predict**

The pointwise scorer in `directau.py` is:

```python
def predict(self, interaction):
	user = interaction[self.USER_ID]
	item = interaction[self.ITEM_ID]
	user_e = self.user_embedding(user)
	item_e = self.item_embedding(item)
	return torch.mul(user_e, item_e).sum(dim=1)
```

Mechanical issue:

- **DirectAU** does not define **self.user_embedding** or **self.item_embedding** on the top-level module.
- Those attributes live inside **self.encoder**.
- In the repository's default `RO_RS,full` evaluation setup, **Trainer** uses **full_sort_predict**, so this mismatch is usually bypassed.

## 4. Data Flow And Tensor Shapes

Use these names for the trace below:

- **B**: batch size
- **D**: embedding size
- **P**: `B * (B - 1) // 2`
- **U**: number of users
- **I**: number of items
- **L**: `n_layers + 1`

### Training-step trace

| Stage | Tensor | Shape | Mechanical note |
| --- | --- | --- | --- |
| Batch input | **user_ids** | `[B]` | Long tensor from `interaction[self.USER_ID]` |
| Batch input | **pos_item_ids** | `[B]` | Long tensor from `interaction[self.ITEM_ID]` |
| Batch input | **neg_item_ids** | not used | Negative sampling is disabled for DirectAU |
| MF lookup | **u_raw** | `[B, D]` | `self.user_embedding(user_id)` |
| MF lookup | **i_raw** | `[B, D]` | `self.item_embedding(item_id)` |
| Normalization | **user_e** | `[B, D]` | `F.normalize(u_raw, dim=-1)` |
| Normalization | **item_e** | `[B, D]` | `F.normalize(i_raw, dim=-1)` |
| Alignment diff | **delta** | `[B, D]` | `user_e - item_e` |
| Alignment norm | **pair_dist** | `[B]` | `.norm(p=2, dim=1)` |
| Alignment output | **align** | `[]` | `.pow(2).mean()` |
| User uniformity pairs | **user_pair_dist** | `[P]` | `torch.pdist(user_e, p=2)` |
| User uniformity output | **user_uni** | `[]` | `.pow(2).mul(-2).exp().mean().log()` |
| Item uniformity pairs | **item_pair_dist** | `[P]` | `torch.pdist(item_e, p=2)` |
| Item uniformity output | **item_uni** | `[]` | `.pow(2).mul(-2).exp().mean().log()` |
| Combined uniformity | **uniform** | `[]` | `gamma * (user_uni + item_uni) / 2` |
| Final loss | **loss** | `[]` | `align + uniform` |

### Extra tensors in the **LGCNEncoder** path

| Stage | Tensor | Shape | Mechanical note |
| --- | --- | --- | --- |
| Concatenated embeddings | **ego_embeddings** | `[U + I, D]` | User table concatenated with item table |
| Per-layer stack | **lightgcn_all_embeddings** before mean | `[U + I, L, D]` | Built by `torch.stack(embeddings_list, dim=1)` |
| Layer-averaged table | **lightgcn_all_embeddings** after mean | `[U + I, D]` | Mean over layer axis |
| Split users | **user_all_embeddings** | `[U, D]` | First block after split |
| Split items | **item_all_embeddings** | `[I, D]` | Second block after split |
| Gathered batch users | **u_embed** | `[B, D]` | `user_all_embeddings[user_id]` |
| Gathered batch items | **i_embed** | `[B, D]` | `item_all_embeddings[item_id]` |

### Full-sort evaluation trace

| Stage | Tensor | Shape | Mechanical note |
| --- | --- | --- | --- |
| Eval input | **user_ids** | `[B_eval]` | The batch contains users only |
| Full-sort score matrix | **score** | `[B_eval, I]` | `torch.matmul(user_e, all_item_e.transpose(0, 1))` |
| Returned value | **score.view(-1)** | `[B_eval * I]` | Trainer reshapes it back to `[B_eval, I]` |

## 5. Hardware Adaptation And Performance Profiling

### Pairwise complexity

The expensive part is **uniformity**:

```python
torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
```

Mechanical implications:

- Compute cost scales quadratically with batch size because every unique pair in the batch is visited.
- Memory also scales quadratically because `torch.pdist` materializes the condensed pair list.
- The implementation is more memory-efficient than building a full `[B, B]` matrix, but it is still quadratic.

### Condensed pair counts and rough float32 memory

| Batch size | Pair count from `torch.pdist` | One float32 output vector |
| --- | --- | --- |
| `256` | `32640` | about `0.12 MB` |
| `1024` | `523776` | about `2.00 MB` |
| `4096` | `8386560` | about `32.00 MB` |
| `8192` | `33550336` | about `128.00 MB` |

Practical note:

- The loss computes this once for users and once for items.
- The `.pow()`, `.mul()`, and `.exp()` chain also creates additional same-shape intermediates for autograd.
- For MF, the default batch sizes in this repo are small enough that the loss kernel itself is unlikely to stress a 16 GB GPU.
- For large-batch **LightGCN**, the encoder's full-graph propagation tensors can dominate memory before the loss kernel does.

### Vectorization choices

The code uses:

- `torch.pdist` for pairwise distances in **uniformity**
- `torch.sparse.mm` for graph propagation in **LGCNEncoder**
- `torch.matmul` for full-sort user-item scoring

The code does not use:

- `torch.cdist`
- manual broadcasted pairwise distance matrices
- chunked uniformity passes
- sampled pair subsets inside **uniformity**

### Replication-critical quirks

- The README shows normalization inside the loss helpers, but the actual model file normalizes in **DirectAU.forward**.
- Training uses normalized embeddings, while full-sort inference uses raw dot products.
- **uniformity** is batch-row based. If the same user or item appears multiple times in one batch, duplicates are kept.
- There is no guard against a batch with fewer than two rows.

## 6. Minimal Viable Code For Replication

The following snippet reproduces the repository's actual loss mechanics.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def alignment(x, y, alpha=2):
	return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x, t=2):
	return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class DirectAULoss(nn.Module):
	def __init__(self, encoder, gamma):
		super().__init__()
		self.encoder = encoder
		self.gamma = gamma

	def forward(self, user_ids, item_ids):
		user_e, item_e = self.encoder(user_ids, item_ids)
		user_e = F.normalize(user_e, dim=-1)
		item_e = F.normalize(item_e, dim=-1)

		align = alignment(user_e, item_e)
		uniform = self.gamma * (uniformity(user_e) + uniformity(item_e)) / 2
		return align + uniform
```

If exact parity matters, keep these details unchanged:

- Normalize after the encoder returns batch embeddings.
- Use `torch.pdist`, not a full pairwise matrix.
- Compute user and item uniformity separately.
- Average the two uniformity scalars, then multiply by **gamma**.
- Use raw dot-product scoring for full-sort inference.

### Minimal raw-dot-product inference block

```python
def full_sort_scores(user_ids, user_all_embeddings, item_all_embeddings):
	return torch.matmul(user_all_embeddings[user_ids], item_all_embeddings.t())
```

This matches the repository's evaluation logic more closely than a cosine-similarity scorer.
