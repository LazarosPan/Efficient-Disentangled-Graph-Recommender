# MGCE Technical Audit — Standalone Code-First Developer Blueprint

## Mechanical Summary Table

| Axis | Specification (mechanical) | Evidence |
|---|---|---|
| Framework | TensorFlow 1.x graph mode (`tf.placeholder`, `tf.Session`, `tf.contrib`) | `external/MGCE/Model-art/model-art.py` ~157-159, ~585-591; similar in Beauty/Taobao |
| Task | Implicit-feedback recommendation with BPR triplets `(u, i+, i-)` | `external/MGCE/Model-art/model-art.py` ~432-495; `external/MGCE/Model-art/load_data.py` ~228-251 |
| Data structure | User-item interaction matrix `R` as `scipy.sparse.dok_matrix` then `csr/coo`; bipartite adjacency as sparse SciPy matrix | `external/MGCE/Model-art/load_data.py` ~76, ~103, ~156-158, ~186-226 |
| Graph propagation kernel | Folded sparse-dense matmul via `tf.sparse_tensor_dense_matmul` on normalized adjacency | `external/MGCE/Model-art/model-art.py` ~312-423 |
| Causal branches | No explicit factor routing; separate branch embeddings for interest vs conformity per modality | `external/MGCE/Model-art/model-art.py` ~181-243 |
| Independence loss | Negative mean cosine similarity between branch means (not HSIC/DCor) | `external/MGCE/Model-art/model-art.py` ~497-518 |
| Modalities | Art/Beauty: visual+text+popularity; Taobao: visual+popularity only | `external/MGCE/Model-taobao/model-taobao.py` ~136-180, ~200-223 |
| Scoring | Sum of dot products from base + branch-specific embeddings | `external/MGCE/Model-art/model-art.py` ~245-254; `Model-taobao/model-taobao.py` ~222-227 |
| Sampler | Python-loop random user, one random positive, rejection-sampled random negative | `external/MGCE/Model-art/load_data.py` ~228-251 |
| Evaluation | Per-user sampled ranking: 99 random negatives + positives, top-K via heap | `external/MGCE/Model-art/model-art.py` ~45-91 |
| Early stop | Score = `HR@5 + NDCG@5`, stop after 10 non-improving evals | `external/MGCE/Model-art/model-art.py` ~641-659 |

---

## 1) End-to-End Data Flow & Ingestion Mechanics

### 1.1 Raw Data Ingestion Pipeline (disk → memory)

**Primary implementation files**:
- `external/MGCE/Model-art/load_data.py` (~10-275)
- `external/MGCE/Model-beauty/load_data.py` (~10-275)
- `external/MGCE/Model-taobao/load_data.py` (~10-275)

All three loaders are effectively identical.

### 1.2 Exact ingestion logic (code extraction)

```python
class Data(object):
    def __init__(self, path, batch_size):
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        img_feat_file = path + '/item2imgfeat.txt'
        text_feat_file = path + '/itemtitle2vec.txt'
        pop_file = path + '/item_popularity.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                self.exist_users.append(uid)
                self.n_items = max(self.n_items, max(items))
                self.n_users = max(self.n_users, uid)
                self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')[1:]]
                except Exception:
                    continue
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1
        self.exist_items = list(range(self.n_items))

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    for i in train_items:
                        self.R[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
```

Reference: `external/MGCE/Model-art/load_data.py` ~29-124.

### 1.3 Feature tensors and storage type

- `item2imgfeat.txt` → `self.imageFeaMatrix` (Python list-of-lists shape `[n_items, 4096]`)
- `itemtitle2vec.txt` → `self.textFeatMatrix` (Python list-of-lists shape `[n_items, 300]`, skipped when `title_enable=False`)
- `item_popularity.txt` → `self.popFeaMatrix` (Python list-of-lists shape `[n_items, 1]`)
- Interaction matrix:
  - initially `sp.dok_matrix`
  - then `self.R = self.R.tocsr()`
  - also cached as `self.coo_R = self.R.tocoo()`

Reference: `external/MGCE/Model-art/load_data.py` ~126-158.

### 1.4 Graph construction & normalization format

Graph is a **sparse SciPy adjacency matrix**, not adjacency-list and not dense:

```python
adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
adj_mat = adj_mat.tolil()
R = self.R.tolil()

adj_mat[:self.n_users, self.n_users: self.n_users + self.n_items] = R
adj_mat[self.n_users: self.n_users + self.n_items, :self.n_users] = R.T

def normalized_adj_symetric(adj, d1, d2):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, d1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    d_inv_sqrt_last = np.power(rowsum, d2).flatten()
    d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
    d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()
```

Reference: `external/MGCE/Model-art/load_data.py` ~186-226.

Adjacency variants cached to disk (`.npz`):
- `adj_mat_left.npz` (`d1=-1.0, d2=-0.0`)
- `adj_mat_3.npz` (`-0.5, -0.3`)
- `adj_mat_4.npz` (`-0.5, -0.4`)
- `adj_mat_5.npz` (`-0.5, -0.5`)

Reference: `external/MGCE/Model-art/load_data.py` ~169-182, ~215-218.

### 1.5 Sampling & batching engine (negative sampling)

```python
def sample_u(self):
    total_users = self.exist_users
    users = rd.sample(total_users, self.batch_size)

    def sample_pos_items_for_u(u):
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        return pos_items[pos_id]

    def sample_neg_items_for_u(u):
        pos_items = self.train_items[u]
        while True:
            neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_id not in pos_items:
                return neg_id

    pos_items, neg_items = [], []
    for u in users:
        pos_i = sample_pos_items_for_u(u)
        neg_i = sample_neg_items_for_u(u)
        pos_items.append(pos_i)
        neg_items.append(neg_i)

    return users, pos_items, neg_items
```

Reference: `external/MGCE/Model-art/load_data.py` ~228-251.

**System audit**:
- Sampling is fully Python-loop + per-user rejection loop (CPU-bound), not vectorized.
- No hard negative mining: negatives are uniform random non-interacted items only.

### 1.6 Tensor shape lineage (single forward pass)

Let:
- `B` = train batch size (`2048` default)
- `U` = number of users
- `I` = number of items
- `N = U + I`
- `D` = embedding size (`64`)

**Inputs**:
- `users`: `[B]` (`tf.int32` placeholder)
- `pos_items`: `[B]`
- `neg_items`: `[B]`

Reference: `external/MGCE/Model-art/model-art.py` ~157-159.

**Base learnable embeddings**:
- `user_embedding`: `[U, D]`
- `item_embedding`: `[I, D]`

Reference: `external/MGCE/Model-art/model-art.py` ~272-280.

**Modality projections**:
- `im_v_pre = img_feat @ w1_v`: `[I, 4096] @ [4096, D] -> [I, D]`
- `im_t_pre = text_feat @ w1_t`: `[I, 300] @ [300, D] -> [I, D]`
- `im_p` from bucket embedding: `[I, D]`

Reference: `external/MGCE/Model-art/model-art.py` ~166-180.

**Graph propagation state per branch**:
- Concatenated node embedding before split: `[N, D]`
- After each message-passing layer: `[N, D]`
- After split: user `[U, D]`, item `[I, D]`

Reference: `external/MGCE/Model-art/model-art.py` ~327-423.

**Batch lookups**:
- user batch embedding: `[B, D]`
- positive item batch embedding: `[B, D]`
- negative item batch embedding: `[B, D]`

Reference: `external/MGCE/Model-art/model-art.py` ~193-243.

**Score tensors**:
- Training triplet branch scores: `reduce_sum(u * i, axis=1)` → `[B]`
- Full ranking at eval: `batch_ratings = u_batch @ item_all^T` + branch terms → `[B_eval, I]`

Reference: `external/MGCE/Model-art/model-art.py` ~245-254, ~432-495.

**Important note on requested factorized shape**:
- Repository has **no explicit factor axis** (`[N_factors, B, D/N_factors]` does not exist).
- “Disentanglement” is implemented as separate branch-specific embedding tables and propagation paths (interest/conformity × modality), each still `[*, D]`.

---

## 2) Core Engine Deconstruction (Reusable Modules)

## 2.1 Disentanglement engine (actual implementation)

### Mechanism verdict
- Not static slice partitioning
- Not iterative capsule-style routing
- Not per-factor projection stack

It is a **branch-wise parameter separation**:
- separate user embeddings for interest/conformity (`user_int_embedding_*`, `user_con_embedding_*`)
- separate graph propagation calls with different adjacency configs
- conformity branch injects popularity embedding (`im_p`) into item state

Reference: `external/MGCE/Model-art/model-art.py` ~181-243, ~349-423.

### Full code blocks (interest/conformity signal routing)

```python
def _create_norm_embed_int_v(self):

    A_fold_hat = self._split_A_hat(self.norm_adj_int)

    ego_embeddings_v = tf.concat([self.um_int_v, self.im_v], axis=0)

    for k in range(0, self.int_layers):
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_v))
        side_embeddings = tf.concat(temp_embed, 0)
        ego_embeddings_v = side_embeddings

    u_embed, i_embed = tf.split(ego_embeddings_v, [self.n_users, self.n_items], 0)
    return u_embed, i_embed
```

```python
def _create_norm_embed_con_v(self):

    A_fold_hat = self._split_A_hat(self.norm_adj_con)

    ego_embeddings_v = tf.concat([self.um_con_v, self.im_v+self.im_p], axis=0)

    for k in range(0, self.con_layers):
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_v))
        side_embeddings = tf.concat(temp_embed, 0)
        ego_embeddings_v = side_embeddings

    u_embed, i_embed = tf.split(ego_embeddings_v, [self.n_users, self.n_items], 0)
    return u_embed, i_embed
```

Textual equivalents:
- `_create_norm_embed_int_t`
- `_create_norm_embed_con_t`

Reference: `external/MGCE/Model-art/model-art.py` ~349-423.

### Contract

Input contract for each branch function:
- `norm_adj_*`: SciPy CSR/COO matrix shape `[N, N]`
- branch user embedding table: `[U, D]`
- branch item init embedding: `[I, D]`
- integer `layers`

Output contract:
- user branch embedding: `[U, D]`
- item branch embedding: `[I, D]`

## 2.2 Independence engine (causal regularizer)

### Full implementation extraction

```python
def _dis_loss(self):

    discrepancy_loss = self.mean_cos_dis(self.u_g_int_embeddings_v, self.u_g_con_embeddings_v) + \
                       self.mean_cos_dis(self.u_g_int_embeddings_t, self.u_g_con_embeddings_t) + \
                       self.mean_cos_dis(self.pos_i_g_int_embeddings_v, self.pos_i_g_con_embeddings_v) + \
                       self.mean_cos_dis(self.pos_i_g_int_embeddings_t, self.pos_i_g_con_embeddings_t)

    dis_loss = - discrepancy_loss / self.batch_size
    return dis_loss


def mean_cos_dis(self, x, y):
    x_mean = tf.reduce_mean(x, axis=0)
    y_mean = tf.reduce_mean(y, axis=0)

    x_norm = tf.nn.l2_normalize(x_mean)
    y_norm = tf.nn.l2_normalize(y_mean)

    x_square_sqrt = tf.sqrt(tf.reduce_sum(tf.square(x_norm)))
    y_square_sqrt = tf.sqrt(tf.reduce_sum(tf.square(y_norm)))
    xy = tf.reduce_sum(x_norm * y_norm)
    cov = xy / (x_square_sqrt * y_square_sqrt + 1e-8)

    return cov
```

Reference: `external/MGCE/Model-art/model-art.py` ~497-518.

### Mechanism classification
- This is a **mean-direction cosine discrepancy** regularizer.
- It is **not** HSIC, DCor, MMD, or pairwise contrastive estimation.

### Stability guards detected
- `+ 1e-8` in cosine denominator (`mean_cos_dis`) to avoid divide-by-zero.
- `epsilon = 1e-5` in ZCA whitening SVD inversion path.

Reference: `external/MGCE/Model-art/model-art.py` ~303, ~517.

## 2.3 Propagation kernel (message passing)

### Sparse kernel extraction

```python
def _split_A_hat(self, X):
    A_fold_hat = []
    fold_len = (self.n_users + self.n_items) // self.n_fold
    for i_fold in range(self.n_fold):
        start = i_fold * fold_len
        if i_fold == self.n_fold - 1:
            end = self.n_users + self.n_items
        else:
            end = (i_fold + 1) * fold_len

        A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
    return A_fold_hat


def _create_norm_embed(self):

    A_fold_hat = self._split_A_hat(self.norm_adj_base)
    ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

    for k in range(0, self.base_layers):
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
        side_embeddings = tf.concat(temp_embed, 0)
        ego_embeddings = side_embeddings

    u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)
    return u_g_embeddings, i_g_embeddings
```

Reference: `external/MGCE/Model-art/model-art.py` ~312-347.

### Kernel traits
- Operation: `tf.sparse_tensor_dense_matmul`
- Edge weights: yes, from normalized adjacency values
- Attention: no
- Factor-specific edge routing: no
- Fold partition (`n_fold=10`) used for memory control

---

## 3) Performance Profiling & Linux/RTX 5080 Adaptation

## 3.1 Latency sinks (code smells)

1. **CPU-bound negative sampling loops**
   - Rejection sampling in Python `while True` per user.
   - Reference: `external/MGCE/Model-art/load_data.py` ~236-251.

2. **Python dictionaries + loops in evaluation**
   - Build per-user `item_score` dict and `heapq.nlargest`.
   - Reference: `external/MGCE/Model-art/model-art.py` ~45-91.

3. **Multiprocessing pool overhead during eval**
   - `multiprocessing.Pool(cores)` + serialization costs each eval call.
   - Reference: `external/MGCE/Model-art/model-art.py` ~99-132.

4. **ZCA whitening with per-batch SVD in graph build**
   - `tf.linalg.svd` over covariance at startup.
   - Reference: `external/MGCE/Model-art/model-art.py` ~290-309.

5. **TF1 static graph + session feed_dict**
   - Repeated host feed dict each training step.
   - Reference: `external/MGCE/Model-art/model-art.py` ~614-623.

## 3.2 VRAM scaling audit

### Potentially large tensors
- `batch_ratings` in eval: `[B_eval, I]` (dense), where `B_eval = 2 * batch_size`.
  - Reference: `external/MGCE/Model-art/model-art.py` ~101, ~116-121, ~245-254.
- Node embeddings per branch: `[U+I, D]` repeated across multiple branches.

### Does causal loss allocate dense $N \times N$?
- **No.** `mean_cos_dis` uses only feature means and vector dot products; no pairwise matrix.
- Therefore causal regularizer memory is $O(D)$ per term, not $O(N^2)$.

Reference: `external/MGCE/Model-art/model-art.py` ~507-518.

## 3.3 Formal complexity

Let:
- `E` = number of observed user-item edges in interaction graph
- `N = U + I`
- `D` = embedding dimension
- `L_base`, `L_int`, `L_con` = layer counts
- `B` = training batch size

### Graph construction (preprocessing)
- Build adjacency from `R`: $O(E)$ insertion
- Degree normalization + sparse products: approximately $O(E)$ per normalization variant
- Total for four variants: $O(4E)$ (constant factor)

Reference: `external/MGCE/Model-art/load_data.py` ~186-226.

### Training iteration (forward/backward)
- Sparse propagation (dominant):
  $$O\big((L_{base}+L_{int}+L_{con}(+L_{int,t}+L_{con,t})) \cdot E \cdot D\big)$$
  (Art/Beauty have five propagation paths: base + int_v + con_v + int_t + con_t)
- BPR triplet losses on batch: $O(BD)$ each branch
- Discrepancy loss: $O(BD)$

### Inference/ranking
- Full eval scoring in code computes dense user-batch × item matrix:
  $$O(B_{eval} \cdot I \cdot D)$$
- Then sampled ranking only over `99 + |test_pos|` candidates per user via dict+heap (not true full sort over all items).

Reference: `external/MGCE/Model-art/model-art.py` ~45-91, ~94-132.

---

## 4) Architectural Anomalies & Hyperparameters

## 4.1 Symmetry breaks / biased logic

1. **Asymmetric modality weighting in Art/Beauty scoring**
   - `lambda_m` multiplies visual branch terms but not textual terms.
   - Reference: `external/MGCE/Model-art/model-art.py` ~245-254; `Model-beauty/model_beauty.py` ~245-254.

2. **Popularity injected only into conformity branch**
   - `ego_embeddings_v = concat([um_con_v, im_v + im_p])`
   - Interest branch uses `im_v` only.
   - Reference: `external/MGCE/Model-art/model-art.py` ~353, ~374.

3. **Evaluation randomness**
   - Test candidates sampled randomly each run (`rd.sample` 99 negatives), non-deterministic metric variance.
   - Reference: `external/MGCE/Model-art/model-art.py` ~54.

4. **No explicit hard negatives**
   - Negative sampling uniform random exclusion of interacted set.
   - Reference: `external/MGCE/Model-art/load_data.py` ~236-243.

## 4.2 Implementation shortcuts / code smells

1. **List aliasing in feature matrices**
   - `self.imageFeaMatrix = [[0.] * d1] * self.n_items` (same row object repeated).
   - Same pattern for text/pop matrices.
   - Reference: `external/MGCE/Model-art/load_data.py` ~132, ~143, ~153.

2. **Potential bug in `train_users_f` initialization path**
   - On first encounter, creates empty list but does not append current user in that branch.
   - Reference: `external/MGCE/Model-art/load_data.py` ~86-95.

3. **Unused variable in Taobao model**
   - `textual_enable` set but not used.
   - Reference: `external/MGCE/Model-taobao/model-taobao.py` ~42-46.

## 4.3 Magic number list

### Architectural constants
- `embed_size = 64`
- `n_fold = 10`
- `num_buckets = 3` for popularity embedding
- SVD epsilon `1e-5`
- Cosine denominator epsilon `1e-8`

References:
- `external/MGCE/Model-art/model-art.py` ~34, ~144, ~173, ~303, ~517.

### Optimization constants (global)
- `lr = 0.001`
- `batch_size = 2048`
- `decay = 0.001`
- `epoch = 500`

References:
- `external/MGCE/Model-art/model-art.py` ~25, ~32-35.

### Dataset-specific constants

**Art** (`external/MGCE/Model-art/model-art.py` ~21-30, ~569-571):
- `base_layers=5`, `con_layers=2`, `int_layers=1`
- `lambda_m=0.6`, `mju_mf=0.7`, `mju_emb=0.2`, `eit=0.5`, `interval=10`
- adjacency config: base=`norm_4`, int=`norm_5`, con=`norm_5`

**Beauty** (`external/MGCE/Model-beauty/model_beauty.py` ~23-32, ~569-571):
- `base_layers=4`, `con_layers=2`, `int_layers=1`
- `lambda_m=0.5`, `mju_mf=0.3`, `mju_emb=0.1`, `eit=0.3`, `interval=10`
- adjacency config: base=`norm_4`, int=`norm_3`, con=`norm_5`

**Taobao** (`external/MGCE/Model-taobao/model-taobao.py` ~22-30, ~476-478):
- `base_layers=5`, `con_layers=2`, `int_layers=1`
- no `lambda_m`
- `mju_mf=0.7`, `mju_emb=0.5`, `eit=0.7`, `interval=5`
- adjacency config: base=`norm_3`, int=`norm_5`, con=`norm_5`

---

## 5) U-CaGNN Integration Interface

## 5.1 Minimal Viable Code (exactly three functions)

Below are the three highest-impact functions to replicate MGCE’s causal behavior in a unified PyTorch-style module.

### Function 1 — Branch propagation kernel (interest/conformity)

```python
import torch

def propagate_branch(
    user_init: torch.Tensor,       # [U, D]
    item_init: torch.Tensor,       # [I, D]
    norm_adj: torch.Tensor,        # sparse COO [U+I, U+I]
    layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MGCE-equivalent branch propagation.
    Mirrors TF path: concat -> repeated sparse mm -> split.
    """
    n_users = user_init.size(0)
    ego = torch.cat([user_init, item_init], dim=0)   # [N, D]
    for _ in range(layers):
        ego = torch.sparse.mm(norm_adj, ego)         # [N, D]
    user_out, item_out = torch.split(ego, [n_users, item_init.size(0)], dim=0)
    return user_out, item_out
```

### Function 2 — Causal discrepancy regularizer

```python
import torch
import torch.nn.functional as F

def discrepancy_loss(*pairs: tuple[torch.Tensor, torch.Tensor], batch_size: int) -> torch.Tensor:
    """
    MGCE-equivalent discrepancy:
    dis = - (sum cosine(mean(x), mean(y))) / batch_size
    """
    total = 0.0
    for x, y in pairs:
        x_mean = x.mean(dim=0)
        y_mean = y.mean(dim=0)
        x_norm = F.normalize(x_mean, p=2, dim=0)
        y_norm = F.normalize(y_mean, p=2, dim=0)
        denom = (x_norm.pow(2).sum().sqrt() * y_norm.pow(2).sum().sqrt()) + 1e-8
        total = total + (x_norm * y_norm).sum() / denom
    return -total / batch_size
```

### Function 3 — MGCE objective composer (BPR + branch weights + discrepancy)

```python
import torch
import torch.nn.functional as F

def mgce_objective(
    base_pos: torch.Tensor, base_neg: torch.Tensor,
    int_pos_terms: list[tuple[torch.Tensor, torch.Tensor]],
    con_pos_terms: list[tuple[torch.Tensor, torch.Tensor]],
    reg_terms_base: torch.Tensor,
    reg_terms_int: torch.Tensor,
    reg_terms_con: torch.Tensor,
    mju_mf: float,
    mju_emb: float,
    eit: float,
    decay: float,
    batch_size: int,
    dis_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicates MGCE scalar composition:
    mf = mf_base + mju_mf*mf_int + mju_mf*mf_con + eit*dis
    emb = emb_base + mju_emb*emb_int + mju_emb*emb_con
    total = mf + emb
    """
    mf_base = F.softplus(-(base_pos - base_neg)).mean()

    mf_int = 0.0
    for pos, neg in int_pos_terms:
        mf_int = mf_int + F.softplus(-(pos - neg)).mean()

    mf_con = 0.0
    for pos, neg in con_pos_terms:
        mf_con = mf_con + F.softplus(-(pos - neg)).mean()

    emb_base = decay * reg_terms_base / batch_size
    emb_int = decay * reg_terms_int / batch_size
    emb_con = decay * reg_terms_con / batch_size

    mf = mf_base + mju_mf * mf_int + mju_mf * mf_con + eit * dis_loss
    emb = emb_base + mju_emb * emb_int + mju_emb * emb_con
    return mf + emb, mf, emb
```

These three functions cover:
- branch-specific sparse propagation,
- causal discrepancy pressure,
- exact scalar objective assembly.

## 5.2 Module API contract for U-CaGNN wrapper

### Inputs
- `edge_index` / sparse adjacency (pre-normalized variants for base/int/con)
- `user_ids`, `pos_item_ids`, `neg_item_ids` (`[B]`)
- item modality matrices:
  - visual `[I, Fv]`
  - textual `[I, Ft]` (optional, dataset-flagged)
  - popularity scalar `[I, 1]`

### State (learnable)
- Base embeddings: `E_u_base[U,D]`, `E_i_base[I,D]`
- Interest embeddings by modality: `E_u_int_v`, `E_u_int_t`
- Conformity embeddings by modality: `E_u_con_v`, `E_u_con_t`
- Feature projections: `W_v[Fv,D]`, `W_t[Ft,D]`
- Popularity embedding lookup: `Emb_pop[num_buckets,D]`

### Outputs
- `loss_total`, `loss_mf`, `loss_reg`, `loss_dis`
- optional `scores_batch` `[B, I]` for ranking
- optional branch embeddings for diagnostics

### Forward contract
1. Project modalities (optionally whiten/offline normalize).
2. Build branch initial node states:
   - int branch uses content only,
   - con branch uses content + pop embedding.
3. Apply `propagate_branch` per branch with corresponding normalized adjacency.
4. Gather batch triplets by index.
5. Compute branch BPR terms and discrepancy loss.
6. Compose weighted total objective.

---

## Appendix A — Key Training Loop Wiring (for exact behavior)

```python
for epoch in range(500):
    n_batch = data_generator.n_train // batch_size + 1
    for idx in range(n_batch):
        users, pos_items, neg_items = data_generator.sample_u()
        _, batch_mf_loss, batch_emb_loss = sess.run(
            [model.opt_1, model.mf_loss, model.emb_loss],
            feed_dict={
                model.users: users,
                model.pos_items: pos_items,
                model.neg_items: neg_items
            }
        )

    if (epoch + 1) % interval == 0:
        result = test(sess, model, users_to_test, data_generator.exist_items, batch_size, cores)
        score = result['hit_ratio'][4] + result['ndcg'][4]
        if score improves: early_stopping = 0 else early_stopping += 1
        if early_stopping == 10: break
```

Reference: `external/MGCE/Model-art/model-art.py` ~606-659.

---

## Appendix B — Direct reconstruction notes (no source revisit needed)

1. Use sparse bipartite adjacency over `U + I` nodes and run pure linear propagation (no nonlinearity, no residual).
2. Implement separate branch embeddings for interest/conformity rather than latent factor splits.
3. Conformity branch must include popularity-conditioned item signal.
4. Use BPR on base/int/con branches + weighted discrepancy term exactly as scalar-composed in source.
5. Keep evaluation protocol if reproducing reported numbers: sampled 99 negatives per user, not full deterministic ranking.
