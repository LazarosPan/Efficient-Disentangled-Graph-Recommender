# MCLN Technical Repository Audit (Code-First Architectural Blueprint)

## Mechanical Summary Table

| Spec | Implementation Evidence | Mechanical Reading |
|---|---|---|
| Framework | `external/MCLN/README.md:L24-L26` | TensorFlow 1.10 + Python 3.6 runtime. |
| Core model family | `external/MCLN/Model-art/model-art.py:L14`, `external/MCLN/Model-beauty/model-beauty.py:L14`, `external/MCLN/Model-taobao/model-taobao.py:L14` | Single architecture with dataset-specific variants. |
| Graph encoder | `external/MCLN/Model-art/model-art.py:L157-L176`, `L168` | LightGCN-style propagation over sparse normalized adjacency (`tf.sparse_tensor_dense_matmul`). |
| Causal module | `external/MCLN/Model-art/model-art.py:L178-L215`, `L217-L258` | Counterfactual attention blocks (positive path subtracts uninteracted attention score). |
| Objective | `external/MCLN/Model-art/model-art.py:L274-L310` | Multi-branch BPR + L2 regularization only (no explicit independence regularizer). |
| Sampler | `external/MCLN/Model-art/load_data.py:L200-L239` | Python loop + rejection sampling (`while True`) on CPU. |
| Adjacency build | `external/MCLN/Model-art/load_data.py:L158-L196` | Bipartite block matrix + multiple degree normalizations; saved as `.npz`. |
| Training loop | `external/MCLN/Model-art/model-art.py:L417-L434` | `feed_dict` session runs per mini-batch. |
| Eval protocol | `external/MCLN/Model-art/load_data.py:L243-L262`, `external/MCLN/Model-art/model-art.py:L318-L357` | 99 negatives + 1 positive per user, heap-based top-k ranking. |

---

## 1) End-to-End Data Flow (Single Datapoint Pipeline)

### 1.1 Ingestion Logic (raw `train.txt` -> sparse graph + multimodal arrays)

**Code path (Art/Beauty share same loader logic):**

```python
# external/MCLN/Model-art/load_data.py:L36-L55
with open(train_file) as f:
    for l in f.readlines():
        ...
        items = [int(i) for i in l[1:]]
        uid = int(l[0])
        self.exist_users.append(uid)
        self.n_items = max(self.n_items, max(items))
        self.n_users = max(self.n_users, uid)
        self.n_train += len(items)

# external/MCLN/Model-art/load_data.py:L63-L66
self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
self.train_items, self.test_set = {}, {}

# external/MCLN/Model-art/load_data.py:L82-L95
for i in train_items:
    self.R[uid, i] = 1.
    ...
self.train_items[uid] = train_items

# external/MCLN/Model-art/load_data.py:L132-L134
self.R = self.R.tocsr()
self.coo_R = self.R.tocoo()
```

**Resulting structures:**
- Interaction matrix: `scipy.sparse.dok_matrix` -> `csr_matrix` (`R`) -> `coo_matrix` (`coo_R`)  
  Evidence: `external/MCLN/Model-art/load_data.py:L63`, `L132-L134`.
- User->train items map: Python `dict[int, list[int]]` (`self.train_items`)  
  Evidence: `external/MCLN/Model-art/load_data.py:L65`, `L95`.
- Test map: Python `dict[int, list[int]]` (`self.test_set`)  
  Evidence: `external/MCLN/Model-art/load_data.py:L65`, `L106`.
- Image/title features: Python nested lists (`imageFeatMatrix`, `textFeatMatrix`)  
  Evidence: `external/MCLN/Model-art/load_data.py:L116-L130`.

### 1.2 Graph Construction + Normalization

```python
# external/MCLN/Model-art/load_data.py:L158-L196
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

norm_adj_mat_54 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), -0.5, -0.4)
```

**Training selection:**
- Art/Beauty pick `norm_4` as `config['norm_adj']`.  
  Evidence: `external/MCLN/Model-art/model-art.py:L394-L396`, `external/MCLN/Model-beauty/model-beauty.py:L393-L395`.
- Taobao builds/loads an extra `norm_adj_mat_2` but still picks `norm_4`.  
  Evidence: `external/MCLN/Model-taobao/load_data.py:L142-L158`, `external/MCLN/Model-taobao/model-taobao.py:L376-L378`.

### 1.3 Batching & Sampling

**Sampling loop (CPU-bound Python, rejection sampling):**

```python
# external/MCLN/Model-art/load_data.py:L200-L239
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

    def sample_int_items_for_u(u):
        pos_items = self.train_items[u]
        while True:
            int_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if int_id not in pos_items:
                return int_id

    pos_items, neg_items, int_items = [], [], []
    for u in users:
        pos_items.append(sample_pos_items_for_u(u))
        neg_items.append(sample_neg_items_for_u(u))
        int_items.append(sample_int_items_for_u(u))

    return users, pos_items, neg_items, int_items
```

**Execution in train loop:**
- `users, pos_items, neg_items, int_items = data_generator.sample_u()` per batch.  
  Evidence: `external/MCLN/Model-art/model-art.py:L424`.
- CPU list output is fed through `feed_dict` to TF placeholders.  
  Evidence: `external/MCLN/Model-art/model-art.py:L61-L64`, `L426-L433`.

### 1.4 Tensor Shape Tracking (Art/Beauty path, `embed_size = 64`)

- **Input IDs:**
  - `users`, `pos_items`, `neg_items`, `int_items`: `[B]` (`tf.placeholder(... shape=(None,))`)  
    Evidence: `external/MCLN/Model-art/model-art.py:L61-L64`.

- **Hidden graph states:**
  - User/item base tables: `[U, D]`, `[I, D]` from Xavier init.  
    Evidence: `external/MCLN/Model-art/model-art.py:L131-L134`.
  - Concatenated ego: `[U+I, D]`.  
    Evidence: `external/MCLN/Model-art/model-art.py:L161`.
  - After each propagation layer: `[U+I, D]` via sparse-dense matmul and concat folds.  
    Evidence: `external/MCLN/Model-art/model-art.py:L164-L171`.
  - Final split: `u_g_embeddings [U, D]`, `i_g_embeddings [I, D]`.  
    Evidence: `external/MCLN/Model-art/model-art.py:L175`.

- **Multimodal item embeddings:**
  - `im_v = img_feat @ w1_v`: `[I, 4096] @ [4096, D] -> [I, D]`  
    Evidence: `external/MCLN/Model-art/model-art.py:L68`, `L139`.
  - `im_t = text_feat @ w1_t`: `[I, 300] @ [300, D] -> [I, D]`  
    Evidence: `external/MCLN/Model-art/model-art.py:L71`, `L140`.

- **Causal input tensor:**
  - `pos_inputs_embeddings = concat([graph, visual, text], axis=1)`: `[B, 3D]`.  
    Evidence: `external/MCLN/Model-art/model-art.py:L103`.

- **Counterfactual attention internals:**
  - `Q,K,V`: `[B, 3D]`.
  - `score = Q @ K^T`: `[B, B]`.
  - `attention = softmax(score) @ V`: `[B, 3D]`.  
    Evidence: `external/MCLN/Model-art/model-art.py:L220-L234`.

- **“Causal Factors” mapping to requested shape:**
  - Repository does **not** keep an explicit factor axis `[N_Factors, B, Cluster_Dim]`; factors are flattened in concat (`[B, 3D]` or `[B, 2D]` Taobao).  
    Evidence: `external/MCLN/Model-art/model-art.py:L103-L105`, `external/MCLN/Model-taobao/model-taobao.py:L95-L97`.
  - Porting equivalent factorized view for EDGRec: reshape concat as `[M, B, D]` where `M=3` (Art/Beauty) or `M=2` (Taobao).

### 1.5 Single-point scoring path

```python
# external/MCLN/Model-art/model-art.py:L113-L117
self.multiply = tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings, 1) + \
                tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings_v, 1) + \
                tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings_t, 1) + \
                tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings_m, 1)
```

- Per candidate item score is scalar per row (`[B]`), then ranked with `heapq.nlargest`.  
  Evidence: `external/MCLN/Model-art/model-art.py:L324-L329`.

---

## 2) Module Contracts & Reusable Logic

### 2.1 Disentanglement Engine (actual implementation)

**Observed mechanism:** learned projections + counterfactual self-attention; **not** `split/chunk`.

```python
# external/MCLN/Model-art/model-art.py:L217-L237
def counterfactual_learning_layer_1(self, query, key_value, query_int, key_value_int, activation=None, name=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        V = tf.layers.dense(key_value, units=3 * self.emb_dim, activation=activation, use_bias=False, name='V')
        K = tf.layers.dense(key_value, units=3 * self.emb_dim, activation=activation, use_bias=False, name='K')
        Q = tf.layers.dense(query, units=3 * self.emb_dim, activation=activation, use_bias=False, name='Q')
        K_int = tf.layers.dense(key_value_int, units=3 * self.emb_dim, activation=activation, use_bias=False, name='K_int')
        Q_int = tf.layers.dense(query_int, units=3 * self.emb_dim, activation=activation, use_bias=False, name='Q_int')

        score = tf.matmul(Q, tf.transpose(K)) / np.sqrt(3 * self.emb_dim)
        score_int = tf.matmul(Q_int, tf.transpose(K_int)) / np.sqrt(3 * self.emb_dim)
        score = score - score_int
        softmax = tf.nn.softmax(score, axis=1)
        attention = tf.matmul(softmax, V)

        counterfactual_learning = tf.layers.dense(attention, units=3 * self.emb_dim, activation=activation,
                                                  use_bias=False, name='linear')
        counterfactual_learning += query
        counterfactual_learning = tf.contrib.layers.layer_norm(counterfactual_learning, begin_norm_axis=1)
        return counterfactual_learning
```

```python
# external/MCLN/Model-art/model-art.py:L178-L197
def causal_difference_1(self, cd_inputs_embedding, cd_inputs_embedding_int):
    cd_outputs = cd_inputs_embedding
    cd_outputs_int = cd_inputs_embedding_int
    for i in range(self.n_mca):
        cl_outputs = self.counterfactual_learning_layer_1(...)
        cd_outputs = self.feed_forward_layer(cl_outputs, activation=tf.nn.relu, name='cd_dense' + str(i))
    return cd_outputs
```

**Contract for EDGRec (ported):**
- **Input:**
  - `x_pos`: `[B, M*D]` (interacted item multimodal concat)
  - `x_int`: `[B, M*D]` (uninteracted sampled item multimodal concat)
- **Output:**
  - `x_cf`: `[B, M*D]` (counterfactually filtered representation)
- **Mechanism:** learned linear projections + attention matrix subtraction (`score - score_int`) + residual + layer norm.

### 2.2 Independence Engine (Loss Implementation)

**Repository status:** no independence regularizer (no HSIC / dCor / covariance penalty in training objective).

```python
# external/MCLN/Model-art/model-art.py:L274-L310
def create_bpr_loss(self):
    ...
    mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores))) + ...
    emb_loss = self.decay * (regularizer_mf + regularizer_mf_t + regularizer_mf_v + regularizer_mf_m) / self.batch_size
    loss = mf_loss + emb_loss
    return loss, mf_loss, emb_loss
```

**Evidence for absence in codebase search:** no code matches for `hsic|distance|independ|covariance|corr|dcor` under `external/MCLN/**` except README prose.  
Evidence: `external/MCLN/README.md:L11`, `L14`.

### 2.3 Stability Guards

**Present guards:**
- Degree normalization inf handling: `d_inv_sqrt[np.isinf(...)] = 0.`  
  Evidence: `external/MCLN/Model-art/load_data.py:L178`, `L182`.
- Runtime NaN guard at epoch level: `if np.isnan(loss) == True: sys.exit()`.  
  Evidence: `external/MCLN/Model-art/model-art.py:L437-L439`.

**Missing guards:**
- No epsilon constants (e.g., `1e-8`) in attention denominator, softmax, or BPR softplus path.

### 2.4 Propagation Kernel

```python
# external/MCLN/Model-art/model-art.py:L142-L155, L362-L365
A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
...
temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
...
def _convert_sp_mat_to_sp_tensor(self, X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
```

**Contract:**
- **Input:** sparse COO/CSR adjacency block `[N_block, N_total]`, dense node embeddings `[N_total, D]`.
- **Output:** propagated block embeddings `[N_block, D]`.
- **Optimization note:** direct equivalent in PyTorch/DGL should use `torch.sparse.mm` / DGL SpMM (or `torch_sparse.spmm`), avoiding TF1 `feed_dict` orchestration.

---

## 3) Performance & Bottleneck Profiling (RTX 5080 / Linux)

### 3.1 Latency Sinks (CPU bottlenecks before GPU)

1. **Negative/intervention sampling uses Python `while True` rejection loops** (`sample_u`), per-user per-batch.  
   Evidence: `external/MCLN/Model-art/load_data.py:L213-L223`.
2. **Batch creation is list-based Python loop** (`for u in users:` append arrays).  
   Evidence: `external/MCLN/Model-art/load_data.py:L225-L236`.
3. **`feed_dict` transfer every step** from Python lists to graph placeholders.  
   Evidence: `external/MCLN/Model-art/model-art.py:L61-L64`, `L426-L433`.
4. **Evaluation negative construction loop (`99` negatives) is Python-side**.  
   Evidence: `external/MCLN/Model-art/load_data.py:L249-L254`.
5. **Per-user heap ranking in Python** (`heapq.nlargest`), not vectorized top-k on GPU.  
   Evidence: `external/MCLN/Model-art/model-art.py:L327-L329`.

### 3.2 VRAM / Compute Scaling

- **Graph propagation:** $O(L \cdot |E| \cdot D)$ sparse-dense multiplies (`L=n_layers`, 100 folds).  
  Evidence: `external/MCLN/Model-art/model-art.py:L51`, `L164-L169`.
- **Counterfactual attention:** `score = QK^T` per MCA block gives $O(B^2 \cdot M D)$ compute and $O(B^2)$ memory for score matrix.  
  Evidence: `external/MCLN/Model-art/model-art.py:L228-L233`, `L178-L197`.
- **BPR objective:** $O(B\cdot D)$ per score branch, linear in batch size.  
  Evidence: `external/MCLN/Model-art/model-art.py:L276-L295`.
- **Training epoch:** $O((N_{train}/B)\cdot[\text{sampling} + \text{forward/backward}])$.  
  Evidence: `external/MCLN/Model-art/model-art.py:L421-L434`.
- **Inference (reported protocol):** per test user fixed 100-item candidate set, ranking cost $O(100\log K)$ via heap (`K=5,10,20`).  
  Evidence: `external/MCLN/Model-art/load_data.py:L249-L258`, `external/MCLN/Model-art/model-art.py:L327-L329`.

### 3.3 Memory Efficiency Issues

- **Potential aliasing/wasteful list initialization for feature matrices:**
  - `self.imageFeatMatrix = [[0.] * 4096] * self.n_items`
  - `self.textFeatMatrix = [[0.] * 300] * self.n_items`  
  Evidence: `external/MCLN/Model-art/load_data.py:L116`, `L128`.
- **Dense attention matrix (`[B,B]`) materialized each MCA block.**  
  Evidence: `external/MCLN/Model-art/model-art.py:L228-L233`.
- **No sampled neighborhood graph mini-batching; full adjacency used each epoch.**  
  Evidence: `external/MCLN/Model-art/model-art.py:L394-L396`, `L157-L176`.

---

## 4) Architectural Anomalies & Code Smells

1. **`train_users_f` construction bug (first user per item is dropped):**

```python
# external/MCLN/Model-art/load_data.py:L76-L79
if i not in self.train_users_f:
    self.train_users_f[i] = []
else:
    self.train_users_f[i].append(uid)
```

- On first encounter, user is not appended.

2. **Positive/negative asymmetry in causal treatment:**
- Positive branch uses intervention-aware layer (`causal_difference_1`, score subtraction).
- Negative branch uses non-intervention layer (`causal_difference_2`).  
  Evidence: `external/MCLN/Model-art/model-art.py:L107-L108`, `L178-L215`, `L217-L237`, `L241-L258`.

3. **Dataset-tuned hardcoded constants (magic values):**
- Art: `n_layers=5`, `decay=1e-2`, `n_mca=2`, `epochs=1000`.  
  Evidence: `external/MCLN/Model-art/model-art.py:L24-L31`.
- Beauty: `n_layers=4`, `lambda_m=0.3`, `epochs=1000`.  
  Evidence: `external/MCLN/Model-beauty/model-beauty.py:L25-L27`, `L32`.
- Taobao: `n_layers=5`, `decay=1e-3`, `lambda_m=0.2`, `interval=5`, `epochs=500`, `n_mca=4`.  
  Evidence: `external/MCLN/Model-taobao/model-taobao.py:L25-L33`.

4. **Unused or weakly connected artifacts:**
- `cores = multiprocessing.cpu_count() // 3` computed but not used.  
  Evidence: `external/MCLN/Model-art/model-art.py:L376`.
- Taobao loader special-cases title disable by path string check (`'Taobao'` case-sensitive).  
  Evidence: `external/MCLN/Model-taobao/load_data.py:L12-L15`.

---

## 5) EDGRec Integration Blueprint (Minimal Viable Code)

### 5.1 If copying only three functions for causal effect

**Directly reusable core trio:**
1. `counterfactual_learning_layer_1` (intervention-aware attention)  
   Evidence: `external/MCLN/Model-art/model-art.py:L217-L237`.
2. `counterfactual_learning_layer_2` (standard attention for non-intervention path)  
   Evidence: `external/MCLN/Model-art/model-art.py:L241-L258`.
3. `feed_forward_layer` (residual MLP post-attention)  
   Evidence: `external/MCLN/Model-art/model-art.py:L261-L272`.

**Minimal wrapper loop needed around trio (same mechanics as repo):**

```python
# derived from external/MCLN/Model-art/model-art.py:L178-L215
x_pos = pos_concat      # [B, M*D]
x_int = int_concat      # [B, M*D]
x_neg = neg_concat      # [B, M*D]

for _ in range(n_mca):
    x_pos = feed_forward_layer(counterfactual_learning_layer_1(x_pos, x_pos, x_int, x_int), activation=relu)
    x_neg = feed_forward_layer(counterfactual_learning_layer_2(x_neg, x_neg), activation=relu)
```

### 5.2 Required interface for EDGRec (PyTorch/DGL)

**Inputs:**
- `user_emb`: `[U, D]`
- `item_emb_graph`: `[I, D]`
- `item_emb_visual`: `[I, D]` (optional depending on modality)
- `item_emb_text`: `[I, D]` (optional)
- `batch_users, batch_pos, batch_neg, batch_int`: `[B]`
- `norm_adj_sparse`: sparse tensor `[U+I, U+I]`

**Core contracts:**
- **Propagation Kernel:**
  - In: sparse adjacency + `[U+I, D]`
  - Out: propagated `[U+I, D]` (layer-averaged)
  - Source mechanics: `external/MCLN/Model-art/model-art.py:L157-L176`.
- **Causal Difference Block:**
  - In: `pos_concat [B,M*D]`, `neg_concat [B,M*D]`, `int_concat [B,M*D]`
  - Out: `pos_cf [B,M*D]`, `neg_cf [B,M*D]`
  - Source mechanics: `external/MCLN/Model-art/model-art.py:L178-L272`.
- **Scoring Head:**
  - In: `u_batch [B,D]`, item branch embeddings `[B,D]`
  - Out: scalar logits `[B]`
  - Source mechanics: `external/MCLN/Model-art/model-art.py:L113-L117`, `L276-L289`.

### 5.3 Complexity Summary for EDGRec port

- **Graph construction (offline):** $O(|E|)$ to read interactions + $O(|E|)$ to build bipartite adjacency; normalization sparse operations over nonzeros.  
  Source analog: `external/MCLN/Model-art/load_data.py:L36-L55`, `L158-L196`.
- **Training step:** $O(L|E|D + n_{mca}B^2MD + B D)$.
- **Inference step (current protocol):** $O(100\cdot D + 100\log K)$ per user with 99 sampled negatives + 1 positive.

---

## Appendix: Variant Delta Map (for unification)

- **Modality count:**
  - Art/Beauty use 3-way concat (`graph+visual+text`) -> `[B, 3D]`.  
    Evidence: `external/MCLN/Model-art/model-art.py:L103-L105`.
  - Taobao uses 2-way concat (`graph+visual`) -> `[B, 2D]`.  
    Evidence: `external/MCLN/Model-taobao/model-taobao.py:L95-L97`.
- **Causal projection widths:**
  - Art/Beauty attention/project units: `3 * emb_dim`.  
    Evidence: `external/MCLN/Model-art/model-art.py:L220-L227`.
  - Taobao: `2 * emb_dim`.  
    Evidence: `external/MCLN/Model-taobao/model-taobao.py:L210-L216`.
- **Scoring weight for visual branch (`lambda_m`):**
  - Beauty `0.3` (`external/MCLN/Model-beauty/model-beauty.py:L27`, `L114-L117`)
  - Taobao `0.2` (`external/MCLN/Model-taobao/model-taobao.py:L27`, `L105-L107`)
  - Art uses implicit `1.0` (no lambda scaling) (`external/MCLN/Model-art/model-art.py:L113-L117`).
