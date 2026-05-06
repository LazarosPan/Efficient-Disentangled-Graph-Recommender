# CausE — Standalone Developer Blueprint (Code-First, Repository-Derived)

## Mechanical Summary Table

| Spec | Extracted implementation | Evidence (file:line-range) |
|---|---|---|
| Framework/runtime | TensorFlow 1.x static graph (`tf.placeholder`, `tf.Session`, `tf.app.flags`) | `external/CausE/src/models.py:30-40, 93-110`; `external/CausE/src/causal_prod2vec.py:14-69`; `external/CausE/src/causal_prod2vec2i.py:14-70` |
| Task form | Pointwise binary classification on `(user_id, product_id, label)` triples | `external/CausE/src/utils.py:10-27, 29-47`; `external/CausE/src/models.py:82-92` |
| Core backbone | Matrix-factorization-style embedding lookup + dot product + biases | `external/CausE/src/models.py:44-65` |
| Causal mechanism | Counterfactual embedding alignment loss (`l1`/`l2`/`cos`) | `external/CausE/src/models.py:146-169, 175-197` |
| Causal variants | `adapt_0` (single control anchor id=0), `adapt_2i` (paired item IDs in doubled item table) | `external/CausE/src/models.py:149-151, 178-180`; `external/CausE/src/causal_prod2vec2i.py:42, 115` |
| Graph/GNN kernel | Not implemented (no adjacency build, no message passing op in train path) | `external/CausE/src/models.py:41-65`; `external/CausE/src/causal_prod2vec*.py` |
| Sampling | No negative sampling in training path; bootstrap resampling for evaluation only | `external/CausE/src/causal_prod2vec.py:107-115`; `external/CausE/src/utils.py:48-55, 104-139` |
| Device placement | Training explicitly pinned to CPU (`with tf.device('/cpu:0')`) | `external/CausE/src/causal_prod2vec.py:47, 108, 158`; `external/CausE/src/causal_prod2vec2i.py:48, 111, 171` |

---

## 1) End-to-End Data Flow & Ingestion Mechanics

### 1.1 Raw data ingestion path (disk -> memory -> tensors)

### A) Runtime training/evaluation ingestion (`src/utils.py`)

```python
def load_train_dataset(dataset_location, batch_size, num_epochs):
    record_defaults = [[1], [1], [0.]]
    dataset = tf.data.TextLineDataset(dataset_location).map(
        lambda line: tf.decode_csv(line, record_defaults=record_defaults)
    )
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    dataset = dataset.cache()
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    user_batch, product_batch, label_batch = iterator.get_next()
    label_batch = tf.expand_dims(label_batch, 1)
    return user_batch, product_batch, label_batch
```

Evidence: `external/CausE/src/utils.py:10-27`

- Source files: CSV, 3 columns per row.
- In-memory train representation: TensorFlow dataset pipeline; batch tensors only.
- Stored graph format: **none** in runtime path (no adjacency list/sparse graph object).

```python
def load_test_dataset(dataset_location):
    user_list = []
    product_list = []
    labels = []
    with open(dataset_location, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            user_list.append(row[0])
            product_list.append(row[1])
            labels.append(row[2])
    labels = np.reshape(labels, [-1, 1])
    cr = compute_empircal_cr(labels)
    return user_list, product_list, labels, cr
```

Evidence: `external/CausE/src/utils.py:29-47`

- In-memory test/val representation: Python lists + NumPy arrays.
- Note: mixed data path (TF tensors for train vs Python/NumPy for eval).

### B) Offline preprocessing utilities (`src/Data/dataset_loading.py`)

```python
view_matrix = np.zeros(shape=(num_unique_users, num_unique_products))
view_matrix[:] = np.NAN
for i in range(userid.shape[0]):
    if rating[i] < ratings_threshold:
        view_matrix[userid[i], productid[i]] = 0
    elif rating[i] >= ratings_threshold:
        view_matrix[userid[i], productid[i]] = 1
    np.savetxt('ML_view.txt', view_matrix)
```

Evidence: `external/CausE/src/Data/dataset_loading.py:53-79`

- Matrix storage type in this utility: dense NumPy matrix.
- `csr_matrix` is imported but unused (`line 7`).
- This utility is not called by training scripts directly.

---

### 1.2 Sampling & batching engine extraction

### Training mini-batching
- Implemented via TF Dataset API (`shuffle/batch/repeat`) in `load_train_dataset`.
- No explicit negative sample generator.

Evidence: `external/CausE/src/utils.py:10-27`

### Evaluation sampling
- Bootstrap subsampling only:

```python
def generate_bootstrap_batch(seed, data_set_size):
    random.seed(seed)
    ids = [random.randint(0, data_set_size-1) for j in range(int(data_set_size*0.8))]
    return ids
```

Evidence: `external/CausE/src/utils.py:48-55`

### System audit conclusion
- Batching: mostly vectorized in TF for training.
- Sampling loops: Python-loop based for bootstrap ID generation.
- Negative sampling: **not present**.
- Hard negatives: **not present**.

---

### 1.3 Tensor shape lineage (single forward pass)

Let $B$ = batch size, $U$ = num users, $P$ = num products, $d$ = embedding size.

#### Input IDs
- `user_list_placeholder`: `[B]` int32
- `product_list_placeholder`: `[B]` int32
- `reg_list_placeholder`: `[B]` int32 (2i only)

Evidence: `external/CausE/src/models.py:33-37`

#### Embeddings
- `user_embeddings`: `[U, d]`
- `product_embeddings`: `[P, d]` or `[2P, d]` for `adapt_2i`

Evidence: `external/CausE/src/models.py:46, 52`; `external/CausE/src/causal_prod2vec2i.py:42`

#### Lookup tensors
- `user_embed`: `[B, d]`
- `product_embed`: `[B, d]`
- `control_embed`:
  - `adapt_0`: `[d]` (single embedding id=0)
  - `adapt_2i`: `[B, d]` (paired ids)

Evidence: `external/CausE/src/models.py:47, 53, 151, 180`

#### Score tensors
- Interaction: `[B, 1]` via reduced dot product
- Bias term: `[B, 1]`
- `logits`: `[B, 1]`
- `prediction`: `[B, 1]`

Evidence: `external/CausE/src/models.py:61-65`

#### Candidate ranking tensor
- `Final Score Tensor [B, N_candidates]`: **not implemented in this repo**.
- Current scoring is pointwise only (`[B,1]`).

---

## 2) Core Engine Deconstruction (Reusable Modules)

### 2.1 Disentanglement engine

Status: **not implemented**.

- No factor split (`tf.split`/chunking) in model path.
- No per-factor projection heads.
- No iterative routing.

What exists instead: counterfactual alignment branch.

#### Extracted causal branch code (full)

```python
class CausalProd2Vec(SupervisedProd2vec):
    def create_control_embeddings(self):
        with tf.name_scope('control_embedding'):
            self.control_embed = tf.stop_gradient(
                tf.nn.embedding_lookup(self.product_embeddings, 0)
            )

    def create_counter_factual_loss(self):
        with tf.name_scope('counter_factual'):
            if self.cf_distance == "l1":
                self.cf_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.abs(
                            tf.subtract(
                                tf.nn.l2_normalize(self.product_embed, axis=1),
                                tf.nn.l2_normalize(self.control_embed, axis=0)
                            )
                        ), axis=1
                    )
                )
            elif self.cf_distance == "l2":
                self.cf_loss = tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            tf.subtract(
                                tf.nn.l2_normalize(self.product_embed,axis=1),
                                tf.nn.l2_normalize(self.control_embed,axis=0)
                            )
                        )
                    )
                )
            elif self.cf_distance == "cos":
                self.cf_loss = tf.losses.cosine_distance(
                    tf.nn.l2_normalize(self.control_embed,axis=0),
                    tf.nn.l2_normalize(self.product_embed,axis=1),
                    axis=0
                )
```

Evidence: `external/CausE/src/models.py:141-169`

```python
class CausalProd2Vec2i(SupervisedProd2vec):
    def create_control_embeddings(self):
        with tf.name_scope('control_embedding'):
            self.control_embed = tf.stop_gradient(
                tf.nn.embedding_lookup(self.product_embeddings, self.reg_list_placeholder)
            )

    def create_counter_factual_loss(self):
        with tf.name_scope('counter_factual'):
            if self.cf_distance == "l1":
                self.cf_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.abs(tf.subtract(self.product_embed, self.control_embed)), axis=1)
                )
            elif self.cf_distance == "l2":
                self.cf_loss = tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            tf.subtract(
                                tf.nn.l2_normalize(self.product_embed,axis=1),
                                tf.nn.l2_normalize(self.control_embed,axis=0)
                            )
                        )
                    )
                )
            elif self.cf_distance == "cos":
                self.cf_loss = tf.losses.cosine_distance(
                    tf.nn.l2_normalize(self.control_embed,axis=0),
                    tf.nn.l2_normalize(self.product_embed,axis=1),
                    axis=0
                )
```

Evidence: `external/CausE/src/models.py:170-197`

#### Contract (actual)
- Input: `product_embed [B,d]`, `control_embed [d]` (`adapt_0`) or `[B,d]` (`adapt_2i`), `cf_distance` string.
- Output: scalar `cf_loss`.

---

### 2.2 Independence engine (causal loss)

Status: implemented as distance matching; **not** HSIC/DisCo/contrastive.

Core integration:

```python
self.factual_loss = self.log_loss + reg_term + reg_term_biases
self.loss = self.factual_loss + (self.cf_pen * self.cf_loss)
```

Evidence: `external/CausE/src/models.py:87-91`

Numerical stabilizers found:
- `tf.nn.l2_normalize(...)` inside `l1/l2/cos` branches.
- No explicit epsilon constant (`+1e-10`) written in repository code.
- `alpha` initialized very small (`1e-8`) for interaction scaling.

Evidence: `external/CausE/src/models.py:60, 158-168, 191-197`

---

### 2.3 Propagation kernel (message passing)

Status: **absent** in this repository.

No code for:
- Graph adjacency construction in train path.
- Sparse message passing ops (`tf.sparse.sparse_dense_matmul`, `torch.sparse.mm`, etc.).
- Edge weights/attention/factor routing over edges.

Closest operation to propagation:
- direct embedding lookup + elementwise multiply + reduction.

```python
self.user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_list_placeholder)
self.product_embed = tf.nn.embedding_lookup(self.product_embeddings, self.product_list_placeholder)
emb_logits = self.alpha * tf.reshape(
    tf.reduce_sum(tf.multiply(self.user_embed, self.product_embed), 1),
    [tf.shape(self.user_list_placeholder)[0], 1]
)
```

Evidence: `external/CausE/src/models.py:47, 53, 61`

---

## 3) Performance Profiling & Hardware Adaptation (Linux + RTX 5080)

### 3.1 Latency sinks / code smells

1. CPU pinning of graph build and train loop:
- `with tf.device('/cpu:0')` around model setup and per-step run path.
- Likely blocks GPU acceleration on RTX 5080 unless modified.

Evidence: `external/CausE/src/causal_prod2vec.py:47, 108, 158`; `external/CausE/src/causal_prod2vec2i.py:48, 111, 171`

2. Python-loop regularization-id mapping in hot path (`adapt_2i`):

```python
for x in np.nditer(prods):
    if x >= num_products:
        reg_ids.append(x)
    elif x < num_products:
        reg_ids.append(x + num_products)
```

Evidence: `external/CausE/src/utils.py:71-76`; call site `external/CausE/src/causal_prod2vec2i.py:115`

3. Bootstrap path loops with repeated host conversions and `sess.run` calls.
- 30 rounds, each building feed dict and running model.

Evidence: `external/CausE/src/utils.py:104-139, 152-187`

4. Offline matrix writer writes full matrix inside interaction loop (`np.savetxt` per iteration).

Evidence: `external/CausE/src/Data/dataset_loading.py:68-77`

---

### 3.2 VRAM scaling and $O(N^2)$ checks

Model memory scales linearly:
- Embeddings: $O(Ud + Pd)$ (`adapt_0`), $O(Ud + 2Pd)$ (`adapt_2i`).
- Batch activations: $O(Bd)$.

No dense pairwise matrix in causal loss:
- Causal loss compares aligned vectors only; no explicit `[N,N]` affinity/kernel matrix.

Potential $O(UP)$ memory issue (preprocessing utility only):
- dense `view_matrix` allocation in dataset utility.

Evidence: `external/CausE/src/models.py:46, 52, 158-169, 187-197`; `external/CausE/src/Data/dataset_loading.py:65`

---

### 3.3 Big-O complexity

#### Graph construction / preprocessing
- Runtime training path graph construction: not graph-theoretic; model init is parameter allocation $O(Ud + Pd)$ or $O(Ud + 2Pd)$.
- Optional dense utility `create_movielens_userproduct_matrix`: memory $O(UP)$, fill loop $O(R)$ where $R$ is rating count, with pathological repeated write cost due to per-iteration `savetxt`.

Evidence: `external/CausE/src/models.py:46, 52`; `external/CausE/src/Data/dataset_loading.py:65-77`

#### Training iteration (forward/backward)
- Lookup + dot + bias + BCE + causal distance: $O(Bd)$ per batch (both forward and gradient scale with same order).
- `adapt_2i` adds Python mapping overhead $O(B)` in host code before feed.

Evidence: `external/CausE/src/models.py:47-65, 82-91, 185-197`; `external/CausE/src/causal_prod2vec2i.py:113-119`; `external/CausE/src/utils.py:66-77`

#### Inference / ranking
- Implemented inference is sampled/pointwise on provided test tuples: $O(Bd)$ per call.
- Full-sort ranking over all items is not implemented in repository.

Evidence: `external/CausE/src/causal_prod2vec.py:165-172`; `external/CausE/src/causal_prod2vec2i.py:182-194`

---

## 4) Architectural Anomalies & Hyperparameters

### 4.1 Symmetry breaks / biased logic

1. `adapt_0`: single fixed control anchor at item id `0` for all products.
- Breaks symmetry by privileging one index as universal control target.

Evidence: `external/CausE/src/models.py:149-151`

2. `adapt_2i`: one-way mapping rule.
- IDs in lower half are remapped to upper half; upper half stay unchanged.
- Asymmetric regularization pairing rule.

Evidence: `external/CausE/src/utils.py:71-76`

3. Branch-specific normalization mismatch.
- `adapt_0` L1 uses normalized vectors; `adapt_2i` L1 uses raw vectors.

Evidence: `external/CausE/src/models.py:160` vs `189`

---

### 4.2 Magic number list (hardcoded / defaults)

Training scripts:
- `num_products=1683`, `num_users=944`
- `learning_rate=1.0`
- `l2_pen=0.0`
- `num_epochs=1`
- `batch_size=512`
- `num_steps=500`
- `early_stopping_enabled=False`
- `early_stopping=200`
- `embedding_size=50`
- `cf_pen=1.0`
- `cf_distance='l1'`
- `seed=123`
- `logging_dir='/tmp/tensorboard'`
- `adapt_stat='adapt_0'` or `'adapt_2i'`

Evidence: `external/CausE/src/causal_prod2vec.py:16-33`; `external/CausE/src/causal_prod2vec2i.py:16-33`

Data/util constants:
- train shuffle buffer `10000`
- prefetch `5`
- bootstrap rounds `30`
- bootstrap sample ratio `0.8`
- matrix threshold `ratings_threshold=5`

Evidence: `external/CausE/src/utils.py:18, 20, 52, 105`; `external/CausE/src/Data/dataset_loading.py:57`

Architectural constants vs dataset-specific:
- Dataset-specific defaults: `num_users`, `num_products`.
- Architectural defaults: `embedding_size`, `cf_pen`, `cf_distance`, loss composition pattern.

---

## 5) U-CaGNN Integration Interface

### 5.1 Minimal Viable Code (MVC): three functions to replicate causal effect

### Function 1 — Pair-index generator (`adapt_2i` causal coupling)

```python
def compute_2i_regularization_id(prods, num_products):
    """Compute the ID for the regularization for the 2i approach"""
    reg_ids = []
    for x in np.nditer(prods):
        if x >= num_products:
            reg_ids.append(x)
        elif x < num_products:
            reg_ids.append(x + num_products)
    return np.asarray(reg_ids)
```

Evidence: `external/CausE/src/utils.py:66-77`

### Function 2 — Causal control embedding builder (2i mode)

```python
def create_control_embeddings(self):
    """Create the control embeddings"""
    with tf.name_scope('control_embedding'):
        self.control_embed = tf.stop_gradient(
            tf.nn.embedding_lookup(self.product_embeddings, self.reg_list_placeholder)
        )
```

Evidence: `external/CausE/src/models.py:175-181`

### Function 3 — Causal alignment loss + total loss coupling

```python
def create_counter_factual_loss(self):
    with tf.name_scope('counter_factual'):
        if self.cf_distance == "l1":
            self.cf_loss = tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.subtract(self.product_embed, self.control_embed)), axis=1)
            )
        elif self.cf_distance == "l2":
            self.cf_loss = tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        tf.subtract(
                            tf.nn.l2_normalize(self.product_embed, axis=1),
                            tf.nn.l2_normalize(self.control_embed, axis=0)
                        )
                    )
                )
            )
        elif self.cf_distance == "cos":
            self.cf_loss = tf.losses.cosine_distance(
                tf.nn.l2_normalize(self.control_embed, axis=0),
                tf.nn.l2_normalize(self.product_embed, axis=1),
                axis=0
            )

def create_losses(self):
    self.log_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label_list_placeholder)
    )
    reg_term = self.l2_pen * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.product_embeddings))
    reg_term_biases = self.l2_pen * (tf.nn.l2_loss(self.prod_b) + tf.nn.l2_loss(self.user_b))
    self.factual_loss = self.log_loss + reg_term + reg_term_biases
    self.loss = self.factual_loss + (self.cf_pen * self.cf_loss)
```

Evidence: `external/CausE/src/models.py:182-197, 77-91`

---

### 5.2 Module API for modern PyTorch/DGL/PyG wrapper

### Required Inputs

```text
batch: {
  user_id: LongTensor[B],
  item_id: LongTensor[B],
  label: FloatTensor[B,1],
  pair_item_id: LongTensor[B] | None,     # required for 2i-like mode
  exposure_domain: LongTensor[B] | None   # optional explicit domain tag
}

state: {
  user_emb: FloatTensor[U,d],
  item_emb: FloatTensor[P_or_2P,d],
  user_bias: FloatTensor[U],
  item_bias: FloatTensor[P_or_2P],
  global_bias: FloatTensor[1],
  alpha: FloatTensor[]
}
```

### Required Outputs

```text
forward_out: {
  logits: FloatTensor[B,1],
  score: FloatTensor[B,1],
  item_vec: FloatTensor[B,d],
  control_vec: FloatTensor[B,d] or FloatTensor[d]
}

loss_out: {
  factual_loss: FloatTensor[],
  causal_loss: FloatTensor[],
  total_loss: FloatTensor[]
}
```

### Stateful controls
- `cf_distance in {l1,l2,cos}`
- `cf_pen` coefficient
- `mode in {anchor_0, pair_2i}`

---

## 6) Explicit GNN Gap for U-CaGNN Planning

Repository has no native GNN module. For U-CaGNN integration:
- Keep CausE logic as a **causal regularization head** over item/user embeddings.
- Feed this head from your GNN encoder output instead of raw lookup embeddings.

Drop-in replacement contract:
- Replace `product_embed` in causal loss with `gnn_item_embed[item_id]`.
- Replace `control_embed` with paired/anchor embedding from same latent space.
- Keep total loss assembly unchanged (`factual + cf_pen * causal`).

Evidence basis for compatibility: causal loss consumes only vector pairs and does not depend on graph structure internals (`external/CausE/src/models.py:156-197`).

---

## 7) Reconstruction Checklist (No Source Revisit Needed)

1. Build pointwise scorer: user/item embeddings + biases + sigmoid logits.
2. Implement two causal modes:
   - `anchor_0`: single control vector (id=0).
   - `pair_2i`: doubled item bank + pair-id mapping.
3. Add `cf_distance` branch (`l1/l2/cos`) and `cf_pen` weighted merge.
4. Keep batch schema fixed: `user_id`, `item_id`, `label`, optional `pair_item_id`.
5. Train with minibatches; evaluate pointwise; bootstrap optional.
6. Do not assume negative sampling/hard negatives; they are absent in this implementation.
7. Treat graph propagation as external (to be supplied by U-CaGNN backbone).
