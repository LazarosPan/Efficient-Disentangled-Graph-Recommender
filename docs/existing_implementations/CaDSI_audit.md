# CaDSI Developer Blueprint (Standalone Code-First, U-CaGNN Integration Ready)

## Mechanical Summary Table

| Dimension | Mechanical Spec | Evidence |
|---|---|---|
| Runtime | TensorFlow 1.x graph mode (`disable_eager_execution`), session-based train/eval | `external/CaDSI/CaDSI/CaDSI.py` ~L1-14 |
| Data core | Text interactions + 4 side embedding files -> sparse matrix + dict maps + dense side arrays | `external/CaDSI/CaDSI/utility/load_data.py` ~L7-264 |
| Graph form | Static bipartite adjacency `[0, R; R^T, 0]` in SciPy sparse | `external/CaDSI/CaDSI/utility/load_data.py` ~L298-335 |
| Encoder | Iterative factor routing over sparse graph with factor-specific sparse tensors | `external/CaDSI/CaDSI/CaDSI.py` ~L189-296, L375-416 |
| Causal/independence | Distance Correlation penalty across factor blocks | `external/CaDSI/CaDSI/CaDSI.py` ~L321-373 |
| Ranker | Dot-product scoring `U_batch @ I_batch^T` | `external/CaDSI/CaDSI/CaDSI.py` ~L147-148 |
| Objective | `BPR + L2 + corDecay * dCor` (when `corDecay >= 1e-9`) | `external/CaDSI/CaDSI/CaDSI.py` ~L153-159, L298-319 |
| Evaluation | Full-catalog batched scoring + Python heap ranking + multiprocessing | `external/CaDSI/CaDSI/utility/batch_test.py` ~L128-205 |
| Critical coupling | Global `args`, `data_generator` imported across modules; constructor uses global `config` | `external/CaDSI/CaDSI/utility/batch_test.py` ~L11-16; `external/CaDSI/CaDSI/CaDSI.py` ~L48-94, L455+ |

---

## 1) End-to-End Data Flow & Ingestion Mechanics

### 1.1 Raw Data Ingestion (Disk -> In-memory)

### Code path
- `Data.__init__(path, batch_size)` is the ingestion root.
- Reads:
  - `train.txt`, `test.txt`
  - `author.txt`, `publisher.txt`, `user.txt`, `year.txt`

### Disk-to-structure transformation
1. Parse train/test lines into user/item IDs.
2. Build scalar stats (`n_users`, `n_items`, `n_train`, `n_test`).
3. Build sparse interaction matrix:
   - `self.R = sp.dok_matrix((n_users, n_items), dtype=np.float32)`
4. Build maps:
   - `self.train_items[uid] = [item...]`
   - `self.test_set[uid] = [item...]`
5. Parse side files into dicts and arrays:
   - user-side arrays: `A1_vec_user ... A4_vec_user`
   - aspect-side arrays: `A1_vec_aspect ... A4_vec_aspect`

### Data representation type
- Interaction and graph: **SciPy sparse** (`dok`, `lil`, `csr`, `coo`) 
- Train/test user-item index: **Python dict/list**
- Side embeddings: **NumPy dense float32 arrays**

### Hard filter behavior
- Users filtered with `uid < 12771` in multiple branches.

### Evidence
- `external/CaDSI/CaDSI/utility/load_data.py` ~L10-80, L53, L64-80, L83-264

---

### 1.2 Graph Construction Pipeline

### Code path
- `Data.get_adj_mat()`
- `Data.create_adj_mat()`

### Mechanics
1. Attempt load cached sparse files:
   - `s_adj_mat.npz`, `s_norm_adj_mat.npz`, `s_mean_adj_mat.npz`, `s_pre_adj_mat.npz`
2. If missing, create from `R`:
   - `adj[:U, U:] = R`
   - `adj[U:, :U] = R.T`
3. Produce normalized variants via row normalization and symmetric pre-normalization.
4. Return `adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat`.

### Representation
- Stored as sparse SciPy matrices (primarily CSR/COO).

### Complexity
- Build/normalize sparse graph: $O(E)$ time, $O(N+E)$ memory.

### Evidence
- `external/CaDSI/CaDSI/utility/load_data.py` ~L266-335

---

### 1.3 Sampling & Batching Engine

### Training batch sampler (exact mechanism)
- `Data.sample()`:
  - user sampling with `rd.sample` / `rd.choice`
  - positive item sample: random index in user positives
  - negative item sample: rejection sampling loop with `np.random.randint`
- `Data.sample_test()` analogous for test positives and train+test exclusion negatives.

### Hard negatives
- No explicit hard-negative mining by score/popularity/embedding distance.
- Negatives are random non-interacted IDs via rejection sampling.
- Optional `negative_pool()` exists but not used in main loop.

### CPU vs vectorization audit
- Sampling is **Python-loop + while-loop rejection**, CPU-bound and non-vectorized.

### Evidence
- `external/CaDSI/CaDSI/utility/load_data.py` ~L337-426
- Training uses `data_generator.sample()` in loop: `external/CaDSI/CaDSI/CaDSI.py` ~L567-581

---

### 1.4 Tensor Shape Lineage (Single Forward Pass)

Symbols:
- $B$ training batch size
- $U$ users, $I$ items, $N=U+I$
- $d$ embedding size
- $F$ factors
- $C$ correlation sample size
- $B_u$ eval user batch, $B_i$ eval item batch

### Input IDs
- `users`: `[B]`
- `pos_items`: `[B]`
- `neg_items`: `[B]`
- `cor_users`: `[C]`
- `cor_items`: `[C]`

### Initial embeddings
- `user_embedding`: `[U, d]`
- `item_embedding`: `[I, d]`
- `ego_embeddings = concat([user, item], axis=0)`: `[N, d]`

### Factorized / causal latents
- `ego_layer_embeddings = tf.split(ego_embeddings, F, axis=1)` -> list of `F` tensors `[N, d/F]`
- Batch-level factorized latent view (conceptual): `[F, B, d/F]`
  - obtained by selecting user/item IDs from each factor block.

### Final score tensor
- Training score vectors:
  - `pos_scores`: `[B]`
  - `neg_scores`: `[B]`
- Ranking score matrix:
  - `batch_ratings = matmul(U_batch, I_batch^T)`: `[B_u, B_i]`
  - full-catalog effective shape in eval assembly: `[B_u, I]`

### Evidence
- `external/CaDSI/CaDSI/CaDSI.py` ~L116-148, L205-221, L293-308
- `external/CaDSI/CaDSI/utility/batch_test.py` ~L149-171

---

## 2) Core Engine Deconstruction (Reusable Modules)

## 2.1 Disentanglement Engine (Full Code Block)

### Source function
- `CaDSI._disentangle_intent_learning(self, pick_=False)`
- Evidence: `external/CaDSI/CaDSI/CaDSI.py` ~L189-296

```python
def _disentangle_intent_learning(self, pick_ = False):
    '''
    pick_ : True, the model would narrow the weight of the least important factor down to 1/args.pick_scale.
    pick_ : False, do nothing.
    '''
    p_test = False
    p_train = False

    A_values = tf.ones(shape=[self.n_factors, len(self.all_h_list)])

    ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
    all_embeddings = [ego_embeddings]
    all_embeddings_t = [ego_embeddings]

    output_factors_distribution = []

    factor_num = [self.n_factors, self.n_factors, self.n_factors]
    iter_num = [self.n_iterations, self.n_iterations, self.n_iterations]
    for k in range(0, self.n_layers):

        n_factors_l = factor_num[k]
        n_iterations_l = iter_num[k]
        layer_embeddings = []
        layer_embeddings_t = []

        ego_layer_embeddings = tf.split(ego_embeddings, n_factors_l, 1)
        ego_layer_embeddings_t = tf.split(ego_embeddings, n_factors_l, 1)

        # perform routing mechanism
        for t in range(0, n_iterations_l):
            iter_embeddings = []
            iter_embeddings_t = []
            A_iter_values = []

            if t == n_iterations_l - 1:
                p_test = pick_
                p_train = False

            A_factors, D_col_factors, D_row_factors = self._change_values_to_factors_with_P(n_factors_l, A_values, pick= p_train)
            A_factors_t, D_col_factors_t, D_row_factors_t = self._change_values_to_factors_with_P(n_factors_l, A_values, pick= p_test)
            for i in range(0, n_factors_l):

                factor_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], ego_layer_embeddings[i])
                factor_embeddings_t = tf.sparse.sparse_dense_matmul(D_col_factors_t[i], ego_layer_embeddings_t[i])

                factor_embeddings_t = tf.sparse.sparse_dense_matmul(A_factors_t[i], factor_embeddings_t)
                factor_embeddings = tf.sparse.sparse_dense_matmul(A_factors[i], factor_embeddings)

                factor_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], factor_embeddings)
                factor_embeddings_t = tf.sparse.sparse_dense_matmul(D_col_factors_t[i], factor_embeddings_t)

                iter_embeddings.append(factor_embeddings)
                iter_embeddings_t.append(factor_embeddings_t)

                if t == n_iterations_l - 1:
                    layer_embeddings = iter_embeddings
                    layer_embeddings_t = iter_embeddings_t

                head_factor_embedings = tf.nn.embedding_lookup(factor_embeddings, self.all_h_list)
                tail_factor_embedings = tf.nn.embedding_lookup(ego_layer_embeddings[i], self.all_t_list)

                head_factor_embedings = tf.math.l2_normalize(head_factor_embedings, axis=1)
                tail_factor_embedings = tf.math.l2_normalize(tail_factor_embedings, axis=1)

                A_factor_values = tf.reduce_sum(tf.multiply(head_factor_embedings, tf.tanh(tail_factor_embedings)), axis=1)

                # update the attentive weights
                A_iter_values.append(A_factor_values)

            # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
            A_iter_values = tf.stack(A_iter_values, 0)
            # add all layer-wise attentive weights up.
            A_values += A_iter_values

            if t == n_iterations_l - 1:
                output_factors_distribution.append(A_factors)

        # sum messages of neighbors, [n_users+n_items, embed_size]
        side_embeddings = tf.concat(layer_embeddings, 1)
        side_embeddings_t = tf.concat(layer_embeddings_t, 1)

        ego_embeddings = side_embeddings
        ego_embeddings_t = side_embeddings_t
        # concatenate outputs of all layers
        all_embeddings_t += [ego_embeddings_t]
        all_embeddings += [ego_embeddings]

    all_embeddings = tf.stack(all_embeddings, 1)
    all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

    all_embeddings_t = tf.stack(all_embeddings_t, 1)
    all_embeddings_t = tf.reduce_mean(all_embeddings_t, axis=1, keepdims=False)

    u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
    u_g_embeddings_t, i_g_embeddings_t = tf.split(all_embeddings_t, [self.n_users, self.n_items], 0)

    return u_g_embeddings, i_g_embeddings, output_factors_distribution, u_g_embeddings_t, i_g_embeddings_t
```

### Mechanism classification
- Not static slicing-only.
- Not per-factor learnable linear projection.
- **Iterative routing algorithm** with edge-factor score updates.

### Contract
- Input state: base user/item embeddings `[U,d]`,`[I,d]`, edge lists (`all_h_list`, `all_t_list`), factors/layers/iterations.
- Output:
  - final user/item embeddings `[U,d]`,`[I,d]`
  - factor distribution snapshot list (sparse factor tensors)
  - optional pick-path embeddings for ranking head.

---

## 2.2 Propagation Kernel (Full Code Block)

### Source function
- `CaDSI._change_values_to_factors_with_P(self, f_num, A_factor_values, pick=True)`
- Evidence: `external/CaDSI/CaDSI/CaDSI.py` ~L375-416

```python
def _change_values_to_factors_with_P(self, f_num, A_factor_values, pick=True):

    A_factors = []
    D_col_factors = []
    D_row_factors = []
    # get the indices of adjacency matrix.
    A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
    D_indices = np.mat([list(range(self.n_users+self.n_items)), list(range(self.n_users+self.n_items))]).transpose()
    if pick:
        A_factor_scores = tf.nn.softmax(A_factor_values, 0)
        min_A = tf.reduce_min(A_factor_scores, 0)
        index = A_factor_scores > (min_A + 0.0000001)
        index = tf.cast(index, tf.float32)*(self.pick_level-1.0) + 1.0  # adjust the weight of the minimum factor to 1/self.pick_level

        A_factor_scores = A_factor_scores * index
        A_factor_scores = A_factor_scores / tf.reduce_sum(A_factor_scores, 0)
    else:
        A_factor_scores = tf.nn.softmax(A_factor_values, 0)

    for i in range(0, f_num):
        # in the i-th factor, couple the adjacency values with the adjacency indices
        # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
        A_i_scores = A_factor_scores[i]
        A_i_tensor = tf.SparseTensor(A_indices, A_i_scores, self.A_in_shape)

        # get the degree values of A_i_tensor
        # .... D_i_scores_col is [n_users+n_items, 1]
        # .... D_i_scores_row is [1, n_users+n_items]
        D_i_col_scores = 1/tf.math.sqrt(tfv1.sparse_reduce_sum(A_i_tensor, axis=1)+0.00001)
        D_i_row_scores = 1/tf.math.sqrt(tfv1.sparse_reduce_sum(A_i_tensor, axis=0)+0.00001)

        # couple the laplacian values with the adjacency indices
        # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
        D_i_col_tensor = tf.SparseTensor(D_indices, D_i_col_scores, self.A_in_shape)
        D_i_row_tensor = tf.SparseTensor(D_indices, D_i_row_scores, self.A_in_shape)

        A_factors.append(A_i_tensor)
        D_col_factors.append(D_i_col_tensor)
        D_row_factors.append(D_i_row_tensor)

    # return a (n_factors)-length list of laplacian matrix
    return A_factors, D_col_factors, D_row_factors
```

### Kernel properties
- Sparse ops used directly:
  - `tf.SparseTensor`
  - `tfv1.sparse_reduce_sum`
  - `tf.sparse.sparse_dense_matmul` (called by disentanglement engine)
- Edge weights are factor-specific (`A_i_scores`).
- Attention-like behavior via factor softmax over edge-factor scores.

---

## 2.3 Independence Engine (Full Code Blocks)

### Source functions
- `create_loss_function_cor(...)`
- `_create_distance_correlation(...)`
- Evidence: `external/CaDSI/CaDSI/CaDSI.py` ~L321-373

```python
def create_loss_function_cor(self, cor_u_embeddings, cor_i_embeddings):
    cor_loss = tf.constant(0.0, tf.float32)

    if self.cor_flag == 0:
        return  cor_loss

    ui_embeddings = tf.concat([cor_u_embeddings, cor_i_embeddings], axis=0)
    ui_factor_embeddings = tf.split(ui_embeddings, self.n_factors, 1)

    for i in range(0, self.n_factors-1):
        x = ui_factor_embeddings[i]
        y = ui_factor_embeddings[i+1]
        cor_loss += self._create_distance_correlation(x, y)

    cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)

    return cor_loss
```

```python
def _create_distance_correlation(self, X1, X2):

    def _create_centered_distance(X):

        r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
        D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

        D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
            + tf.reduce_mean(D)
        return D

    def _create_distance_covariance(D1, D2):
        # calculate distance covariance between D1 and D2
        n_samples = tf.dtypes.cast(tf.shape(D1)[0], tf.float32)
        dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
        return dcov

    D1 = _create_centered_distance(X1)
    D2 = _create_centered_distance(X2)

    dcov_12 = _create_distance_covariance(D1, D2)
    dcov_11 = _create_distance_covariance(D1, D1)
    dcov_22 = _create_distance_covariance(D2, D2)

    # calculate the distance correlation
    dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
    return dcor
```

### Stability guards found
- `+1e-8` in distance matrix sqrt and covariance sqrt
- `+1e-10` in final dCor denominator
- `tf.maximum(..., 0.0)` before sqrt
- `+0.00001` in degree normalization (propagation kernel)
- `tf.nn.softplus` in BPR objective

### Quadratic memory/time hotspot
- Dense pairwise matrix `D`: shape `[n_samples, n_samples]` -> $O(n_{samples}^2)$.

---

## 3) Performance Profiling & Hardware Adaptation (Linux + RTX 5080)

## 3.1 Latency sinks (code smells)
1. Python-loop negative sampling with rejection while-loops.
2. Per-user ranking in Python dict + heapq.
3. Dense host-side `rate_batch` assembly `[batch_users, ITEM_NUM]`.
4. Multiprocessing overhead for metric aggregation.
5. Global object creation at import time (startup coupling, hard to profile cleanly).

### Evidence
- Sampling: `external/CaDSI/CaDSI/utility/load_data.py` ~L345-426
- Ranking/eval: `external/CaDSI/CaDSI/utility/batch_test.py` ~L20-205
- Import-time globals: `external/CaDSI/CaDSI/utility/batch_test.py` ~L11-16

---

## 3.2 VRAM scaling audit

### Linear scaling tensors
- Embeddings: `[U,d]`, `[I,d]` -> $O((U+I)d)$
- Factor edge scores / sparse factors -> approximately $O(FE)$ storage overhead

### Quadratic scaling tensors
- dCor centered distance matrix `D`: $O(C^2)$ memory/time component
- If $C$ grows aggressively, this can dominate memory even on high-end GPU

### Evidence
- Embeddings: `external/CaDSI/CaDSI/CaDSI.py` ~L173-177
- dCor matrix: `external/CaDSI/CaDSI/CaDSI.py` ~L349-353
- Eval dense matrix: `external/CaDSI/CaDSI/utility/batch_test.py` ~L151

---

## 3.3 Formal Big-O

### Graph construction (pre-processing)
- Build `R` + adjacency + normalization: $O(E)$ time, $O(N+E)$ memory.

### Training iteration
Let $L$ layers, $T$ routing iterations, $F$ factors, $d$ dim, batch size $B$, correlation sample $C$.
- Routing propagation: $O(L\cdot T\cdot E\cdot d)$
- BPR + L2: $O(B\cdot d)$
- dCor: $O((F-1)\cdot C^2\cdot d/F)$ and $O(C^2)$ memory
- Total dominant: $O(LTEd + BCd + C^2d)$ (routing + dCor dominated)

### Inference/ranking
- Score compute: $O(|U_{test}|\cdot I\cdot d)$
- Ranking with heapq: per user approx $O(I + I\log K)$

### Evidence
- Routing kernel and loop: `external/CaDSI/CaDSI/CaDSI.py` ~L213-271
- dCor: `external/CaDSI/CaDSI/CaDSI.py` ~L321-373
- Eval loop: `external/CaDSI/CaDSI/utility/batch_test.py` ~L142-191

---

## 4) Architectural Anomalies & Hyperparameters

## 4.1 Symmetry breaks / biased logic
1. BPR negative A4 term uses positive tensor (`aspect_pos_scores_A4`) in `neg_scores` expression.
2. Constructor references global `config` instead of passed `data_config` for many fields.
3. `self.A_values` placeholder declared, but routing uses local `A_values = tf.ones(...)`.
4. `model_save` uses `model.weights[...]` inside instance method instead of `self.weights[...]`.
5. Evaluation non-batch branch references missing model attrs (`_1`, `print_pick`, `print_embed`).

### Evidence
- `external/CaDSI/CaDSI/CaDSI.py` ~L307-308, L48-94, L109 vs L197, L341-342
- `external/CaDSI/CaDSI/utility/batch_test.py` ~L173-179

---

## 4.2 Magic Number List (hardcoded)

### Architectural / training defaults (CLI)
- `epoch=2`, `embed_size=64`, `batch_size=1024`, `lr=0.01`
- `n_layers=1`, `n_factors=4`, `n_iterations=2`
- `cor_flag=1`, `corDecay=0.0`
- `pick=0`, `pick_scale=1e10`
- `regs='[1e-3,1e-4,1e-4]'`
- `show_step=15`, `early=40`, `Ks='[20,40,60,80,100]'`

### In-code constants
- user filter: `uid < 12771` (dataset-specific shortcut)
- negative pool size: `100`
- numerical eps: `1e-8`, `1e-10`, `1e-5`, threshold `1e-9`
- CPU workers: `multiprocessing.cpu_count() // 2`

### Evidence
- CLI: `external/CaDSI/CaDSI/utility/parser.py` ~L5-62
- Data filter/pool: `external/CaDSI/CaDSI/utility/load_data.py` ~L30, L64, L79, L341
- Stability values: `external/CaDSI/CaDSI/CaDSI.py` ~L153, L349, L370, L402-403
- Workers: `external/CaDSI/CaDSI/utility/batch_test.py` ~L10

---

## 5) U-CaGNN Integration Interface

## 5.1 Minimal Viable Code (MVC): 3 Functions to Replicate Core Causal Effect

### MVC Function 1: Disentanglement + routing
- Extract `CaDSI._disentangle_intent_learning` (full block in Section 2.1).
- Why: this is the causal channel generator and graph encoder backbone.

### MVC Function 2: Factor sparse kernel builder
- Extract `CaDSI._change_values_to_factors_with_P` (full block in Section 2.2).
- Why: this defines per-factor edge weighting and normalized sparse propagation operators.

### MVC Function 3: Independence regularizer
- Extract `CaDSI._create_distance_correlation` + wrapper `create_loss_function_cor` (full blocks in Section 2.3).
- Why: this is the causal disentanglement pressure term.

---

## 5.2 Unified Module API (PyTorch/DGL/PyG wrapper contract)

### `UCaGNNDataAdapter`
- **Input**: raw paths (`train`, `test`, optional side embeddings)
- **Output state**:
  - `num_users`, `num_items`
  - sparse bipartite graph (`edge_index`, optional edge_weight)
  - train/test user-item maps
  - optional side feature tensors
- **State**: cached sparse artifacts

### `FactorRoutingEncoder`
- **Input**:
  - base node embeddings `[N,d]`
  - sparse graph edge index / edge weights
  - hyperparams `{F,L,T,pick,pick_scale}`
- **Output**:
  - node embeddings `[N,d]`
  - optional routing diagnostics per factor/iteration
- **Internal state**:
  - routing logits/weights per edge-factor

### `IndependenceRegularizer`
- **Input**: sampled node embeddings `[C,d]`, factor count `F`
- **Output**: scalar loss
- **Modes**: `dcor` primary; extensible to `hsic`/contrastive in unified architecture

### `RankingHead`
- **Input**: user repr `[B_u,d]`, item repr `[B_i,d]`
- **Output**: scores `[B_u,B_i]`
- **Default**: dot product; optional MLP head in unified system

### `TrainerOrchestrator`
- **Input**: batch IDs + module outputs
- **Output**: optimize
- **Objective contract**:
  - $L = L_{rank} + \lambda_{reg}L_2 + \lambda_{indep}L_{indep}$

---

## 6) Reconstruction Checklist (Use This Instead of Original Repo)

1. Implement data adapter that outputs sparse bipartite graph + train/test maps.
2. Implement factor-routing encoder exactly with iterative edge-factor updates.
3. Implement sparse factor kernel builder with softmax factor routing and degree normalization.
4. Implement BPR objective and dot-product ranker.
5. Implement dCor regularizer with pairwise centered distance matrices.
6. Compose objective with configurable `lambda` values.
7. Replace Python-loop samplers and evaluators with vectorized/GPU versions for scale.
8. Remove all global module state; pass explicit objects/configs only.
9. Validate shape lineage at each stage using the shape ledger in Section 1.4.

This checklist is the independent reconstruction path for U-CaGNN integration.
