# DICE Developer Blueprint (Standalone, Code-First, Reconstruction-Ready)

## Mechanical Summary Table

| Spec | Implementation |
|---|---|
| Runtime entry | `app.py::main` (flags -> managers -> trainer.train() -> trainer.test()) [Ref: `src/app.py` ~L12-96] |
| Data format consumed | Preprocessed sparse/dense files (`.npz`, `.npy`, optional `.csv`, `.json`), not raw `train.txt` parsing in runtime path [Ref: `src/data_utils/loader.py` ~L14-76] |
| Main in-memory structures | SciPy `coo_matrix`, `lil_matrix`, `dok_matrix`; NumPy vectors; Torch tensors [Ref: `src/data.py` ~L50-120, `src/data_utils/transformer.py` ~L14-29] |
| Sampler stack | `Sampler`, `PointSampler`, `PairSampler`, `DICESampler` [Ref: `src/data_utils/sampler.py` ~L9-77, `src/utils.py` ~L289-373] |
| Causal families | DICE/LGNDICE (channel disentanglement + discrepancy), CausE/LGNCausE (control/treatment split + discrepancy) [Ref: `src/model.py` ~L325-603, ~L60-205] |
| GNN kernel | `LGConv.forward` with DGL `update_all(copy_u,sum)` and degree norm [Ref: `src/model.py` ~L221-270] |
| Ranking backend | FAISS `IndexFlatIP` (+ optional GPU index) [Ref: `src/candidate_generator.py` ~L60-87] |
| Optimizer/scheduler | Adam (`betas=(0.5,0.99)`, `amsgrad=True`) + ReduceLROnPlateau [Ref: `src/recommender.py` ~L55-59, `src/trainer.py` ~L80-87] |
| Eval loop | candidate generation -> history filtering -> metrics (recall/hr/ndcg) [Ref: `src/tester.py` ~L44-109, `src/metrics.py` ~L17-72] |

---

## 1) End-to-End Data Flow & Ingestion Mechanics

## 1.1 Disk-to-memory ingestion path

### Runtime orchestrator
```python
def main(argv):
    flags_obj = FLAGS
    cm = utils.ContextManager(flags_obj)
    vm = utils.VizManager(flags_obj)
    dm = utils.DatasetManager(flags_obj)
    dm.get_dataset_info()
    trainer = utils.ContextManager.set_trainer(flags_obj, cm, vm, dm)
    trainer.train()
    trainer.test()
```
[Ref: `src/app.py` ~L72-96]

### File IO abstraction
```python
class CooLoader(Loader):
    def load(self, filename, **kwargs):
        filename = os.path.join(self.load_path, filename)
        record = sp.load_npz(filename)
        return record

class NpyLoader(Loader):
    def load(self, filename, **kwargs):
        filename = os.path.join(self.load_path, filename)
        record = np.load(filename)
        return record
```
[Ref: `src/data_utils/loader.py` ~L44-76]

### Dataset manager load points
```python
def get_dataset_info(self):
    coo_record = self.coo_loader.load(const_util.train_coo_record)
    self.n_user = coo_record.shape[0]
    self.n_item = coo_record.shape[1]
    self.coo_record = coo_record

def get_skew_dataset(self):
    self.skew_coo_record = self.coo_loader.load(const_util.train_skew_coo_record)

def get_popularity(self):
    self.popularity = self.npy_loader.load(const_util.popularity)
    return self.popularity
```
[Ref: `src/utils.py` ~L250-270]

### Graph storage type
- Stored/loaded as SciPy COO (`sp.load_npz`).
- Converted to DGL graph for LGN variants.
- Not stored as dense adjacency matrix in runtime path.
[Ref: `src/data_utils/loader.py` ~L44-51, `src/recommender.py` ~L171-196, ~L237-257]

### Raw text (`train.txt`) path check
- No runtime function converts text interaction files to tensors in this repository execution path.
- Runtime consumes prebuilt `.npz`/`.npy` artifacts only.
[Ref: `src/config/const.py` ~L7-20, `src/data_utils/loader.py` ~L44-76]

## 1.2 Sampling and mini-batching engine

### Core negative sampler (CPU loop)
```python
def generate_negative_samples(self, user, **kwargs):
    negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
    user_pos = self.lil_record.rows[user]
    for count in range(self.neg_sample_rate):
        item = np.random.randint(self.n_item)
        while item in user_pos or item in negative_samples:
            item = np.random.randint(self.n_item)
        negative_samples[count] = item
    return negative_samples
```
[Ref: `src/data_utils/sampler.py` ~L30-41]

### DICE hard-negative sampler (pop/unpop split)
```python
def generate_negative_samples(self, user, pos_item):
    negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
    mask_type = np.full(self.neg_sample_rate, False, dtype=np.bool)

    user_pos = self.lil_record.rows[user]
    item_pos_pop = self.popularity[pos_item]

    pop_items = np.nonzero(self.popularity > item_pos_pop + self.margin)[0]
    pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]

    unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0]
    unpop_items = np.nonzero(self.popularity < item_pos_pop/2)[0]
    unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]

    ... # branch by pool size, random draw per negative, duplicate checks

    return negative_samples, mask_type
```
[Ref: `src/utils.py` ~L303-362]

### Batch container construction
```python
class DICEFactorizationDataset(FactorizationDataset):
    def __getitem__(self, index):
        if index < len(self.sampler.record):
            users, items_p, items_n, mask = self.sampler.sample(index)
            mask = torch.BoolTensor(mask)
        else:
            users, items_p, items_n, mask = self.skew_sampler.sample(index - len(self.sampler.record))
            mask = torch.BoolTensor(mask)
        return users, items_p, items_n, mask
```
[Ref: `src/data.py` ~L211-247]

### DataLoader wiring
```python
return DataLoader(dataset,
    batch_size=flags_obj.batch_size,
    shuffle=flags_obj.shuffle,
    num_workers=flags_obj.num_workers,
    drop_last=True)
```
[Ref: `src/data.py` ~L20-42]

## 1.3 Sampling system audit
- Negative sampling is Python/NumPy loop heavy (CPU-bound), not vectorized tensor sampling.
- Hard negatives exist only in DICE sampler through popularity-conditioned candidate pools.
- No ANN or learned hard-negative miner in training loop.
[Ref: `src/data_utils/sampler.py` ~L30-41, `src/utils.py` ~L303-362]

## 1.4 Tensor Shape Lineage (single forward pass)

Assume:
- `B` = DataLoader batch size
- `r` = `neg_sample_rate`
- `d` = embedding size
- `N = n_user + n_item`
- `K` = top-k candidates at inference

### DICE / LGNDICE training pass
1. Dataset returns per-sample arrays length `r`; DataLoader stacks ->
   - `user`, `item_p`, `item_n`, `mask`: `[B, r]`
2. Embedding lookup (DICE):
   - `users_int`, `users_pop`, `items_*`: `[B, r, d]`
3. Channel scores:
   - `p_score_int`, `n_score_int`, `p_score_pop`, `n_score_pop`: `[B, r]`
4. Total scores:
   - `p_score_total`, `n_score_total`: `[B, r]`
5. Final scalar loss: `[]`

### LGNDICE additions
1. Initial embeddings:
   - `embeddings_int`, `embeddings_pop`: `[N, d]`
2. Layer stack:
   - `features_int` list length `(L+1)` each `[N, d]`
   - `torch.stack(..., dim=2)`: `[N, d, L+1]`
   - `torch.mean(..., dim=2)`: `[N, d]`
3. Gather with `user/item` ids -> `[B, r, d]`
4. Final score tensors remain `[B, r]`

### CausE / LGNCausE training pass
1. Batch tensors:
   - `user`, `item`, `label`: `[B, 1+r]`
   - `mask`: `[B]` after squeeze
2. Control/treatment boolean indexing produces variable-size first axis (`n_c`, `n_t`)
   - scores each: `[n_c, 1+r]` and `[n_t, 1+r]` (batch-dependent)
3. Loss terms scalar each, combined in trainer/recommender.

### Inference score tensor
- FAISS `index.search(users, k)` returns rank matrix `I`: `[B, K]` (indices) and optional scores `D`: `[B, K]`.
[Ref: `src/candidate_generator.py` ~L79-87]

---

## 2) Core Engine Deconstruction (Reusable Modules)

## 2.1 Disentanglement Engine (factor splitting + routing)

### Mechanism type
- Static parameter split by channel (two independent embedding tables), not chunk/slice from one tensor.
- Routing is fixed by which table is used in each term; no learned router network.

### Full code block (DICE forward routing)
```python
def forward(self, user, item_p, item_n, mask):

    users_int = self.users_int[user]
    users_pop = self.users_pop[user]
    items_p_int = self.items_int[item_p]
    items_p_pop = self.items_pop[item_p]
    items_n_int = self.items_int[item_n]
    items_n_pop = self.items_pop[item_n]

    p_score_int = torch.sum(users_int*items_p_int, 2)
    n_score_int = torch.sum(users_int*items_n_int, 2)

    p_score_pop = torch.sum(users_pop*items_p_pop, 2)
    n_score_pop = torch.sum(users_pop*items_n_pop, 2)

    p_score_total = p_score_int + p_score_pop
    n_score_total = n_score_int + n_score_pop

    loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
    loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
    loss_total = self.bpr_loss(p_score_total, n_score_total)

    item_all = torch.unique(torch.cat((item_p, item_n)))
    item_int = self.items_int[item_all]
    item_pop = self.items_pop[item_all]
    user_all = torch.unique(user)
    user_int = self.users_int[user_all]
    user_pop = self.users_pop[user_all]
    discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

    loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

    return loss
```
[Ref: `src/model.py` ~L382-419]

### Contract
- **Input**: `user,item_p,item_n,mask` all shape `[B,r]`; ids are integer tensors; mask is bool.
- **State**: 4 embedding tables (`users_int/users_pop/items_int/items_pop`) each `[n_entity,d]`.
- **Output**: scalar loss.

## 2.2 Independence Engine (causal regularizer)

### Mechanism type
- selectable: L1 / L2 / dcor.
- `dcor` implemented explicitly with pairwise distances and doubly-centered matrices.

### Full code block (`dcor`)
```python
def dcor(self, x, y):

    a = torch.norm(x[:,None] - x, p = 2, dim = 2)
    b = torch.norm(y[:,None] - y, p = 2, dim = 2)

    A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
    B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

    n = x.size(0)

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

    return dcor
```
[Ref: `src/model.py` ~L355-371 and mirrored in LGNDICE ~L468-484]

### Stability guards present
- **Present**: `degs.clamp(min=1)` in graph kernel (prevents zero-degree div/NaN).
- **Absent in dcor**: no epsilon in square-root denominator, no clamp/softplus in `dcor` denominator.
- **Potential risk**: division instability if covariance terms approach zero.
[Ref: `src/model.py` ~L245-248, ~L366-369]

## 2.3 Propagation Kernel (GNN)

### Full code block (`LGConv.forward`)
```python
def forward(self, graph, feat):

    graph = graph.local_var()
    if self._cached_h is not None:
        feat = self._cached_h
    else:
        # compute normalization
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(feat.device).unsqueeze(1)
        # compute (D^-1 A^k D)^k X
        for _ in range(self._k):
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = feat * norm

        if self.norm is not None:
            feat = self.norm(feat)

        if self._cached:
            self._cached_h = feat

    return feat
```
[Ref: `src/model.py` ~L238-269]

### Kernel characteristics
- Sparse graph traversal via DGL edge list (`update_all`), dense node feature matrix.
- No attention coefficients.
- No explicit edge weights beyond normalized degree scaling.
- Shared kernel reused by LGN/LGNCausE/LGNDICE.

---

## 3) Performance Profiling & Linux/RTX 5080 Adaptation

## 3.1 Latency sinks (code smells)

1. CPU negative sampling loops with Python `while` and list-membership checks.
   - `Sampler.generate_negative_samples` and `DICESampler.generate_negative_samples`.
   - Bottleneck grows with dense users and high `r`.
   - [Ref: `src/data_utils/sampler.py` ~L30-41, `src/utils.py` ~L303-362]

2. Frequent CPU-side filtering in eval.
   - `np.isin` in Python list comprehension per user.
   - [Ref: `src/tester.py` ~L93-95]

3. Host-device boundary churn.
   - Candidate generation uses NumPy embeddings from model (`detach().cpu().numpy()`) then FAISS search.
   - [Ref: `src/model.py` embedding getters ~L50-57, ~L421-429, ~L582-603; `src/candidate_generator.py` ~L60-87]

4. Pure Python metric aggregation loops.
   - [Ref: `src/metrics.py` ~L23-28, `src/tester.py` ~L100-109]

## 3.2 VRAM scaling and quadratic tensors

### Quadratic tensors
- `dcor` creates dense pairwise distance matrices:
  - `a,b,A,B` each shape `[n,n]` where `n` = unique users/items in batch subset.
- Complexity: time and memory $O(n^2)$.
- Risk on large batch-unique cardinality: high temporary VRAM.
[Ref: `src/model.py` ~L355-371, ~L468-484]

### Linear tensors
- Embedding tables scale as $O(Nd)$.
- LGN stacked features scale as $O(Nd(L+1))` per channel before mean.
[Ref: `src/model.py` ~L284-303, ~L496-513]

## 3.3 Big-O complexity

### Graph construction (preprocessing/runtime init)
- Load COO: $O(E)$.
- DGL graph node+edge insert + self-loop add: $O(N+E)$.
- Total: $O(N+E)$ time, $O(N+E)$ memory.
[Ref: `src/recommender.py` ~L171-196, ~L237-257]

### One training iteration
- Embedding lookup + dot products: $O(Brd)$.
- LGN propagation (if used): $O(L(E+N)d)$; dual stream (LGNDICE/LGNCausE) $O(2L(E+N)d)$.
- Discrepancy `dcor`: $O(n^2 d + n^2)` (dominant non-linear term).
- Backward: same order as forward for each component.

### Inference/ranking
- Exact brute-force inner-product retrieval (`IndexFlatIP`):
  - index build: $O(Id)$
  - query: $O(BId)$ for batch query size `B`.
- History filtering adds CPU post-step approx $O(BKc)$ with `np.isin`-style membership checks.
[Ref: `src/candidate_generator.py` ~L71-87, `src/tester.py` ~L62-95]

---

## 4) Architectural Anomalies & Hyperparameters

## 4.1 Symmetry breaks / biased logic

1. DICE popularity channel uses asymmetric pair ordering:
```python
loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
```
- Positive/negative roles intentionally flipped depending on `mask`.
[Ref: `src/model.py` ~L402-404 and LGNDICE ~L548-550]

2. DICE hard-negative unpop threshold overwrite:
```python
unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0]
unpop_items = np.nonzero(self.popularity < item_pos_pop/2)[0]
```
- First assignment is overwritten by second line.
[Ref: `src/utils.py` ~L317-318]

3. CausE discrepancy computed over current batch unique items only, not full catalog:
```python
item_all = torch.unique(item)
item_control_factual = self.items_control[item_all]
item_control_counterfactual = self.items_treatment[item_all]
```
[Ref: `src/model.py` ~L95-99]

## 4.2 Magic Number List

### Architectural/runtime constants (code)
- `num_layers=2`, `dropout=0.2`, `embedding_size=64` defaults.
- `epochs=500`, `batch_size=128`, `neg_sample_rate=4`.
- `lr=0.001`, `min_lr=1e-4`, `weight_decay=5e-8`.
- `dis_pen=0.01`, `int_weight=0.1`, `pop_weight=0.1`.
- `margin=40`, `pool=40`, `margin_decay=0.9`, `loss_decay=0.9`.
- scheduler patience `5`, early-stop patience `3`.
- Adam internals: `betas=(0.5,0.99)`, `amsgrad=True`.
- logging cadence constants in trainer: `%1000`, `% (num_batch//5)`.
- self-loop add always-on in graph init.
[Ref: `src/app.py` ~L17-68, `src/recommender.py` ~L55-59, `src/trainer.py` ~L125-138, `src/recommender.py` ~L186-188]

### Dataset-specific constants
- `const.py` hardcodes absolute dataset roots for `ml10m` and `nf`.
- config files set model-specific overrides (e.g., many use `lr=0.01`, `embedding_size` 64/128, adaptive on DICE variants).
[Ref: `src/config/const.py` ~L7-20, `src/config/*.cfg`]

---

## 5) EDGRec Integration Interface

## 5.1 Minimal Viable Code (exactly 3 extracted functions)

### MVC-1: Hard-negative causal sampler (`DICESampler.generate_negative_samples`)
```python
def generate_negative_samples(self, user, pos_item):

    negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
    mask_type = np.full(self.neg_sample_rate, False, dtype=np.bool)

    user_pos = self.lil_record.rows[user]

    item_pos_pop = self.popularity[pos_item]

    pop_items = np.nonzero(self.popularity > item_pos_pop + self.margin)[0]
    pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
    num_pop_items = len(pop_items)

    unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0]
    unpop_items = np.nonzero(self.popularity < item_pos_pop/2)[0]
    unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
    num_unpop_items = len(unpop_items)

    if num_pop_items < self.pool:
        
        for count in range(self.neg_sample_rate):

            index = np.random.randint(num_unpop_items)
            item = unpop_items[index]
            while item in negative_samples:
                index = np.random.randint(num_unpop_items)
                item = unpop_items[index]

            negative_samples[count] = item
            mask_type[count] = False

    elif num_unpop_items < self.pool:
        
        for count in range(self.neg_sample_rate):

            index = np.random.randint(num_pop_items)
            item = pop_items[index]
            while item in negative_samples:
                index = np.random.randint(num_pop_items)
                item = pop_items[index]

            negative_samples[count] = item
            mask_type[count] = True
    
    else:

        for count in range(self.neg_sample_rate):

            if np.random.random() < 0.5:

                index = np.random.randint(num_pop_items)
                item = pop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]

                negative_samples[count] = item
                mask_type[count] = True

            else:

                index = np.random.randint(num_unpop_items)
                item = unpop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]

                negative_samples[count] = item
                mask_type[count] = False

    return negative_samples, mask_type
```
[Ref: `src/utils.py` ~L303-362]

### MVC-2: Causal disentanglement head (`DICE.forward`)
```python
def forward(self, user, item_p, item_n, mask):

    users_int = self.users_int[user]
    users_pop = self.users_pop[user]
    items_p_int = self.items_int[item_p]
    items_p_pop = self.items_pop[item_p]
    items_n_int = self.items_int[item_n]
    items_n_pop = self.items_pop[item_n]

    p_score_int = torch.sum(users_int*items_p_int, 2)
    n_score_int = torch.sum(users_int*items_n_int, 2)

    p_score_pop = torch.sum(users_pop*items_p_pop, 2)
    n_score_pop = torch.sum(users_pop*items_n_pop, 2)

    p_score_total = p_score_int + p_score_pop
    n_score_total = n_score_int + n_score_pop

    loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
    loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
    loss_total = self.bpr_loss(p_score_total, n_score_total)

    item_all = torch.unique(torch.cat((item_p, item_n)))
    item_int = self.items_int[item_all]
    item_pop = self.items_pop[item_all]
    user_all = torch.unique(user)
    user_int = self.users_int[user_all]
    user_pop = self.users_pop[user_all]
    discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

    loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

    return loss
```
[Ref: `src/model.py` ~L382-419]

### MVC-3: Independence regularizer kernel (`DICE.dcor`)
```python
def dcor(self, x, y):

    a = torch.norm(x[:,None] - x, p = 2, dim = 2)
    b = torch.norm(y[:,None] - y, p = 2, dim = 2)

    A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
    B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

    n = x.size(0)

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

    return dcor
```
[Ref: `src/model.py` ~L355-371]

## 5.2 EDGRec wrapper module API (recommended)

### Interface definition

```python
class EDGRecCausalHead(nn.Module):
    def __init__(self, d, dis_loss: str, dis_pen: float,
                 int_weight: float, pop_weight: float):
        ...

    def forward(self,
                user_ids: torch.LongTensor,      # [B, r]
                pos_item_ids: torch.LongTensor,  # [B, r]
                neg_item_ids: torch.LongTensor,  # [B, r]
                pop_mask: torch.BoolTensor,      # [B, r]
                user_repr_int: torch.Tensor,     # [B, r, d]
                user_repr_pop: torch.Tensor,     # [B, r, d]
                pos_repr_int: torch.Tensor,      # [B, r, d]
                pos_repr_pop: torch.Tensor,      # [B, r, d]
                neg_repr_int: torch.Tensor,      # [B, r, d]
                neg_repr_pop: torch.Tensor       # [B, r, d]
                ) -> dict:
        # returns {
        #   'loss_total': scalar,
        #   'loss_int': scalar,
        #   'loss_pop': scalar,
        #   'loss_bpr': scalar,
        #   'loss_dis': scalar,
        #   'scores_pos': [B,r],
        #   'scores_neg': [B,r]
        # }
        ...
```

### State requirements
- Channel embeddings (or channel projections) for users/items.
- Discrepancy criterion (`L1`, `L2`, `dcor`).
- Weights (`int_weight`, `pop_weight`, `dis_pen`).

### Encoder compatibility
- Works with:
  - MF encoder outputs (direct embeddings), or
  - GNN encoder outputs (`[N,d]` then gather by ids), including LGN-like propagation.

### Batch schema for unified runtime
```python
batch = {
  'user_ids': LongTensor[B, r],
  'pos_item_ids': LongTensor[B, r],
  'neg_item_ids': LongTensor[B, r],
  'labels': Optional[FloatTensor[B, 1+r]],
  'treatment_mask': Optional[BoolTensor[B]],
  'pop_mask': Optional[BoolTensor[B, r]],
  'sample_weight': Optional[FloatTensor[B, r]]
}
```

---

## 6) Reconstruction Checklist (without revisiting source)

1. Implement loaders that read COO `.npz` + popularity `.npy`.
2. Convert COO -> LIL/DOK for sampling.
3. Implement pair/point/DICE samplers exactly as above.
4. Implement DICE head (`forward`) + discrepancy selector + `dcor`.
5. Implement LGConv kernel exactly (degree norm + `update_all`).
6. Build trainer shell with Adam + plateau scheduler + early stop.
7. Export embeddings to FAISS `IndexFlatIP` for ranking.
8. Apply history filter before metrics.
9. Keep same hyperparameters first; optimize only after parity checks.

This blueprint contains all required mechanics to reconstruct DICE-family causal embedding generation and integrate it into a unified EDGRec implementation stack.
