# SIGformer — Developer Blueprint (Mechanical, Code-First, Standalone)

## Mechanical Summary Table

| Spec | Extracted implementation | Evidence (file:line-range) |
|---|---|---|
| Framework/runtime | PyTorch 2.x + SciPy sparse eigensolver + PyG negative sampler + sparse grad utils | `external/SIGformer/code/model.py:1-8`; `external/SIGformer/code/dataloader.py:1-7`; `external/SIGformer/requirements.txt` |
| Data model | Signed implicit-feedback bipartite graph (user-item), split by threshold `offset` into positive and negative edges | `external/SIGformer/code/dataloader.py:14-17, 27-30, 40-43`; `external/SIGformer/code/parse.py:13` |
| Sign handling | Separate positive/negative adjacency + normalized Laplacians; mixed with scalar `alpha` | `external/SIGformer/code/dataloader.py:130-219`; `external/SIGformer/code/parse.py:27` |
| Transformer block | Custom sparse self-attention over sampled graph paths; no `nn.MultiheadAttention` | `external/SIGformer/code/model.py:11-64` |
| Sequence/path encoding | Path-sign code generated hop-by-hop (`Y = Y*2 + path_bit`) and embedded with `path_emb` | `external/SIGformer/code/dataloader.py:257-271`; `external/SIGformer/code/model.py:15, 35-37` |
| Structural bias | Optional eigenvector similarity term `q_eig · k_eig` weighted by learnable `lambda0` | `external/SIGformer/code/model.py:14, 29-34, 42-43`; `external/SIGformer/code/dataloader.py:222-235` |
| Layer stack | `n_layers` repeated Encoder blocks; final embedding is mean over layer outputs (LightGCN-style) | `external/SIGformer/code/model.py:81-105`; `external/SIGformer/code/parse.py:24` |
| Training objective | Pairwise softplus ranking with positive and negative user groups, weighted by `beta` | `external/SIGformer/code/model.py:143-187`; `external/SIGformer/code/parse.py:28` |
| Inference | Full user-item score matrix per eval batch via dense matmul + top-k | `external/SIGformer/code/model.py:118-135` |
| Core bottleneck control | Sparse edge-restricted attention (avoids dense all-pairs attention); hop count via `sample_hop` | `external/SIGformer/code/model.py:24-48`; `external/SIGformer/code/dataloader.py:261-271`; `external/SIGformer/code/parse.py:30` |

---

## 1) Hybrid Data Flow & Tensor Lineage

### 1.1 Raw graph -> signed graph tensors

#### Data ingestion and sign split

```python
train_data = pd.read_table(train_file, header=None, sep=' ')
train_pos_data = train_data[train_data[2] >= args.offset]
train_neg_data = train_data[train_data[2] < args.offset]
self.train_pos_user = torch.from_numpy(train_pos_data[0].values).to(self.device)
self.train_pos_item = torch.from_numpy(train_pos_data[1].values).to(self.device)
self.train_neg_user = torch.from_numpy(train_neg_data[0].values).to(self.device)
self.train_neg_item = torch.from_numpy(train_neg_data[1].values).to(self.device)
```

Evidence: `external/SIGformer/code/dataloader.py:14-25`

#### Signed adjacency construction (bipartite mirrored edges)

```python
self._A_pos = torch.sparse_coo_tensor(
    torch.cat([
        torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
        torch.stack([self.train_pos_item+self.num_users, self.train_pos_user])], dim=1),
    torch.ones(self.train_pos_user.shape[0]*2).to(parse.device),
    torch.Size([self.num_nodes, self.num_nodes]))

self._A_neg = torch.sparse_coo_tensor(
    torch.cat([
        torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
        torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1),
    torch.ones(self.train_neg_user.shape[0]*2).to(parse.device),
    torch.Size([self.num_nodes, self.num_nodes]))
```

Evidence: `external/SIGformer/code/dataloader.py:130-138, 173-181`

### 1.2 Sign-aware Laplacian and spectral side channel

#### Normalized signed Laplacians and mixing

```python
self._tildeA_pos = torch.sparse.mm(torch.sparse.mm(D1, self.A_pos), D2)
self._L_pos = D-self.tildeA_pos

self._tildeA_neg = torch.sparse.mm(torch.sparse.mm(D1, self.A_neg), D2)
self._L_neg = D-self.tildeA_neg

self._L = (self.L_pos+args.alpha*self.L_neg)/(1+args.alpha)
```

Evidence: `external/SIGformer/code/dataloader.py:147-170, 190-219`

#### Eigenvector features used by attention

```python
_, self._L_eigs = sp.linalg.eigs(
    sp.csr_matrix(
        (self.L._values().cpu(), self.L._indices().cpu()),
        (self.num_nodes, self.num_nodes)),
    k=args.eigs_dim,
    which='SR')
self._L_eigs = torch.tensor(self._L_eigs.real).to(parse.device)
self._L_eigs = F.layer_norm(self._L_eigs, normalized_shape=(args.eigs_dim,))
```

Evidence: `external/SIGformer/code/dataloader.py:227-235`

### 1.3 Graph-to-sequence mapping (path-wise sparse sequence generation)

This repository does not build tokenized text-like sequences. Instead, it builds **hop-indexed sparse edge sets** and **path-type codes** that act as sequence channels for attention.

#### Hybrid sampling / path coding engine (core)

```python
def sample(self):
    if self._indices is None:
        self._indices = torch.cat([
            torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
            torch.stack([self.train_pos_item+self.num_users, self.train_pos_user]),
            torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
            torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1)
        self._paths = torch.cat([
            torch.ones(self.train_pos_user.shape).repeat(2),
            torch.zeros(self.train_neg_user.shape).repeat(2)], dim=0).long().to(parse.device)
        sorted_indices = torch.argsort(self._indices[0, :])
        self._indices = self._indices[:, sorted_indices]
        self._paths = self._paths[sorted_indices]
        self._counts = torch.bincount(self._indices[0], minlength=self.num_nodes)
        self._counts_sum = torch.cumsum(self._counts, dim=0)
        d = torch.sqrt(self._counts)
        d[d == 0.] = 1.
        d = 1./d
        self._values = torch.ones(self._indices.shape[1]).to(
            parse.device)*d[self._indices[0]]*d[self._indices[1]]
    res_X, res_Y = [], []
    record_X = []
    X,  Y,  = self._indices,  torch.ones_like(self._paths).long()*2+self._paths
    loop_indices = torch.zeros_like(Y).bool()
    for hop in range(args.sample_hop):
        loop_indices = loop_indices | (X[0] == X[1])
        for i in range(hop % 2, hop, 2):
            loop_indices = loop_indices | (record_X[i][1] == X[1])
        record_X.append(X)
        res_X.append(X[:, ~loop_indices])
        res_Y.append(Y[~loop_indices]-2)
        next_indices = self._counts_sum[X[1]]-(torch.rand(X.shape[1]).to(parse.device)*self._counts[X[1]]).long()-1
        X = torch.stack([X[0], self._indices[1, next_indices]], dim=0)
        Y = Y*2+self._paths[next_indices]
    return res_X, res_Y
```

Evidence: `external/SIGformer/code/dataloader.py:237-271`

**Mechanical interpretation:**
- `res_X[hop]`: sparse edge index pairs `(src, sampled_dst)` for each hop.
- `res_Y[hop]`: integer path-type code encoding the signed edge sequence encountered so far.
- `Y` recurrence (`Y = Y*2 + path_bit`) is a binary trie encoding over sign bits.

### 1.4 Transformer-ready attention input construction

In each encoder layer, sampled graph structures feed sparse self-attention:

```python
indices, paths = self.dataset.sample()
all_emb = self.layers[i](all_emb,
                         indices,
                         self.dataset.L_eigs,
                         paths)
```

Evidence: `external/SIGformer/code/model.py:96-101`

Within attention:
- Content score: `q[src] · k[dst]`.
- Structural score: `eig[src] · eig[dst]`.
- Path score: learned scalar from `path_emb[path_code]`.
- Sparse row-normalization via custom softmax on sampled edges.

Evidence: `external/SIGformer/code/model.py:24-48`; `external/SIGformer/code/utils.py:41-48`

### 1.5 Tensor Shape Map (single forward pass)

Let:
- $N$ = number of graph nodes (`num_users + num_items`)
- $d$ = hidden dimension
- $H$ = `sample_hop`
- $E_h$ = sampled edge count at hop $h$ after loop filtering
- $E_{tot}=\sum_h E_h$
- $B$ = user eval batch size
- $I$ = number of items

#### Implemented SIGformer shapes

- Base embedding input (`all_emb`): $[N, d]$  
  Evidence: `external/SIGformer/code/model.py:92-95`
- Per-hop sampled edges (`indices[h]`): $[2, E_h]$  
  Evidence: `external/SIGformer/code/dataloader.py:266`
- Per-hop path types (`path_type[h]`): $[E_h]$  
  Evidence: `external/SIGformer/code/dataloader.py:267`
- Attention logits per hop (`x`): $[E_h]$  
  Evidence: `external/SIGformer/code/model.py:27-28`
- Sparse attention matrix (assembled): shape $[N, N]$ with $E_{tot}$ non-zeros  
  Evidence: `external/SIGformer/code/model.py:39, 48`
- Layer output: $[N, d]$  
  Evidence: `external/SIGformer/code/model.py:57-64`
- Final split embeddings: users $[U,d]$, items $[I,d]$  
  Evidence: `external/SIGformer/code/model.py:105`
- Eval rating tensor: $[B, I]$  
  Evidence: `external/SIGformer/code/model.py:119-122`

#### Requested canonical Transformer mapping (for EDGRec wrapper)

A sequence-compatible adapter can materialize:
- GNN input: $[N, d]$
- Transformer input (post-gather/projection): $[B, L, d]$, where $L$ = sampled neighbors per anchor (+self token if needed)
- Attention scores (dense canonical form): $[B, n_{heads}, L, L]$
- Output score: $[B, N_{candidates}]$

**Important:** original SIGformer code does **not** instantiate dense $[B,n_{heads},L,L]$; it uses sparse edge-list attention.

---

## 2) Module Contracts (Reusable Components)

### 2.1 Engine A — Sign-aware GNN kernel (graph structural channel)

This is implemented as sparse Laplacian construction plus spectral embedding extraction.

#### Full extracted kernel code (construction side)

```python
@ property
def A_pos(self):
    if self._A_pos is None:
        self._A_pos = torch.sparse_coo_tensor(
            torch.cat([
                torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                torch.stack([self.train_pos_item+self.num_users, self.train_pos_user])], dim=1),
            torch.ones(self.train_pos_user.shape[0]*2).to(parse.device),
            torch.Size([self.num_nodes, self.num_nodes]))
    return self._A_pos

@ property
def A_neg(self):
    if self._A_neg is None:
        self._A_neg = torch.sparse_coo_tensor(
            torch.cat([
                torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
                torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1),
            torch.ones(self.train_neg_user.shape[0]*2).to(parse.device),
            torch.Size([self.num_nodes, self.num_nodes]))
    return self._A_neg

@ property
def L(self):
    if self._L is None:
        self._L = (self.L_pos+args.alpha*self.L_neg)/(1+args.alpha)
    return self._L
```

Evidence: `external/SIGformer/code/dataloader.py:130-138, 173-181, 215-219`

#### Contract

- **Input:** Signed interactions split into positive/negative edge lists.
- **State:** Sparse `A_pos`, `A_neg`, `L_pos`, `L_neg`, mixed `L`, and optional `L_eigs` cache.
- **Output:**
  - Sparse structures for path sampling (`_indices`, `_paths`)
  - Optional node spectral features `L_eigs` shape $[N, eigs\_dim]$.

#### Mechanism type

- Not a PyG `MessagePassing` subclass.
- Implemented via sparse matrix algebra (`torch.sparse_coo_tensor`, `torch.sparse.mm`) and external eigensolver (`scipy.sparse.linalg.eigs`).

### 2.2 Engine B — Transformer-Graph bridge

The bridge is the `sample()` + `Attention.forward()` coupling:
1) `sample()` emits hop-indexed sparse edges + path codes from signed graph.
2) `Attention.forward()` converts graph edge tuples into attention logits and sparse aggregation weights.

#### Full extracted attention module

```python
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.path_emb = nn.Embedding(2**(args.sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
        self.sqrt_dim = 1./torch.sqrt(torch.tensor(args.hidden_dim))
        self.sqrt_eig = 1./torch.sqrt(torch.tensor(args.eigs_dim))
        self.my_parameters = [
            {'params': self.lambda0, 'weight_decay': 1e-2},
            {'params': self.path_emb.parameters()},
        ]

    def forward(self, q, k, v,  indices, eigs, path_type):
        ni, nx, ny, nz = [], [], [], []
        for i, pt in zip(indices, path_type):
            x = torch.mul(q[i[0]], k[i[1]]).sum(dim=-1)*self.sqrt_dim
            nx.append(x)
            if 'eig' in args.model:
                if args.eigs_dim == 0:
                    y = torch.zeros(i.shape[1]).to(parse.device)
                else:
                    y = torch.mul(eigs[i[0]], eigs[i[1]]).sum(dim=-1)
                ny.append(y)
            if 'path' in args.model:
                z = self.path_emb(pt).view(-1)
                nz.append(z)
            ni.append(i)
        i = torch.concat(ni, dim=-1)
        s = []
        s.append(torch.concat(nx, dim=-1))
        if 'eig' in args.model:
            s[0] = s[0]+torch.exp(self.lambda0)*torch.concat(ny, dim=-1)
        if 'path' in args.model:
            s.append(torch.concat(nz, dim=-1))
        s = [utils.sparse_softmax(i, _, q.shape[0]) for _ in s]
        s = torch.stack(s, dim=1).mean(dim=1)
        return torchsparsegradutils.sparse_mm(torch.sparse_coo_tensor(i, s, torch.Size([q.shape[0], k.shape[0]])), v)
```

Evidence: `external/SIGformer/code/model.py:11-48`

#### Bridge I/O contract

- **Inputs:**
  - `q,k,v`: $[N,d]$ node embeddings
  - `indices`: list of length $H$ with tensors $[2,E_h]$
  - `eigs`: $[N,eigs\_dim]$ or empty tensor when disabled
  - `path_type`: list of length $H$ with tensors $[E_h]$
- **Output:** updated node embeddings $[N,d]$.
- **Projection layers:** None. No learned $W_Q/W_K/W_V$ in current code.

### 2.3 Engine C — Self-attention implementation and masking audit

#### Masking/topology preservation

Topology is preserved by restricting attention to sampled graph edges only:
- No dense all-pairs matrix is formed.
- Row-wise normalization is done on sparse edge index rows (`indices[0]`).

```python
def sparse_softmax(indices, values, n):
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)
```

Evidence: `external/SIGformer/code/utils.py:47-48`

#### Audit result

- Not standard `nn.MultiheadAttention`.
- Custom sparse sign/path-aware attention.
- No explicit causal mask; graph topology itself acts as hard mask.

---

## 3) Performance Profiling & Hardware Adaptation (RTX 5080 / Linux)

### 3.1 Transformer bottleneck handling

#### What exists in code
- Dense $O(L^2)$ attention is avoided entirely.
- Attention complexity scales with sampled edge count $E_{tot}$, not all token pairs.
- Sequence depth pressure is controlled via `sample_hop` (default `4`).

Evidence: `external/SIGformer/code/dataloader.py:261-271`; `external/SIGformer/code/parse.py:30`; `external/SIGformer/code/model.py:24-48`

#### What does not exist
- No FlashAttention.
- No block-sparse fused attention kernel.
- No explicit sequence truncation beyond hop-limited random walk sampling.

### 3.2 VRAM scaling audit

#### Sparse vs dense storage
- Signed adjacency and Laplacian are sparse COO tensors (`torch.sparse_coo_tensor`).
- Sparse matmul used for graph ops and final sparse aggregation in attention.

Evidence: `external/SIGformer/code/dataloader.py:130-203`; `external/SIGformer/code/model.py:48`

#### Quadratic-growth tensors in practice
1) **Evaluation rating matrix** `rating = user_e @ item_emb.T` with shape $[B, I]$ can be large and dense.
   - Evidence: `external/SIGformer/code/model.py:119-122`.
2) **No in-train dense attention score matrix** $[N,N]$ is materialized as dense; only sparse COO is created.

### 3.3 Big-O complexity (formal)

Let $E^+$ and $E^-$ be counts of positive/negative directed edges after mirror expansion, $E=E^+ + E^-$. Let $H$ be `sample_hop`, $d$ hidden dim, and $k$ eigs dimension.

#### A) Graph pre-processing

1) Build signed sparse adjacency:  
$$
T = O(E), \quad M = O(E)
$$

2) Degree and normalization (`D^{-1/2} A D^{-1/2}` sparse operations): approximately  
$$
T = O(E), \quad M = O(E)
$$
(per sign graph, up to constant factors of sparse-sparse multiply)

3) Mixed Laplacian:  
$$
T = O(E), \quad M = O(E)
$$

4) Partial eigendecomposition (`scipy.sparse.linalg.eigs`, ARPACK-style iterative):  
$$
T \approx O(\text{iters} \cdot k \cdot E), \quad M = O(E + Nk)
$$

Evidence: `external/SIGformer/code/dataloader.py:147-235`

#### B) Hybrid forward pass (per layer)

1) Sampling update over hops (`sample()`):  
$$
T \approx O(H \cdot E), \quad M \approx O(H \cdot E)
$$

2) Sparse attention score compute:
- content term + optional eig term over sampled edges:  
$$
T = O(E_{tot} \cdot d + E_{tot} \cdot k)
$$
- path embedding lookup and sparse softmax:
$$
T = O(E_{tot}), \quad M = O(E_{tot})
$$
- sparse aggregation `A_sparse @ V`:
$$
T = O(E_{tot} \cdot d), \quad M = O(E_{tot} + Nd)
$$

Total per layer:
$$
T = O(H\cdot E + E_{tot}(d+k)), \quad M = O(H\cdot E + Nd)
$$

Evidence: `external/SIGformer/code/dataloader.py:237-271`; `external/SIGformer/code/model.py:24-48`

#### C) Ranking inference

For eval batch size $B$ and item count $I$:
$$
T = O(BId), \quad M = O(BI)
$$

Evidence: `external/SIGformer/code/model.py:118-125`

### 3.4 RTX 5080 adaptation recommendations (implementation-constrained)

- Keep sparse graph ops on GPU; avoid accidental dense conversion (`to_dense()`) except degree vector extraction already in code.
- Reduce `test_batch_size` if `[B, I]` rating tensor saturates VRAM.
- For large $E$, cap `sample_hop` or redesign sampling to fixed fanout to bound $E_{tot}$.
- If eigendecomposition dominates startup, precompute/cache `L_eigs` offline per dataset snapshot.

---

## 4) Architectural Anomalies & Hyperparameters

### 4.1 Symmetry / sign-bias audit

- Positive/negative signs are treated via separate graph channels (`A_pos`, `A_neg`) and separate Laplacians (`L_pos`, `L_neg`) before scalar mixing with `alpha`.
- Attention kernel itself shares a single parameterization across signs; no dedicated per-sign Q/K/V projections.
- Path sign histories are encoded via `path_emb` indices, giving implicit sign-specific biasing.

Evidence: `external/SIGformer/code/dataloader.py:130-219, 244-270`; `external/SIGformer/code/model.py:15, 35-37`

### 4.2 Magic Number List (hardcoded constants and defaults)

#### Model/training defaults (`parse.py`)
- `hidden_dim = 64`
- `n_layers = 3`
- `model = "eig+path"`
- `alpha = 0.0`
- `beta = 1.0`
- `eigs_dim = 64`
- `sample_hop = 4`
- `learning_rate = 1e-2`
- `lambda_reg = 1e-4`
- `test_batch_size = 1024`
- `epochs = 1000`
- `topks = [5,10,15,20]`

Evidence: `external/SIGformer/code/parse.py:11-30`

#### Additional hardcoded constants (non-CLI)
- OMP threads: `OMP_NUM_THREADS = "20"`  
  Evidence: `external/SIGformer/code/parse.py:6`
- Path embedding table length: `2**(sample_hop+1)-2`  
  Evidence: `external/SIGformer/code/model.py:15`
- `lambda0` custom weight decay: `1e-2`  
  Evidence: `external/SIGformer/code/model.py:20`
- Embedding init std: `0.1`  
  Evidence: `external/SIGformer/code/model.py:75-76`
- Seen-item masking score: `-(1 << 10)`  
  Evidence: `external/SIGformer/code/model.py:123-124`
- Sparse softmax clamp: `torch.clamp(exp(values), min=-5, max=5)`  
  Evidence: `external/SIGformer/code/utils.py:47-48`

#### Requested values not present in implementation
- Number of attention heads: **not implemented** (single-head sparse attention).
- Dropout rate: **not implemented** (no dropout module in model path).

---

## 5) EDGRec Integration Interface

### 5.1 Minimal Viable Code (MVC): three critical SIGformer functions

### MVC-1: signed multi-hop sampler (graph-to-sequence bridge)

```python
def sigformer_sample(indices_cache, paths_cache, counts, counts_sum, sample_hop, device):
    """
    Args:
        indices_cache: LongTensor [2, E] sorted by source index
        paths_cache: LongTensor [E] with sign bit per edge (1=pos, 0=neg)
        counts: LongTensor [N] out-degree per source in indices_cache
        counts_sum: LongTensor [N] prefix sum of counts
        sample_hop: int
    Returns:
        hop_indices: list[LongTensor [2, E_h]]
        hop_path_types: list[LongTensor [E_h]]
    """
    res_X, res_Y = [], []
    record_X = []
    X = indices_cache
    Y = torch.ones_like(paths_cache, device=device).long() * 2 + paths_cache
    loop_indices = torch.zeros_like(Y).bool()

    for hop in range(sample_hop):
        loop_indices = loop_indices | (X[0] == X[1])
        for i in range(hop % 2, hop, 2):
            loop_indices = loop_indices | (record_X[i][1] == X[1])

        record_X.append(X)
        res_X.append(X[:, ~loop_indices])
        res_Y.append(Y[~loop_indices] - 2)

        next_indices = counts_sum[X[1]] - (torch.rand(X.shape[1], device=device) * counts[X[1]]).long() - 1
        X = torch.stack([X[0], indices_cache[1, next_indices]], dim=0)
        Y = Y * 2 + paths_cache[next_indices]

    return res_X, res_Y
```

Source equivalent: `external/SIGformer/code/dataloader.py:237-271`

### MVC-2: sign-aware sparse attention kernel

```python
def sigformer_sparse_attention(q, k, v, hop_indices, eigs, hop_path_types, path_emb, lambda0, use_eig=True, use_path=True):
    """
    Args:
        q,k,v: FloatTensor [N, d]
        hop_indices: list[[2, E_h]]
        eigs: FloatTensor [N, k] or empty
        hop_path_types: list[[E_h]]
        path_emb: nn.Embedding[num_path_codes, 1]
        lambda0: scalar nn.Parameter
    Returns:
        FloatTensor [N, d]
    """
    ni, nx, ny, nz = [], [], [], []
    sqrt_dim = 1.0 / torch.sqrt(torch.tensor(q.shape[1], device=q.device, dtype=q.dtype))

    for i, pt in zip(hop_indices, hop_path_types):
        x = (q[i[0]] * k[i[1]]).sum(dim=-1) * sqrt_dim
        nx.append(x)

        if use_eig:
            y = torch.zeros(i.shape[1], device=q.device) if eigs.numel() == 0 else (eigs[i[0]] * eigs[i[1]]).sum(dim=-1)
            ny.append(y)

        if use_path:
            z = path_emb(pt).view(-1)
            nz.append(z)

        ni.append(i)

    edge_index = torch.concat(ni, dim=-1)
    channels = [torch.concat(nx, dim=-1)]

    if use_eig:
        channels[0] = channels[0] + torch.exp(lambda0) * torch.concat(ny, dim=-1)
    if use_path:
        channels.append(torch.concat(nz, dim=-1))

    def sparse_softmax(idx, values, n_nodes):
        scores = torch.exp(values).clamp(max=5)
        denom = torch.zeros(n_nodes, device=scores.device).scatter_add(0, idx[0], scores)
        denom[denom == 0] = 1
        return scores / denom[idx[0]]

    attn = torch.stack([sparse_softmax(edge_index, c, q.shape[0]) for c in channels], dim=1).mean(dim=1)
    A = torch.sparse_coo_tensor(edge_index, attn, (q.shape[0], q.shape[0]))
    return torch.sparse.mm(A, v)
```

Source equivalent: `external/SIGformer/code/model.py:24-48`; `external/SIGformer/code/utils.py:41-48`

### MVC-3: layer stack + final split (hybrid forward)

```python
def sigformer_forward(user_emb_table, item_emb_table, layer_fn_list, sampler_fn, eigs):
    """
    Args:
        user_emb_table: FloatTensor [U, d]
        item_emb_table: FloatTensor [I, d]
        layer_fn_list: list(callable(x, indices, eigs, path_types) -> [N,d])
        sampler_fn: callable() -> (list[[2,E_h]], list[[E_h]])
        eigs: FloatTensor [N,k] or empty
    Returns:
        users_out: FloatTensor [U,d]
        items_out: FloatTensor [I,d]
    """
    all_emb = torch.cat([user_emb_table, item_emb_table], dim=0)  # [N,d]
    embs = [all_emb]

    for layer in layer_fn_list:
        indices, paths = sampler_fn()
        all_emb = layer(all_emb, indices, eigs, paths)
        embs.append(all_emb)

    stacked = torch.stack(embs, dim=1)      # [N, L+1, d]
    pooled = torch.mean(stacked, dim=1)     # [N, d]
    U = user_emb_table.shape[0]
    users_out, items_out = torch.split(pooled, [U, item_emb_table.shape[0]], dim=0)
    return users_out, items_out
```

Source equivalent: `external/SIGformer/code/model.py:91-105`

### 5.2 Module API for unified PyTorch/DGL integration

## Interface: `SignAwareGraphTransformerLayer`

### Inputs
- `node_x`: FloatTensor `[N, d]`
- `signed_edges`: Dict with
  - `pos_edge_index`: LongTensor `[2, E_pos]`
  - `neg_edge_index`: LongTensor `[2, E_neg]`
- `cache` (optional): precomputed
  - normalized signed Laplacians
  - eigenvectors `eigs` `[N, k]`
  - sorted edge index / path bits / degree counts
- `sampler_cfg`: `{sample_hop: int, fanout: Optional[int]}`
- `mode_cfg`: `{use_eig: bool, use_path: bool}`

### State (learnable)
- `node_embedding` tables (if end-to-end recommender module)
- `path_emb`: Embedding `[2^(sample_hop+1)-2, 1]`
- `lambda0`: scalar for eig channel weighting
- optional projection matrices (recommended for EDGRec modernization):
  - `W_q, W_k, W_v: [d, d]`
  - optional multi-head decomposition

### Outputs
- `node_x_next`: FloatTensor `[N, d]`
- optional diagnostics:
  - sampled edges per hop
  - path-type histograms
  - sparse attention nnz and row entropy

### Required lifecycle hooks
1. `build_signed_graph(batch_or_full_interactions)`
2. `precompute_structural_cache()`
3. `sample_paths()`
4. `forward(node_x, cache)`
5. `score(users, items)`

### 5.3 Integration notes for canonical Transformer compatibility

To expose a standard Transformer contract in EDGRec while preserving SIGformer semantics:
1) Build per-anchor neighbor sequences from sampled edges -> tensor `[B,L,d]`.
2) Use path-type embedding as additive attention bias tensor.
3) Use eig-similarity as structural bias term in attention logits.
4) Keep hard topology mask from sampled edges.
5) Optionally backport to sparse kernels for efficiency when $L$ is large.

---

## 6) Reconstruction Checklist (without original repo)

1. Parse interaction triples and split by sign using threshold.
2. Build mirrored bipartite sparse adjacency for both signs.
3. Build normalized signed Laplacians and mixed Laplacian with `alpha`.
4. Optionally compute lowest-eigenvector features of mixed Laplacian.
5. Build sorted edge cache and path sign bits.
6. For each layer, sample multi-hop path edges and path codes.
7. Run sparse sign-aware attention over sampled edges.
8. Residual-by-averaging over layer outputs to final node embeddings.
9. Train with pairwise softplus objective with `beta`-weighted negative branch.
10. Evaluate by dense user-item scoring and top-k metrics.

All required mechanics above are fully specified by extracted code sections and contracts in this document.
