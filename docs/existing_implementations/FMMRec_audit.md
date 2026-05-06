# FMMRec Developer Blueprint (Standalone Code-First Audit)

## Mechanical Summary Table

| Axis | Implementation Reality | Where in Code |
|---|---|---|
| Primary runtime modes | (A) Base recommender pretrain (`LATTICE`/`DRAGON`), (B) modality disentanglement pretrain (`BMMF_runner`), (C) fairness tuning (`BFMMR`) | `src/main.py`, `src/BMMF_runner.py`, `src/utils/quick_start.py` |
| In-memory interaction format | `scipy.sparse.coo_matrix` -> `torch.sparse.FloatTensor` -> **dense** `torch.Tensor` with row-normalization in multiple modules | `src/utils/dataloader.py` (~L253-L291), `src/BMMF_filters.py` (~L29-L33), `src/models/fairness_models/bfmmr.py` (~L48-L52), `src/models/recommendation_models/lattice.py` (~L41-L45), `src/models/recommendation_models/dragon.py` (~L95-L99) |
| Negative sampling | Python-loop rejection sampling over user history; no hard-negative mining | `src/utils/dataloader.py` (~L307-L368) |
| Causal/disentanglement core | Dual channels (`biased_trs`, `filtered_trs`) + cosine losses + adversarial discriminators; BFMMR uses graph-delta injection `(filtered_h - biased_h)` | `src/BMMF_filters.py` (~L54-L82), `src/BMMF_trainer.py` (~L170-L210), `src/models/fairness_models/bfmmr.py` (~L314-L403) |
| GNN/message passing | `LATTICE`: `torch.sparse.mm` on UI graph + dense item graph propagation; `DRAGON`: PyG `MessagePassing` + `torch.sparse.mm` multimodal item graph | `src/models/recommendation_models/lattice.py` (~L188-L216), `src/models/recommendation_models/dragon.py` (~L286, ~L451-L476) |
| Output score | Dot-product ranking (`sum(u*i)` or `matmul`) | `src/models/fairness_models/bfmmr.py` (~L207-L220), `src/models/recommendation_models/dragon.py` (~L293-L296), `src/models/recommendation_models/lattice.py` (~L262) |
| Independence loss type | Adversarial BCE/NLL + cosine repulsion/reconstruction; no HSIC/DisCo in repository | `src/BMMF_filters.py` (~L66-L82), `src/BMMF.py` (~L118-L131, ~L294-L302) |
| Numerical stabilizers | `1e-7` in DRAGON Laplacian degree sum; `isinf -> 0` in utility Laplacian | `src/models/recommendation_models/dragon.py` (~L193), `src/utils/utils.py` (~L148-L153) |
| Platform assumptions | Linux + CUDA-first (multiple `.cuda()` calls hardcoded) | `src/BMMF_filters.py`, `src/BMMF.py`, `src/models/fairness_models/bfmmr.py` |

---

## 1) End-to-End Data Flow & Ingestion Mechanics

## 1.1 Raw Data Ingestion (disk -> memory)

### Interaction ingestion (`RecDataset`)

```python
class RecDataset(object):
    def __init__(self, config, df=None):
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path'] + self.dataset_name)
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.feature_columns = self.config['feature_columns']
        self.splitting_label = self.config['inter_splitting_label']
        ...
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        cols.extend(self.feature_columns)
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])

    def split(self):
        dfs = []
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)
            dfs.append(temp_df)
        ...
        return [self.copy(_) for _ in dfs]
```

Evidence: `src/utils/dataset.py` (~L16-L75).

### Sparse graph materialization (`TrainDataLoader.inter_matrix`)

```python
def inter_matrix(self, form='coo', value_field=None):
    return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                      self.dataset.iid_field, form, value_field)

def _create_sparse_matrix(...):
    src = df_feat[source_field].values
    tgt = df_feat[target_field].values
    data = np.ones(len(df_feat)) if value_field is None else df_feat[value_field].values
    mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))
    return mat if form == 'coo' else mat.tocsr()
```

Evidence: `src/utils/dataloader.py` (~L253-L291).

### Dense conversion path used by models

```python
self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).float().to_dense()
row_sums = self.interaction_matrix.sum(axis=-1)
self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
```

Used in BMMF filters and BFMMR; same pattern in LATTICE/DRAGON.

Evidence: `src/BMMF_filters.py` (~L29-L33), `src/models/fairness_models/bfmmr.py` (~L48-L52), `src/models/recommendation_models/lattice.py` (~L41-L45), `src/models/recommendation_models/dragon.py` (~L95-L99).

## 1.2 Sampling & Batching Engine

### Negative sampling implementation

```python
def _get_neg_sample(self):
    cur_data = self.dataset[self.pr: self.pr + self.step]
    self.pr += self.step
    user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
    item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
    batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0), torch.unsqueeze(item_tensor, 0)))
    u_ids = cur_data[self.config['USER_ID_FIELD']]
    neg_ids = self._sample_neg_ids(u_ids).to(self.device)
    ...
    for f_i in range(len(self.config['feature_columns'])):
        feat_tensor = torch.tensor(cur_data[self.config['feature_columns'][f_i]].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((batch_tensor, feat_tensor.unsqueeze(0)))
    return batch_tensor

def _sample_neg_ids(self, u_ids):
    neg_ids = []
    for u in u_ids:
        iid = self._random()
        while iid in self.history_items_per_u[u]:
            iid = self._random()
        neg_ids.append(iid)
    return torch.tensor(neg_ids).type(torch.LongTensor)
```

Evidence: `src/utils/dataloader.py` (~L307-L368).

### System audit conclusions
- Sampling is **CPU-bound Python loops** (`for u in u_ids`, membership checks, repeated random sampling).
- No vectorized or GPU-native negative sampler.
- No hard negative mining (no score-aware / neighborhood-aware hard-negative logic for recommendation loss).
- Optional neighborhood sampling exists but is separate from recommendation negatives (`_get_neighborhood_samples`).

Evidence: `src/utils/dataloader.py` (~L361-L399).

## 1.3 Shape Map (single forward lineage)

Repository does not implement explicit factor tensor split (`[N_Factors, B, d/N_Factors]`). Its shape lineage is channel-based:

### Canonical batch tensors
- `interaction[0]`: users -> `[B]`
- `interaction[1]`: positive items -> `[B]`
- `interaction[2]`: negative items -> `[B]`
- `interaction[3:]`: sensitive labels -> `[B]` each

Evidence: `src/utils/dataloader.py` (~L307-L339).

### BMMF channel stage
- `self.feat` (modality item features): `[I, d_m]`
- `self.interaction_matrix[users]`: `[B, I]`
- `user_modal_feats = [B, d_m]`
- `biased_embedding = [B, d_m]`
- `filtered_embedding = [B, d_m]`

Evidence: `src/BMMF_filters.py` (~L71-L82).

### BFMMR stage
- base user embedding `U0`: `[U, d]`
- base item embedding `I0`: `[I, d]`
- graph-propagated user states `biased_h`, `filtered_h`: `[U, d]`
- fair user/item reps after filter bank: `[U, d]`, `[I, d]`
- batch scores (`DRAGON` branch style): `[B]`
- full-sort score matrix: `[B, I]`

Evidence: `src/models/fairness_models/bfmmr.py` (~L40-L45, ~L314-L403, ~L207-L220).

### Requested factorized latent tensor
- **Not present** in this repository.
- Closest analog is **two channels** (`biased`, `filtered`) and feature-wise filter bank averaging.

---

## 2) Core Engine Deconstruction (Reusable Modules)

## 2.1 Disentanglement Engine (factor splitting + signal routing)

## Complete code block (primary engine)

```python
class BMMF_filters(nn.Module):
    def __init__(self, config, dataloader):
        ...
        self.biased_trs = nn.Linear(self.feat.shape[1], self.feat.shape[1])
        self.filtered_trs = nn.Linear(self.feat.shape[1], self.feat.shape[1])
        self.cos_loss = nn.CosineEmbeddingLoss()

    def forward(self, users):
        user_modal_feats = torch.mm(self.interaction_matrix[users], self.feat)
        biased_embedding = self.biased_trs(user_modal_feats)
        filtered_embedding = self.filtered_trs(user_modal_feats)

        target = torch.Tensor([1]).cuda()
        filtered_reconstruction_loss = self.cos_loss(filtered_embedding, user_modal_feats, target)

        target = torch.Tensor([-1]).cuda()
        reg_loss = self.cos_loss(biased_embedding, filtered_embedding, target)

        return filtered_reconstruction_loss + 0.1 * reg_loss, biased_embedding, filtered_embedding
```

Evidence: `src/BMMF_filters.py` (~L10-L82).

### Mechanism type
- Not static tensor slicing.
- Not iterative routing.
- **Learned linear projection per channel** (`biased_trs`, `filtered_trs`).

### Contract
- Input:
  - `users: LongTensor[B]`
  - internal state includes `interaction_matrix[U,I]` and modality feature matrix `feat[I,d_m]`
- Output:
  - scalar loss
  - `biased_embedding[B,d_m]`
  - `filtered_embedding[B,d_m]`

## 2.2 Independence Engine (causal loss)

Repository uses adversarial + cosine constraints (not HSIC/DisCo kernel machinery).

### Cosine channel-separation component

```python
target = torch.Tensor([-1]).cuda()
reg_loss = self.cos_loss(biased_embedding, filtered_embedding, target)
```

Evidence: `src/BMMF_filters.py` (~L79-L82).

### Adversarial component in trainer

```python
losses, biased_embedding, filtered_embedding = self.filters(user_ids)
...
biased_loss, filtered_loss = discriminator(
    biased_embedding,
    filtered_embedding,
    interaction[feat_idx + 1 + self.use_neg_sampling]
)
losses += (self.disc_reg_weight_biased * biased_loss - self.disc_reg_weight_filtered * filtered_loss)
...
for _ in range(self.d_steps):
    biased_loss, filtered_loss = discriminator(
        biased_embedding.detach(),
        filtered_embedding.detach(),
        interaction[feat_idx + 1 + self.use_neg_sampling]
    )
    disc_loss = biased_loss + filtered_loss
    disc_loss.backward()
```

Evidence: `src/BMMF_trainer.py` (~L170-L210).

### Discriminator objective variants

Binary:
```python
biased_loss = BCELoss(sigmoid(biased_head(z_b)), labels)
filtered_loss = BCELoss(sigmoid(filtered_head(z_f)), labels)
```

Multiclass:
```python
random_labels = torch.randint_like(labels, low=0, high=self.out_dim-1)
biased_loss = NLLLoss(log_softmax(biased_head(z_b)), labels)
filtered_loss = NLLLoss(log_softmax(filtered_head(z_f)), random_labels)
```

Evidence: `src/BMMF.py` (~L118-L131, ~L294-L302).

### Stability guards present
- `row_sum = 1e-7 + ...` in DRAGON Laplacian build.
- `d_inv_sqrt[isinf] = 0.` in utility Laplacian build.

Evidence: `src/models/recommendation_models/dragon.py` (~L193), `src/utils/utils.py` (~L151).

No HSIC centering matrices, no RBF kernel bandwidth code, no pairwise kernel independence estimator exists in repo.

## 2.3 Propagation Kernel (message passing)

### LATTICE kernel

```python
for i in range(self.n_layers):
    h = torch.mm(self.item_adj, h)   # dense multimodal item graph propagation

for i in range(self.n_ui_layers):
    side_embeddings = torch.sparse.mm(adj, ego_embeddings)  # sparse UI graph propagation
```

Evidence: `src/models/recommendation_models/lattice.py` (~L188, ~L195, ~L214).

### DRAGON kernel

```python
for i in range(self.n_layers):
    h = torch.sparse.mm(self.mm_adj, h)
```

And PyG message passing:

```python
return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
...
deg = degree(row, size[0], dtype=x_j.dtype)
deg_inv_sqrt = deg.pow(-0.5)
norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
return norm.view(-1, 1) * x_j
```

Evidence: `src/models/recommendation_models/dragon.py` (~L286, ~L451-L476).

### BFMMR user correction kernel

```python
biased_adj = wt_b*Ab_text + wi_b*Ab_img + wa_b*Ab_audio
filtered_adj = wt_f*Af_text + wi_f*Af_img + wa_f*Af_audio

biased_h = self.user_representation
for i in range(self.n_mg_uugraph_layers):
    biased_h = torch.mm(biased_adj, biased_h)

filtered_h = self.user_representation
for i in range(self.n_mg_uugraph_layers):
    filtered_h = torch.mm(filtered_adj, filtered_h)

u_embedding = u_embedding + self.mg_weight * (filtered_h - biased_h)
```

Evidence: `src/models/fairness_models/bfmmr.py` (~L320-L332).

---

## 3) Performance Profiling & Hardware Adaptation (Linux + RTX 5080)

## 3.1 Latency sinks / code smells

1. **Python-loop negative sampling** (CPU-bound):
```python
for u in u_ids:
    iid = self._random()
    while iid in self.history_items_per_u[u]:
        iid = self._random()
```
Evidence: `src/utils/dataloader.py` (~L361-L368).

2. **Repeated host-device label transfer in discriminator path**:
```python
labels = labels.cpu().type(torch.FloatTensor).cuda()
```
Evidence: `src/BMMF.py` (~L122).

3. **Dense interaction matrix materialization (`U x I`) in multiple modules**.
Evidence: `src/BMMF_filters.py` (~L29-L33), `src/models/fairness_models/bfmmr.py` (~L48-L52), `src/models/recommendation_models/lattice.py` (~L41-L45), `src/models/recommendation_models/dragon.py` (~L95-L99).

4. **Dense all-pairs similarity for graph build (`N x N`)**:
```python
sim = torch.mm(context_norm, context_norm.transpose(1, 0))
```
Evidence: `src/utils/utils.py` (~L157-L160), `src/models/recommendation_models/dragon.py` (~L180).

5. **Python loops for user graph top-k sample expansion** in DRAGON.
Evidence: `src/models/recommendation_models/dragon.py` (~L330-L361).

## 3.2 VRAM scaling audit

Quadratic tensors:
- `sim[N,N]` in graph construction (`N=I` for item graphs or `N=U` for user-user BFMMR graph channels): **O(N^2)** memory.
- Dense normalized Laplacian in utility function uses `diagflat` and dense matmul: **O(N^2)** memory.

Code:
```python
sim = torch.mm(context_norm, context_norm.transpose(1, 0))
d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
```

Evidence: `src/utils/utils.py` (~L148-L160).

Interaction matrix dense conversion:
- **O(U*I)** memory footprint.
- Repeated across components, increasing peak VRAM/host RAM pressure.

## 3.3 Big-O complexity

### Graph construction (preprocessing)

- Interaction COO build: `O(E_ui)` time, `O(E_ui)` memory sparse.
- Dense interaction conversion: `O(U*I)` memory.
- kNN graph via dense cosine:
  - similarity: `O(N^2*d)`
  - top-k row selection: `O(N^2 log k)` (framework-level top-k cost)
  - dense Laplacian path (utils): `O(N^2)` memory + dense matmul cost.

### Training iteration

Let `B=batch`, `d=embedding dim`, `F=#sensitive attributes`, `D=d_steps`.

- BMMF filter forward: `O(B*I*d_m + B*d_m^2)`.
- Discriminator pass per feature: approx `O(B*d*hidden)`; repeated `F` times.
- Inner adversarial update multiplier: `D`.
- BFMMR fair rep recomputation includes user graph propagation `O(L_mg*U^2*d)` due to dense `torch.mm(A, H)`.

### Inference/ranking

- Full-sort score matrix: `torch.matmul([B,d],[d,I])` -> `O(B*I*d)` time, `O(B*I)` output memory.
- Sampled pairwise scoring: `O(B*d)`.

Evidence: `src/models/fairness_models/bfmmr.py` (~L219), `src/models/recommendation_models/lattice.py` (~L262), `src/models/recommendation_models/dragon.py` (~L410-L416 equivalent in class).

---

## 4) Architectural Anomalies & Hyperparameters

## 4.1 Symmetry breaks / biased logic

1. **Multiclass filtered head trained on random labels** (intentional anti-predictive pressure):
```python
random_labels = torch.randint_like(labels, low=0, high=self.out_dim-1)
filtered_loss = self.criterion(filtered_output.squeeze(), random_labels)
```
Evidence: `src/BMMF.py` (~L298-L301).

2. **Audio adjacency in BFMMR built from text variable path** (`user_biased_t_feat` / `user_filtered_t_feat` used in audio branch):
```python
biased_audio_adj = build_sim(user_biased_t_feat)
filtered_audio_adj = build_sim(user_filtered_t_feat)
```
Evidence: `src/models/fairness_models/bfmmr.py` (~L126, ~L175).

3. **User-only filter mode bypasses item filter path entirely**.
Evidence: `src/models/fairness_models/bfmmr.py` (~L390-L399).

4. **Model-specific hardcoded logic in DRAGON** (`construction='cat'`, `k=40`, `drop_rate=0.1`) outside YAML.
Evidence: `src/models/recommendation_models/dragon.py` (~L38-L47).

## 4.2 Magic Number List

### Architectural constants (code hardcoded)
- `0.1` repulsion coefficient in BMMF filter loss.
  - `src/BMMF_filters.py` (~L82)
- `threshold=0.5` for binary discriminator predictions.
  - `src/BMMF.py` (~L137-L140, ~L154-L157)
- DRAGON:
  - `k=40`, `drop_rate=0.1`, `dim_latent=64`, `dim_feat=128`, `num_layer=1`.
  - `src/models/recommendation_models/dragon.py` (~L38-L53)
- Laplacian epsilon `1e-7` in DRAGON graph normalization.
  - `src/models/recommendation_models/dragon.py` (~L193)

### Config-driven hyperparameters (dataset-specific / run-specific)
From `overall.yaml` and dataset YAML:
- `epochs=1000`, `train_batch_size=2048`, `learning_rate=0.001`, `eval_batch_size=4096`, `d_steps=1`, `disc_reg_weight=0.1`.
- `attacker_layers=2`, `attacker_dropout=0.2`, `neg_slope=0.2`.
- BFMMR graph fusion weights:
  - `ml1m`: text/image/audio = `0.2/0.6/0.2`
  - `microlens`: `0.6/0.2/0.2`
- `mg_weight=0.1`, `n_mg_uugraph_layers=1`.

Evidence: `src/configs/overall.yaml` (~L20-L80), `src/configs/dataset/ml1m.yaml` (~L24-L40), `src/configs/dataset/microlens.yaml` (~L26-L50), `src/main.py` (~L31-L36).

---

## 5) U-CaGNN Integration Interface

## 5.1 Minimal Viable Code (extract exactly three functions)

These three functions reproduce the main causal effect chain in this repository.

## MVC Function 1 — Channel Generator (`BMMF_filters.forward`)

```python
def forward(self, users):
    user_modal_feats = torch.mm(self.interaction_matrix[users], self.feat)
    biased_embedding = self.biased_trs(user_modal_feats)
    filtered_embedding = self.filtered_trs(user_modal_feats)

    target = torch.Tensor([1]).cuda()
    filtered_reconstruction_loss = self.cos_loss(filtered_embedding, user_modal_feats, target)

    target = torch.Tensor([-1]).cuda()
    reg_loss = self.cos_loss(biased_embedding, filtered_embedding, target)

    return filtered_reconstruction_loss + 0.1 * reg_loss, biased_embedding, filtered_embedding
```

Source: `src/BMMF_filters.py` (~L71-L82).

## MVC Function 2 — Adversarial Coupler (`BMMFTrainer._train_epoch` core)

```python
losses, biased_embedding, filtered_embedding = self.filters(user_ids)

for feat_idx in self.fair_disc_dict:
    discriminator = self.fair_disc_dict[feat_idx]
    biased_loss, filtered_loss = discriminator(
        biased_embedding,
        filtered_embedding,
        interaction[feat_idx + 1 + self.use_neg_sampling]
    )
    losses += (self.disc_reg_weight_biased * biased_loss - self.disc_reg_weight_filtered * filtered_loss)

loss.backward()
self.filters.optimizer.step()

for feat_idx in self.fair_disc_dict:
    for _ in range(self.d_steps):
        discriminator = self.fair_disc_dict[feat_idx]
        discriminator.optimizer.zero_grad()
        biased_loss, filtered_loss = discriminator(
            biased_embedding.detach(),
            filtered_embedding.detach(),
            interaction[feat_idx + 1 + self.use_neg_sampling]
        )
        disc_loss = biased_loss + filtered_loss
        disc_loss.backward(retain_graph=False)
        discriminator.optimizer.step()
```

Source: `src/BMMF_trainer.py` (~L170-L210).

## MVC Function 3 — Fair Representation Generator (`BFMMR.get_fair_representation`)

```python
def get_fair_representation(self):
    fair_user_representation = None
    fair_item_representation = None
    u_embedding = self.user_representation
    i_embedding = self.item_representation

    biased_adj = self.mg_uugraph_text_weight * self.biased_text_original_adj + \
                 self.mg_uugraph_image_weight * self.biased_image_original_adj + \
                 self.mg_uugraph_audio_weight * self.biased_audio_original_adj
    biased_h = self.user_representation
    for i in range(self.n_mg_uugraph_layers):
        biased_h = torch.mm(biased_adj, biased_h)

    filtered_adj = self.mg_uugraph_text_weight * self.filtered_text_original_adj + \
                   self.mg_uugraph_image_weight * self.filtered_image_original_adj + \
                   self.mg_uugraph_audio_weight * self.filtered_audio_original_adj
    filtered_h = self.user_representation
    for i in range(self.n_mg_uugraph_layers):
        filtered_h = torch.mm(filtered_adj, filtered_h)

    u_embedding = u_embedding + self.mg_weight * (filtered_h - biased_h)

    if self.filter_mode == 'independent':
        for _, filter in self.user_filters.items():
            if fair_user_representation is None:
                fair_user_representation = filter(u_embedding)
            else:
                fair_user_representation += filter(u_embedding)
        fair_user_representation /= float(self.num_features)

        for _, filter in self.item_filters.items():
            if fair_item_representation is None:
                fair_item_representation = filter(i_embedding)
            else:
                fair_item_representation += filter(i_embedding)
        fair_item_representation /= float(self.num_features)

    elif self.filter_mode == 'shared':
        ...  # prompt routing: none/add/concat
        ...  # same averaging over shared filter_dict

    elif self.filter_mode == 'user-only':
        ...  # user filters only, item stays i_embedding

    self.fair_user_representation = fair_user_representation
    self.fair_item_representation = fair_item_representation
    return fair_user_representation, fair_item_representation
```

Source: `src/models/fairness_models/bfmmr.py` (~L314-L403).

## 5.2 Module API to wrap into U-CaGNN

### Recommended unified interface

```python
class UCaGNNCausalAdapter(nn.Module):
    """
    State:
      - interaction operator R (sparse preferred)
      - modality encoders/features
      - channel projectors (biased/filtered)
      - graph builder + propagator
      - filter router (independent/shared/user-only)
      - adversarial heads for sensitive attributes
    """

    def build_channels(self, user_ids, modality_feats):
        """
        Input:
          user_ids: LongTensor[B]
          modality_feats: Dict[str, Tensor[I, d_m]]
        Output:
          channels: Dict[str, Dict[str, Tensor[B, d_m]]]
            e.g. channels['v']['biased'], channels['v']['filtered']
          aux_losses: Dict[str, Tensor[]]
        """

    def build_user_graphs(self, channels_item_level_or_user_level):
        """
        Input:
          channel tensors needed for graph build
        Output:
          A_b, A_f (sparse preferred; fallback dense)
        """

    def get_fair_embeddings(self, U_base, I_base, A_b, A_f, mode='independent', prompt_mode='add'):
        """
        Input:
          U_base: Tensor[U, d]
          I_base: Tensor[I, d]
          A_b, A_f: Tensor/SpTensor[U, U]
        Output:
          U_fair: Tensor[U, d]
          I_fair: Tensor[I, d]
        """

    def training_step(self, batch):
        """
        Returns combined recommendation + causal + adversarial loss.
        """
```

### Required adapter I/O contracts

- **Input IDs**: `users[B]`, `pos_items[B]`, `neg_items[B]`, sensitive labels `[B]xF`
- **Base embeddings**: `U_base[U,d]`, `I_base[I,d]`
- **Channel embeddings**: `Z_b[B,d_m]`, `Z_f[B,d_m]` per modality
- **Fair embeddings**: `U_fair[U,d]`, `I_fair[I,d]`
- **Training score output**: `pos_scores[B]`, `neg_scores[B]`
- **Full-sort output**: `[B, I]`

---

## Appendix A — Direct reconstruction notes (implementation-critical)

1. To reproduce behavior exactly, keep two-stage precomputation:
   - stage-1 save modality channels,
   - stage-2 load them in BFMMR.
2. To modernize for U-CaGNN, collapse stages into end-to-end mode with optional checkpointing.
3. Replace dense interaction and dense all-pairs similarity first; these are primary scalability blockers.
4. If preserving exact semantics, keep multiclass random-label filtered discriminator behavior.

(End of standalone blueprint.)
