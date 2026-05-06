# Cross-Repository Technical Synthesis (Code-First, Audit-Derived)

**Scope:** external implementation mechanics only. No theory.

**Sources:** `CaDSI_audit.md`, `CausE_audit.md`, `DICE_audit.md`, `DirectAU_audit.md`, `FMMRec_audit.md`, `LightGCNpp_audit.md`, `LayerGCN_audit.md`, `MCLN_audit.md`, `MGCE_audit.md`, `PropCare_audit.md`, `SIGformer_audit.md`.

---

## 1. Runtime / Execution Matrix

| Repo | Framework | Runtime family | Primary interaction memory | Graph / structure object | Core propagate op | Score op | Eval surface | Source |
|---|---|---|---|---|---|---|---|---|
| **CaDSI** | TF1 | session graph + global `args` / `data_generator` | SciPy sparse + Python dict/list + NumPy side arrays | static bipartite sparse graph | factor-specific sparse routing over graph | `U_batch @ I_batch^T` | full-catalog + Python heap + multiprocessing | `CaDSI_audit.md`, Mechanical Summary Table |
| **CausE** | TF1 | static graph; CSV triple pipeline | TF dataset tensors for train; Python lists + NumPy for eval | none in main train path | none; MF-style embedding lookup | dot + bias | offline list/array eval utilities | `CausE_audit.md`, Mechanical Summary Table; §1.1 |
| **DICE** | PyTorch + DGL | `app.py::main` -> trainer | SciPy sparse + `.npz`/`.npy` artifacts + Torch tensors | DGL LightGCN path or factorization path | `LGConv.forward` via DGL `update_all(copy_u,sum)` | channel scores + ranking backend | FAISS `IndexFlatIP` + metrics | `DICE_audit.md`, Mechanical Summary Table |
| **DirectAU** | PyTorch / RecBole | positive-pair geometry training | interaction matrix + embedding lookup | optional LightGCN encoder or MF encoder | `F.normalize` on encoder output; optional LightGCN norm adj | no separate BPR score head in train path | RecBole full-sort eval | `DirectAU_audit.md`, Scope; §1; §2 |
| **FMMRec** | PyTorch | CUDA-first multimodal/fairness stack | SciPy COO -> torch sparse -> repeated dense tensors | UI graph + dense item/modality graphs | `torch.sparse.mm` + dense modality propagation | dot product | full-sort scoring | `FMMRec_audit.md`, Mechanical Summary Table |
| **LightGCN++** | PyTorch | main path + SELFRec path | sparse bipartite graph | preweighted adjacency | `torch.sparse.mm` after per-layer L2 normalize | dot product | full-sort scoring | `LightGCNpp_audit.md`, Execution Map; §2 |
| **LayerGCN** | PyTorch | `main.py` -> `quick_start` -> `LayerGCN` | train-split SciPy COO -> torch sparse | symmetric bipartite graph + masked pruned graph | `torch.sparse.mm` + cosine gate | dot product | dense `[B,I]` full-sort matrix | `LayerGCN_audit.md`, Execution Map; §2; §4.6 |
| **MCLN** | TF1 | dataset-specific session graph | SciPy sparse + Python maps + multimodal arrays | normalized bipartite sparse graph | `tf.sparse_tensor_dense_matmul` | factual / counterfactual score path | 99-negative sampled ranking | `MCLN_audit.md`, Mechanical Summary Table |
| **MGCE** | TF1 | dataset-specific session graph | SciPy sparse + modality/popularity arrays | normalized bipartite sparse graph | folded `tf.sparse_tensor_dense_matmul` | base + branch dot products | sampled top-k heap ranking | `MGCE_audit.md`, Mechanical Summary Table |
| **PropCare** | TF2 + Pandas/NumPy | two-stage propensity + downstream evaluator | scored candidate dataframe | none in evaluator path | none | `pred` column already present | causal dataframe evaluator (`CPrec`, `CDCG`, IPS variants) | `PropCare_audit.md`, Mechanical Summary Table |
| **SIGformer** | PyTorch 2.x | signed graph + sparse attention stack | signed edge tensors + path codes + eig features | signed pos/neg graphs + sampled edge-path sets | sparse attention over sampled edges | dense full-score matrix | full-sort eval per user batch | `SIGformer_audit.md`, Mechanical Summary Table |

---

## 2. Disk -> Memory -> Batch Contracts

### 2.1 Ingestion / storage contract by repository

| Repo | Disk input | First stable in-memory representation | Persistent side data | Source |
|---|---|---|---|---|
| **CaDSI** | `train.txt`, `test.txt`, 4 side files | `R` as SciPy sparse; `train_items`, `test_set`; NumPy side arrays | author/publisher/user/year embeddings | `CaDSI_audit.md` §1.1 |
| **CausE** | CSV triples | TF dataset batches `(user, product, label)` | none in graph sense | `CausE_audit.md` §1.1 |
| **DICE** | prebuilt `.npz`, `.npy` | SciPy COO + NumPy vectors | popularity arrays, skew datasets | `DICE_audit.md` §1.1 |
| **DirectAU** | RecBole dataset path | interaction matrix + user/item IDs | none beyond encoder config | `DirectAU_audit.md` §1; §3 |
| **FMMRec** | CSV interaction graph | pandas -> SciPy COO -> torch sparse -> dense | feature columns / sensitive labels / modality features | `FMMRec_audit.md` §1.1 |
| **LightGCN++** | dataset loader path | sparse graph + embedding tables | none central to kernel delta | `LightGCNpp_audit.md`, Execution Map; §2 |
| **LayerGCN** | CSV / dataset loader path | SciPy COO interaction matrix | none beyond split/meta config | `LayerGCN_audit.md` §4.1-§4.3 |
| **MCLN** | `train.txt`, `test.txt`, multimodal feature files | `R` as `dok -> csr -> coo`; Python maps | image/text features | `MCLN_audit.md` §1.1 |
| **MGCE** | `train.txt`, `test.txt`, img/text/pop files | `R` as SciPy sparse; train/test maps | visual/text/popularity arrays | `MGCE_audit.md` §1.2 |
| **PropCare** | semi-simulated CSVs / prepared dataframes | dataframe rows with effect/treatment/propensity | evaluator-only columns | `PropCare_audit.md` §1 |
| **SIGformer** | signed implicit train/test tables | positive edge tensors, negative edge tensors | path codes, eigen features | `SIGformer_audit.md` §1.1-§1.2 |

### 2.2 Graph build contract

| Repo family | Graph form | Normalization / weighting | Special channels |
|---|---|---|---|
| **CaDSI / MCLN / MGCE / LayerGCN** | bipartite block graph `[0,R;R^T,0]` | symmetric or cached sparse norm variants | none by sign |
| **LightGCN++** | bipartite block graph | asymmetric `alpha/beta` left-right degree weighting | none by sign |
| **DICE LGN path** | DGL graph from sparse artifacts | degree norm inside `LGConv` | causal channels above graph |
| **SIGformer** | separate positive and negative bipartite graphs | normalized signed Laplacians; mixed with scalar `alpha` | eigenvector channel + path-sign codes |
| **CausE / PropCare** | no graph in main path | N/A | treatment/propensity only |

### 2.3 Batch payload contract

| Family | Repo | Payload | Representative audited shape |
|---|---|---|---|
| **Triplet BPR** | LayerGCN, MGCE, MCLN | `(u, i+, i-)` | LayerGCN gathers `[B,D]` user/pos/neg tables -> scores `[B]` (`LayerGCN_audit.md` §4.5) |
| **Expanded triplet** | DICE | users/positives/negatives + mask / popularity routing | `[B,r]` IDs; channel embeddings `[B,r,d]` (`DICE_audit.md` §1.4) |
| **Positive pair** | DirectAU | `(u, i)` only | normalized embeddings consumed by alignment/uniformity (`DirectAU_audit.md` §2) |
| **CSV label batch** | CausE | `(u, product, label)` | label expanded to `[B,1]` (`CausE_audit.md` §1.1) |
| **Candidate dataframe** | PropCare | rows with `user,item,pred,treated,propensity,effect` | evaluator operates on sorted dataframe, not tensor batches (`PropCare_audit.md` §1-§2) |
| **Sparse edge/path batch** | SIGformer | hop-wise sampled edges + path types | `indices[h]: [2,E_h]`, `paths[h]: [E_h]` (`SIGformer_audit.md` §1.5) |

---

## 3. Encoder / Kernel Matrix

| Repo | Kernel family | Exact operator / implementation style | Pooling / combination | Causal/disentangled module |
|---|---|---|---|---|
| **CaDSI** | routed sparse factor model | factor-specific sparse tensors + iterative routing | factor block aggregation inside routed encoder | dCor across factor blocks |
| **CausE** | no graph; MF backbone | embedding lookup + dot + bias | none beyond MF path | control/treatment embedding alignment (`l1` / `l2` / cosine) |
| **DICE** | factorization + optional LightGCN | DGL `update_all(copy_u,sum)` in LGN path | LightGCN-like stacking in LGN variant | causal channels (`int`, `pop`) + popularity-aware sampler |
| **DirectAU** | MF or LightGCN + geometry | `F.normalize` in `forward`; optional LightGCN encoder | no ranking pooling emphasis; geometry loss dominates | alignment + uniformity only |
| **FMMRec** | multimodal/fairness graph stack | `torch.sparse.mm` UI path + dense item graph propagation | channel/filter-bank combination | `biased` vs `filtered` channels + adversarial discriminators |
| **LightGCN++** | LightGCN delta kernel | preweighted adjacency + per-layer L2 normalization + `torch.sparse.mm` | average propagated layers + fixed `gamma` mix with layer-0 | none |
| **LayerGCN** | LightGCN delta kernel | `torch.sparse.mm` + `F.cosine_similarity` to ego embedding | sum of refined propagated layers only; no layer-0 mean | none |
| **MCLN** | LightGCN + counterfactual attention | `tf.sparse_tensor_dense_matmul` over normalized adj | standard graph propagation feeding later attention | factual vs uninteracted attention subtraction |
| **MGCE** | dual-branch graph encoder | folded `tf.sparse_tensor_dense_matmul` on sparse adj | branch-specific propagation depth by modality structure | interest / conformity branches + popularity injection |
| **PropCare** | evaluator-centered | no graph kernel | N/A | propensity model + IPS/capped evaluator |
| **SIGformer** | sparse sign-aware attention | sparse `q·k` attention on sampled edges + eig bias + path embeddings | mean over layer outputs (LightGCN-style) | sign split + path-sign encoding |

### 3.1 LightGCN-family deltas packed

| Repo | Delta |
|---|---|
| **LightGCN++** | `alpha/beta` adjacency weighting; pre-layer L2 normalize; `gamma` residual mix |
| **LayerGCN** | per-layer cosine gate vs ego embedding; graph dropout/pruning; sum-refined propagated layers |
| **MCLN** | LightGCN core retained; causal novelty moved to counterfactual attention block |
| **MGCE** | LightGCN-like sparse propagation retained; novelty moved to branch duplication + popularity-aware channeling |

### 3.2 Disentanglement families packed

| Family | Repos | Mechanism |
|---|---|---|
| **Static branch duplication** | DICE, MGCE, FMMRec, MCLN | separate tables / channels / branch scores |
| **Routing-heavy factorization** | CaDSI | factor blocks + routing + dCor |
| **Counterfactual alignment** | CausE | control/treatment table alignment |
| **Geometry-only regularization** | DirectAU | alignment/uniformity on normalized embeddings |

---

## 4. Score / Loss Matrix

| Repo | Score construction | Primary train objective | Auxiliary objective | Reduction / quirk | Source |
|---|---|---|---|---|---|
| **CaDSI** | `U_batch @ I_batch^T` | BPR | `corDecay * dCor` + L2 | TF1 global-state training | `CaDSI_audit.md`, Mechanical Summary Table |
| **CausE** | MF dot + bias | pointwise binary classification | control/treatment alignment | no graph ranking core | `CausE_audit.md`, Mechanical Summary Table |
| **DICE** | channel scores (`int`, `pop`) + total score | ranking family | discrepancy + popularity-aware negatives | hard-negative path is popularity-conditioned | `DICE_audit.md`, Mechanical Summary Table; §1.2 |
| **DirectAU** | normalized user/item embeddings; no BPR negative path in `calculate_loss` | `alignment + gamma * (uniformity_u + uniformity_i)/2` | none outside geometry | `torch.pdist` O(B²); no guard for batch size 1 | `DirectAU_audit.md` §2 |
| **FMMRec** | dot-product ranking over filtered/biased representations | ranking family | adversarial BCE/NLL + cosine repulsion/reconstruction | fairness-specific stack | `FMMRec_audit.md`, Mechanical Summary Table |
| **LightGCN++** | dot product | pairwise ranking family | kernel delta is main distinction | fixed scalars `alpha,beta,gamma` | `LightGCNpp_audit.md`, Mechanical Summary; §2 |
| **LayerGCN** | dot product | BPR | raw-table L2 | `torch.sum` in BPR path => batch-size dependence | `LayerGCN_audit.md` §3.4 |
| **MCLN** | factual score minus counterfactual/uninteracted contribution | multi-branch BPR + L2 | none explicit as independence term | counterfactual block is the novelty | `MCLN_audit.md`, Mechanical Summary Table; §2.1 |
| **MGCE** | sum of base + branch-specific dot products | BPR triplets | negative cosine branch separation | sampled eval / popularity branch coupling | `MGCE_audit.md`, Mechanical Summary Table |
| **PropCare** | evaluator consumes existing `pred` column | downstream ranker external to evaluator | IPS-style causal estimate from treated/propensity columns | evaluator contract, not model loss | `PropCare_audit.md` §2; §3 |
| **SIGformer** | dense user-item score matrix after sparse attention encoder | pairwise softplus ranking | sign/path/eig structure inside encoder | signed data contract required | `SIGformer_audit.md`, Mechanical Summary Table |

### 4.1 Discrepancy / auxiliary kernel matrix

| Mechanism | Repos | Exact audited primitive | Cost profile |
|---|---|---|---|
| **Distance correlation** | CaDSI, DICE variants | pairwise dependence across factor blocks | expensive / quadratic-ish |
| **Negative cosine similarity / repulsion** | MGCE, FMMRec | cosine-based branch separation | cheap |
| **Alignment + uniformity** | DirectAU | row-wise Euclidean alignment + `torch.pdist` uniformity | O(B²) |
| **Counterfactual alignment** | CausE, MCLN | control/treatment distance or factual-counterfactual subtraction | medium, task-specific |
| **IPS estimate** | PropCare | row-wise estimate from `(outcome, treatment, propensity)` | evaluator-side only |

### 4.2 Popularity vs propensity split

| Subsystem | Repos | Technical role |
|---|---|---|
| **Popularity** | DICE, MGCE, partly FMMRec-style multimodal paths | negative sampling pressure, conformity-side signal, branch/modality input |
| **Propensity** | PropCare | exposure weighting, capping/clipping, IPS evaluation contract |

---

## 5. Evaluation Contract Matrix

| Repo family | Contract | Technical surface | Main caveat |
|---|---|---|---|
| **LayerGCN / LightGCN++ / SIGformer / DirectAU full-sort** | dense full-sort | `[B,D] @ [I,D]^T -> [B,I]` | high item-count memory + compute |
| **MGCE / MCLN / older TF1 paths** | sampled ranking | 1 positive + sampled negatives + heap top-k | weak as sole benchmark |
| **DICE** | retrieval + metric pass | FAISS candidate generation before ranking metrics | candidate generator becomes part of eval semantics |
| **PropCare** | causal dataframe eval | sort dataframe, cap propensities, aggregate causal metrics | requires treatment/propensity/effect columns |

---

## 6. Performance / Code-Smell Matrix

| Issue | Repos | Exact mechanism | Porting value |
|---|---|---|---|
| **CPU rejection sampler** | DICE, FMMRec, MGCE, MCLN, LayerGCN | Python loop + membership checks | do **not** copy literally |
| **Heap / Python-heavy eval** | CaDSI, MGCE, MCLN | Python heap or per-user loops | do **not** copy literally |
| **Dense sparse->dense conversion** | FMMRec | `torch.sparse.FloatTensor(...).to_dense()` paths | avoid as default |
| **TF1 global state coupling** | CaDSI, CausE, MGCE, MCLN | placeholders / sessions / global config objects | avoid as architecture reference |
| **Train/eval mismatch** | DirectAU, LayerGCN | normalized forward but not eval; masked graph train vs full graph eval | fix in any modern port |
| **Batch-size-sensitive loss** | LayerGCN | `torch.sum` BPR term | normalize / mean-reduce in any port |
| **Quadratic auxiliary with no guards** | DirectAU, CaDSI | `torch.pdist`, dCor | keep only with safeguards |
| **Hardcoded platform assumptions** | FMMRec, CaDSI | `.cuda()`-first style, dataset-specific filters | strip during port |
| **Front-loaded spectral cost** | SIGformer | eigendecomposition for structural bias | conditional-only feature |

---

## 7. Minimal Technical Carryover Set

### 7.1 Default set

| Component | Pull from | Exact mechanic to keep | Exact mechanic to avoid |
|---|---|---|---|
| **Sparse bipartite graph core** | LightGCN++, LayerGCN, MGCE, MCLN | block graph + sparse normalized propagation | dense graph materialization |
| **Explicit causal branches** | DICE, MGCE | duplicate embedding tables / branch tensors | immediate routing-heavy factorization |
| **Dot-product score head** | LayerGCN, MGCE, MCLN, CaDSI | `sum(u*i)` / `u@I^T` | overbuilt scorer before embeddings are right |
| **BPR + L2 core** | LayerGCN, MGCE, MCLN, CaDSI | ranking-first objective | pointwise-only objective as default |
| **Explicit popularity path** | DICE, MGCE | popularity-aware sampler or branch input | implicit popularity leakage only |

### 7.2 Optional set

| Component | Pull from | Exact mechanic |
|---|---|---|
| **`alpha/beta/gamma` kernel** | LightGCN++ | asymmetric degree weighting + gamma residual pooling |
| **Cosine refinement gate** | LayerGCN | `cos(all_embeddings, ego_embeddings)` row gate after sparse mm |
| **Geometry auxiliary** | DirectAU | alignment + uniformity on normalized embeddings |
| **Counterfactual alignment** | CausE | control/treatment distance hook |
| **Counterfactual attention** | MCLN | factual-minus-counterfactual scorer block |

### 7.3 Conditional-only set

| Component | Pull from | Only if... |
|---|---|---|
| **Propensity capping / IPS evaluator** | PropCare | treatment + propensity + effect columns exist |
| **Signed sparse attention** | SIGformer | edge polarity is meaningful and available |
| **Routing-heavy factor model** | CaDSI | factor routing is a research target, not just a baseline |
| **Multimodal fairness stack** | FMMRec | fairness + multimodal filtering are primary requirements |

### 7.4 Literal no-copy list

- TF1 `feed_dict` / session plumbing
- CPU rejection samplers
- heap-based Python ranking loops
- dense `U x I` tensors as a default path
- sampled-only evaluation as the sole benchmark
- hardcoded `.cuda()` assumptions

---

## 8. Fast Reopen Index

| Need | Reopen |
|---|---|
| **Branch disentanglement** | `DICE_audit.md`, `MGCE_audit.md` |
| **LightGCN-family kernel deltas** | `LightGCNpp_audit.md`, `LayerGCN_audit.md` |
| **Counterfactual mechanics** | `MCLN_audit.md`, `CausE_audit.md` |
| **Geometry regularization** | `DirectAU_audit.md` |
| **Propensity/effect evaluator** | `PropCare_audit.md` |
| **Signed graph attention** | `SIGformer_audit.md` |
| **Routing-heavy factorization** | `CaDSI_audit.md` |
| **Multimodal fairness filtering** | `FMMRec_audit.md` |

---

## 9. Technical Bottom Line

| Axis | Dense conclusion |
|---|---|
| **Backbone** | sparse normalized propagation dominates |
| **Disentanglement** | duplicate branches/tables dominate; routing is niche |
| **Score head** | dot product dominates |
| **Training core** | BPR-like ranking + L2 dominates |
| **Auxiliaries** | popularity, counterfactual, geometry, propensity are add-ons, not base contracts |
| **Runtime lesson** | copy model mechanics; do not copy old sampling/eval plumbing |
