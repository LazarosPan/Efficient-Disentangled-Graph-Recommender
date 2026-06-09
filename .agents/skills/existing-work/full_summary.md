# Causal Embeddings for Recommendation: Theory Synthesis

## 1. Foundational Causal Formalisms
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

Recommendation treats item exposure as treatment and click, purchase, rating, or downstream engagement as outcome. The literature centers on Potential Outcome (PO) and Structural Causal Models (SCM).

### 1.1 Potential Outcome and Structural Causal Model Views
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Framework | Core definition | Standard quantities |
| --- | --- | --- |
| Potential Outcome | Causal effect is defined by contrasting outcomes under exposure and non-exposure for the same user-item pair. | $ITE_{ui} = Y_{ui}(1) - Y_{ui}(0)$, $ATE = \mathbb{E}[Y(1) - Y(0)]$, ATT, CATE |
| Structural Causal Model | Causal mechanisms are represented by directed acyclic graphs and structural equations $X_i = f_i(PA_i, U_i)$. | $do(\cdot)$ interventions, back-door adjustment, front-door adjustment, do-calculus |

The surveys repeatedly use back-door and front-door adjustment:

$$P(y \mid do(x)) = \sum_z P(y \mid x, z) P(z)$$

$$P(y \mid do(x)) = \sum_m P(m \mid x) \sum_{x'} P(y \mid x', m) P(x')$$

The three do-calculus rules repeatedly cited are insertion or deletion of observations, action or observation exchange, and insertion or deletion of actions.

### 1.2 Core Causal Quantities and Adjustment Formulas
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Concept | Formula or definition |
| --- | --- |
| Propensity score | $e(w) = P(X=1 \mid W=w)$ |
| IPS estimator | $\hat{R}_{IPS} = \frac{1}{\|O\|} \sum_{(u,i) \in O} \frac{y_{ui}}{e_{ui}}$ |
| Clipped IPS | $\hat{R}_{CIPS} = \frac{1}{\|O\|} \sum_{k=1}^{\|O\|} \min \left\{ \frac{e_{\pi}(X)}{e_{\pi_0}(X)}, \lambda_{CIPS} \right\} \cdot y_k$ |
| Self-normalized IPS | $\hat{R}_{SNIPS} = \left( \sum_{k=1}^{\|O\|} \frac{e_{\pi}(X)}{e_{\pi_0}(X)} \right)^{-1} \sum_{k=1}^{\|O\|} \frac{e_{\pi}(X)}{e_{\pi_0}(X)} \cdot y_k$ |
| Doubly robust learning | Combines propensity weighting with outcome imputation to reduce variance; the surveys treat it as consistent when either the propensity model or the imputation model is correct. |
| Back-door adjustment | $P(y \mid do(x)) = \sum_z P(y \mid x, z) P(z)$ |
| Front-door adjustment | $P(y \mid do(x)) = \sum_k P(k \mid x) \sum_{x'} P(y \mid x', k) P(x')$ |
| Natural Direct Effect | $NDE(u_i, v_j^*, f_j^b) = r_{ij}(u_i, v_j^*, f_j^b) - r_{ij}(u_i, v_j^*, f_j^{b*})$ |

IV4Rec is the recurring instrumental-variable example: search queries act as instruments, and the residual after removing display-position artifacts is treated as the causal preference component.

### 1.3 Variable Roles and Bias Types
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Role | Meaning |
| --- | --- |
| Confounder | Affects both treatment and outcome and creates spurious correlations. |
| Collider | Is affected by multiple causes; conditioning on it induces artificial dependence. |
| Mediator | Lies on the causal path from treatment to outcome. |
| Instrument | Affects the outcome only through treatment. |

| Bias type | Description in the source summaries |
| --- | --- |
| Selection bias | Ratings or interactions are missing-not-at-random. |
| Exposure bias | Item visibility is non-random. |
| Conformity bias | Group behavior or social influence is mistaken for preference. |
| Popularity bias | Popular items are over-exposed and over-recommended. |
| Position bias | Ranking position alters interaction probability. |
| Feedback-loop bias | Recommendations change future behavior and future training data. |
| Clickbait bias | Surface features attract clicks independently of underlying content quality. |
| Unfairness bias | Sensitive attributes causally affect learned representations or outcomes. |

The surveys also identify assumption fragility around positivity, spill-over effects in social networks, and dynamic feedback in online environments.

## 2. Additive Causal Model and Disentanglement Logic
Source: [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md); [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md)

Most causal embedding methods decompose the interaction score into interest and conformity components:

$$S_{ui} = S_{ui}^{int} + S_{ui}^{con}$$

Here $S_{ui}^{int}$ denotes genuine user interest and $S_{ui}^{con}$ denotes conformity or popularity bias. The summaries tie this decomposition to a colliding-effect view: the observed click is treated as a collider of interest and conformity, so observing clicks creates spurious dependence between those factors.

### 2.1 Interest-Dominant and Conformity-Dominant Partitioning
Source: [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md); [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md)

| Partition | Condition | Causal interpretation |
| --- | --- | --- |
| Interest-dominant set $O_2$ | User clicks an unpopular item over a popular item | High-confidence signal for genuine preference: $S_{uc}^{int} > S_{ud}^{int}$ and $S_{uc}^{con} < S_{ud}^{con}$ |
| Conformity-dominant set $O_1$ | User clicks a popular item over an unpopular item | Signal for conformity: $S_{ua}^{con} > S_{ub}^{con}$ while the total score still satisfies $S_{ua}^{total} > S_{ub}^{total}$ |

The standard ranking objective remains BPR:

$$\mathcal{L}_{BPR} = -\sum_{(u,i,j)} \ln \sigma(\hat{s}_{ui} - \hat{s}_{uj})$$

### 2.2 Regularization, Contrastive Learning, and Causal Separation
Source: [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Mechanism | Formula | Role |
| --- | --- | --- |
| Cosine discrepancy loss | $\mathcal{L}_{dis} = \cos(q^{int}, h^{con})$ | Pushes disentangled interest and conformity directions apart. |
| Orthogonality loss | $\mathcal{L}_{Orth} = \frac{1}{\|\mathcal{V}_u\|} \sum_{v \in \mathcal{V}_u} \max\left(0, \frac{\overline{e} \cdot \tilde{e}}{\lVert \overline{e} \rVert \cdot \lVert \tilde{e} \rVert}\right)$ | Separates filtered and biased embeddings in FMMRec. |
| Distance correlation | No explicit closed form is repeated in the summaries, but it is described as stronger than cosine similarity for enforcing independence. | Used in DICE to separate interest and conformity distributions. |
| Interest contrastive loss | $L_{int} = -\frac{1}{N} \sum \log \frac{\exp(-I_{pop}) \times \exp(S(E_{int}^u, E_{int}^{i+}))}{\exp(S(E_{int}^u, E_{int}^{i+})) + \sum \exp(S(E_{int}^u, E_{int}^{i-}))}$ | Upweights long-tail or niche preferences. |
| Conformity contrastive loss | $L_{conf} = -\log \frac{(1 - \exp(-I_{pop})) \times \exp(S(E_{conf}^u, E_{conf}^{i+}))}{\exp(S(E_{conf}^u, E_{conf}^{i+})) + \sum \exp(S(E_{conf}^u, E_{conf}^{i-}))}$ | Isolates popularity-driven behavior. |

### 2.3 Specialized Operations Used by Later Methods
Source: [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md); [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md)

| Technique | Formula | Role |
| --- | --- | --- |
| ZCA whitening | $e_v^{(0)} = ZCA(\tilde{e}_v)$ | Decorrelates modality features before propagation. |
| DDCE item popularity | $y_{pop} = \tanh(\langle q_v, c_v \rangle)$ | Models popularity as the interaction of item quality and conformity. |
| DDCE user-interest integration | $u_{int} = \tau \cdot L_u^t + (1 - \tau) S_u^t$ | Blends long-term and short-term interest with learned weight $\tau$. |
| MCLN purification | $e_{cl} = (A - A^*) \cdot V_i$ | Subtracts shared real and counterfactual features to isolate causal preference cues. |
| FMMRec role-indicator embeddings | $r_u, r_v$ | Mark user and item roles during filtering and relation mining. |
| Minimal counterfactual explanation | $\min \lVert \Delta \rVert_2^2 + \lambda \lVert \Delta \rVert_0$ | Finds the smallest perturbation that flips a recommendation. |
| Average Causal Effect for explanation | $ACE = \Pr(y=1 \mid do(x=1)) - \Pr(y=1 \mid do(x=0))$ | Measures explanation validity under intervention. |

## 3. Survey Consensus, Benchmarks, and Evaluation
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [notes_by_paper_10.md](../../../docs/paper_summaries/notes_by_paper_10.md); [summary_propcore.md](../../../docs/paper_summaries/summary_propcore.md)

### 3.1 Survey Scope and Taxonomy
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md)

| Survey | Published | Main framing |
| --- | --- | --- |
| Causal Inference for Recommendation: Foundations, Methods, and Applications | 2025-04 | Foundations, methods, applications, causal discovery, privacy, LLM integration |
| A Survey on Causal Inference for Recommendation | 2024-07 | Theory-driven taxonomy, uplift, OOD, fairness, explainability |
| Causal Inference in Recommender Systems: A Survey and Future Directions | 2023-12 | Debiasing, data augmentation, beyond-accuracy objectives, feedback loops, online gap |
| Causal Inference in Recommender Systems: A Survey of Strategies for Bias Mitigation, Explanation, and Generalization | 2023-01 | Causal debiasing, causal explanation, causal generalization |

Recurring problem families: explainable recommendation, fairness, uplift or heterogeneous effects, robust or out-of-distribution recommendation, unbiased or debiased recommendation, data augmentation and denoising, and causal generalization.

### 3.2 Frequently Cited Datasets
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [notes_by_paper_10.md](../../../docs/paper_summaries/notes_by_paper_10.md)

| Dataset family | Examples and density cues | Typical use |
| --- | --- | --- |
| Large sparse interaction graphs | Amazon, Yelp, Taobao, Gowalla, Tmall; Beauty 0.00103, Art 0.00086, Taobao 0.00076, Gowalla $4.0\times10^{-4}$, Amazon-Book $3.7\times10^{-4}$, Tmall $1.2\times10^{-4}$ | Ranking, debiasing, graph-based representation learning |
| Heterogeneous interaction data | MovieLens, Netflix, Douban; MovieLens-HetRec 4.0%, Douban Book 0.27%, Douban Movie 0.63% | Collaborative filtering with metadata or social links |
| Randomized or unbiased exposure data | Coat, Yahoo! R3, KuaiRand, KuaiRec; KuaiRec 16.8494% density in the reported fully-observed setup | Validation of debiasing, uplift, and causal ranking |
| Multimodal fairness data | MovieLens 1M, MicroLens; MovieLens 95.53% sparsity, MicroLens 99.83% sparsity | Fairness and multimodal representation learning |
| Semi-simulated causal data | PROPCARE evaluation setting | Exposure and uplift evaluation without original exposure logs |

### 3.3 Evaluation Metrics
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [notes_by_paper_10.md](../../../docs/paper_summaries/notes_by_paper_10.md); [summary_propcore.md](../../../docs/paper_summaries/summary_propcore.md)

| Objective | Metrics explicitly used in the sources |
| --- | --- |
| Ranking accuracy | Recall@K, NDCG@K, HR@K, AUC, Precision, MAE, MSE |
| Causal effect estimation | ATE, ATT, CATE, ITE |
| Uplift | Qini, AUUC, CP@K, CDCG |
| Fairness | AUC and micro-averaged F1 of surrogate attackers on sensitive attributes |
| Explanation | ACE, model fidelity, minimal perturbation complexity |
| Debiasing side-effect checks | Novelty@K, popularity-consistency statistics |

Standard ranking metrics can improve while causal quality degrades, especially under popularity bias.
DDCE adds Novelty@K to the ranking view; PROPCARE defines $CP@K$ and $CDCG$ on semi-simulated counterfactual data.

## 4. Graph Backbones and Hybrid Architectures
Source: [gcn_models.md](../../../docs/paper_summaries/gcn_models.md); [summary_hybrid_transGNN.md](../../../docs/paper_summaries/summary_hybrid_transGNN.md)

### 4.1 LightGCN, LayerGCN, and LightGCN++
Source: [gcn_models.md](../../../docs/paper_summaries/gcn_models.md)

| Model | Core operator or rule | Reported configuration, guardrails, and limits |
| --- | --- | --- |
| LightGCN | $E^{(k+1)} = (D^{-1/2} A D^{-1/2}) E^{(k)}$, $e_u = \sum_{k=0}^{K} \alpha_k e_u^{(k)}$, uniform $\alpha_k = 1/(K+1)$ | Xavier initialization; $d=64$; Adam with learning rate 0.001; batch size 1024 on Gowalla and Yelp2018 and 2048 on Amazon-Book; layer combination stabilizes performance relative to single-layer output; removes feature transformation matrices, nonlinear activation, self-connections, and dropout. |
| LayerGCN | Sparsify edges with $p_{e_k} = \frac{1}{\sqrt{d_i}\sqrt{d_j}}$, propagate linearly, refine by ego-layer similarity, then read out hidden layers without the ego layer | Degree-sensitive pruning, cosine-similarity refinement, sum aggregation, dot-product prediction, default depth 4, Adam, edge-dropout ratios $\{0.0, 0.1, 0.2\}$, early stopping patience 50; LayerGCN-4 exceeds LightGCN-3 on MOOC with R@20 0.3979 versus 0.3271 and N@20 0.2272 versus 0.1929. |
| LightGCN++ | $e_i^{(k+1)} = \frac{1}{\|\mathcal{N}_i\|^{\alpha}} \sum_{u \in \mathcal{N}_i} \frac{1}{\|\mathcal{N}_u\|^{\beta}} e_u^{(k)}$, $e_i = \gamma e_i^{(0)} + (1-\gamma) \frac{1}{K} \sum_{k=1}^{K} e_i^{(k)}$ | Replaces fixed normalization with tunable $\alpha$, $\beta$, and $\gamma$; requires embedding normalization at every layer; $d=64$, $B=2048$, $K=2$, learning rate 0.001, regularization 0.0001; time complexity $O(\|E\| + \|E\|Kd + \|V\|Kd)$; runtime increase 0.08% to 5.29% per epoch over LightGCN. |

### 4.2 TransGNN and SIGformer
Source: [summary_hybrid_transGNN.md](../../../docs/paper_summaries/summary_hybrid_transGNN.md)

| Model | Architecture sequence | Loss, configuration, and complexity |
| --- | --- | --- |
| TransGNN | Semantic similarity $S = XX^\top$, refined similarity $S = S + \alpha \hat{A} S$, top-k attention sampling, positional encoding with SPE/DE/PRE, transformer layer, GNN layer, dynamic sample update | Pairwise rank loss; three Transformer layers with two GNN layers sandwiched between them; preprocessing complexity up to $O(N(N+E)\log E)$; hardware note: NVIDIA A100 SXM4 80GB; removal of the GNN layer causes the largest ablation drop. |
| SIGformer | Identity projection to $Q$, $K$, $V$; sign-aware spectral and path encodings; attention-based layer updates; layer averaging; dot-product prediction | Modified BPR over positive and negative feedback; $L=3$, $d=64$, $d_h=64$, learning rate $1e^{-2}$, weight decay $1e^{-4}$; complexity $O((n+m)d\hat{N})$; negative interactions are critical in the reported ablations. |

## 5. Method Families and Representative Models
Source: [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [notes_by_paper_10.md](../../../docs/paper_summaries/notes_by_paper_10.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md); [summary_propcore.md](../../../docs/paper_summaries/summary_propcore.md); [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md)

### 5.1 Disentanglement and Popularity-Debiasing Models
Source: [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [notes_by_paper_10.md](../../../docs/paper_summaries/notes_by_paper_10.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Method | Core mechanism | Losses or scores | Reported data, metrics, or scaling notes |
| --- | --- | --- | --- |
| DICE | Separate user and item embeddings into interest and conformity vectors; additive score $s_{ui}^{click} = s_{ui}^{int} + s_{ui}^{con}$; popularity-based negative sampling with margin isolates high-confidence triplets | Three BPR losses for interest, conformity, and click; discrepancy loss with L1-inv, L2-inv, or dCor | MovieLens-10M and Netflix; dCor costs about 100 seconds per epoch versus about 44 seconds for L1-inv and L2-inv; non-IID evaluation uses inverse-popularity intervention; curriculum weights and margins decay by 0.9 per epoch. |
| DDCE | Splits user interest into long-term and short-term components and item popularity into quality and conformity components, with $u_{int} = \tau \cdot L_u^t + (1 - \tau) S_u^t$ and $y_{pop} = \tanh(\langle q_v, c_v \rangle)$ | Popularity loss and interest disentanglement losses; final score is the sum of user-interest score and item-popularity score | Douban Movie and KuaiRec; metrics include NDCG, HR, Recall, and Novelty; reported to outperform IPS, CausE, DICE, MACR, and PDA on both MF and LightGCN backbones. |
| DCCL | Disentangles interest and conformity embeddings and optimizes them with popularity-aware contrastive learning | $\mathcal{L}_{int}$, $\mathcal{L}_{conf}$, and the main recommendation loss | Yelp and industrial short-video data; explicit training complexity $O(B^2 d)$; inference complexity is unchanged relative to the backbone. |
| CaDCR | Bipartite interaction graph, LightGCN-style encoder, $K$ intent blocks, causal intervention, and multi-task curriculum learning | $\mathcal{L} = \mathcal{L}_{BPR} + \lambda_1(\mathcal{L}_{cl} + \mathcal{L}_{it}) + \lambda_2 \Theta^2$ and $P(Y\mid do(U=u)) - P(Y\mid do(U=0)) = \frac{1}{N} \sum_{i=1}^{N}(P(\hat{y}_{ui}\mid i \odot H) - P(\hat{y}_{ui}\mid i))$ | Gowalla, Amazon-Book, and Tmall; reported depth $L=3$, embedding size 64, batch size 4096, Adam optimizer, time per epoch of about 8 seconds on Gowalla and about 10 seconds on Amazon-Book and Tmall, and intent-granularity sensitivity when $K$ becomes large. |

### 5.2 Multimodal and Fairness-Oriented Models
Source: [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [notes_by_paper_10.md](../../../docs/paper_summaries/notes_by_paper_10.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Method | Core mechanism | Losses or scores | Reported data, metrics, or scaling notes |
| --- | --- | --- | --- |
| MGCE | GCN-based multimodal interest and multimodal conformity branches; ZCA-whitened VGG16 visual features and Sentence2Vec textual features; popularity embeddings injected into the conformity branch | BPR losses for interest, conformity, and base signals; discrepancy loss between interest and conformity embeddings | Beauty, Art, and Taobao; interest graph depth $n=1$, conformity graph depth $k=2$, embedding size 64, batch size 2048; sensitive to multimodal noise and over-smoothing. |
| MCLN | Linear GCN for collaborative signals plus counterfactual preference comparison between interacted and uninteracted items, with purification through $(A - A^*)$ | Base BPR loss and multimodal loss; causal effect written as $Y_{effect} = Y_{u,i,a} - Y_{u,i,a^*}$ | Beauty, Art, and Taobao; graph layers optimized at 4 or 5, counterfactual layers at 2 or 4; counterfactual attention introduces $O(d_x^2)$ overhead and deeper stacks interfere with learning. |
| FMMRec | Filter network and biased learner split modal embeddings into fair and unfair components; role-indicator embeddings distinguish user and item roles; fair and unfair relation mining are constructed by kNN sparsification | Recommendation loss, sensitive-attribute prediction losses, and orthogonality loss; final adjustment $\hat{e}_u = e_u + \lambda_h(\overline{h}_u - \tilde{h}_u)$ | MovieLens and MicroLens; fairness is evaluated with attacker AUC and micro-averaged F1; kNN sparsification reduces user-relation complexity from dense similarity matrices to $O(|\mathcal{U}|d^2 + |\mathcal{V}|d^2)$. |

### 5.3 Semantic and Explanation Models
Source: [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Method | Core mechanism | Losses or scores | Reported data, metrics, or scaling notes |
| --- | --- | --- | --- |
| CaDSI | Heterogeneous skip-gram for semantic-aware context embeddings, intent-aware graph disentangling, and back-door adjustment over aspect contexts | Deconfounded preference estimation $P(Y\mid do(U=u)) = \frac{1}{N} \sum_{i=1}^{N} P(\hat{y}_{ui} \mid u^u \odot c_{a_i})$ and BPR optimization | MovieLens-HetRec, Douban Book, and Douban Movie; reported graph layers $L=2$, latent intent number $k=4$, intervention iterations $n=140$, meta paths such as UMU, UMAMU, UMDMU, UMCMU, and UMGMU on MovieLens-HetRec, and bottlenecks from meta-path random walks and intervention over aspect types. |
| Learning Causal Explanations | Variational auto-encoder perturbation model generates counterfactual user histories and ranks candidate explanations by causal dependency and ACE | ACE, fidelity, and minimal perturbation complexity $\min \lVert \Delta \rVert_2^2 + \lambda \lVert \Delta \rVert_0$ | MovieLens 100K and Amazon Office; uses 500 counterfactual samples, time decay $\gamma = 0.7$, and two 1024-unit hidden layers; CR-VAE reaches about 96.5% fidelity on MovieLens versus about 16% for association rules; exhaustive counterfactual search is infeasible without sampling. |
| PRINCE | Knowledge-graph or HIN search for minimal action sets that reverse recommendations | Survey-level explanation benchmark, not expanded into one fixed loss in the provided summaries | The surveys emphasize explainability strength and also note I/O and latency costs from large random-walk style operations. |
| CountER | Counterfactual explanation by minimal feature perturbation and causal rule discovery | Survey-level explanation benchmark, often paired with PRINCE | The surveys use it as a state-of-the-art counterfactual explanation reference. |

### 5.4 Treatment-Control and Propensity Models
Source: [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [summary_propcore.md](../../../docs/paper_summaries/summary_propcore.md); [methematical_formulations.md](../../../docs/paper_summaries/methematical_formulations.md)

| Method | Core mechanism | Losses or scores | Reported data, metrics, or scaling notes |
| --- | --- | --- | --- |
| CausE | Learns separate treatment and control representations and regularizes their discrepancy to transfer evidence from biased logs to randomized logs | Multi-task objective with discrepancy regularizer over treatment and control representations; focuses on ITE-aligned prediction | MovieLens 10M, Netflix, and MovieLens 100K; requires a sample from a randomized policy, which is the main practical limit; variance and randomized-data sparsity are explicit limitations. |
| PROPCARE | Learns propensity and relevance jointly from ordinary interaction logs, then infers exposure labels for downstream causal ranking | $y_{u,i} = p_{u,i} r_{u,i}$, $\mathcal{L}_{naive}$, popularity-guided $\mathcal{L}_{pop}$, global $KL(Q \parallel Beta(\alpha, \beta))$, exposure prediction $\hat{Z}_{u,i}$, DLCE loss, causal metrics $CP@K$ and $CDCG$, popularity-consistency statistic $ratio_b$, and explicit bias decomposition | Semi-simulated counterfactual evaluation; treats popularity as an exposure prior; clipping thresholds $\chi_1$ and $\chi_0$ stabilize ranking; training time is described as linear in the number of user-item pairs. |

## 6. Computational Bottlenecks, Training Regimes, and ANN Acceleration
Source: [summary_performance_papers.md](summary_performance_papers.md); [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [gcn_models.md](../../../docs/paper_summaries/gcn_models.md); [summary_hybrid_transGNN.md](../../../docs/paper_summaries/summary_hybrid_transGNN.md)

### 6.1 Recurring Bottlenecks Across the Literature
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [summary_performance_papers.md](summary_performance_papers.md)

| Bottleneck | Source-backed characterization |
| --- | --- |
| IPS variance | Estimated propensities near zero create instability and high variance; clipping, trimming, and doubly robust learning are the recurring stabilizers. |
| Propensity correctness | No direct quantitative method confirms estimated propensity correctness on observational data. |
| Randomized-data scarcity | RCT-style data are expensive, sparse, and often too small for complex models. |
| Contrastive batch cost | DCCL-style contrastive objectives increase batch complexity to $O(B^2 d)$. |
| HIN preprocessing | Meta-path random walks, heterogeneous skip-gram, and explanation search introduce I/O and latency costs. |
| Multimodal noise | MGCE and MCLN report degradation when multimodal graph depth is increased too far. |
| Over-smoothing | LightGCN, LayerGCN, MGCE, MCLN, and CaDCR all report depth limits or degradation from overly deep propagation. |
| Pre-defined causal graphs | The surveys repeatedly identify manually defined causal graphs as oversimplified and inconsistent across problems. |
| Spill-over effects and dynamic feedback | Social networks and online deployment create temporal and cross-user dependencies that the surveys treat as unresolved. |

### 6.2 Full-Graph Versus Mini-Batch Training
Source: [summary_performance_papers.md](summary_performance_papers.md)

Full-graph training is the limit case with maximal batch size $b$ and fan-out size $\beta$.

| Finding | Source-backed statement |
| --- | --- |
| Convergence under MSE | Increasing batch size can require more iterations because larger batches introduce structural bias. |
| Convergence under CE | Increasing batch size reduces iterations more in line with conventional deep learning. |
| Generalization | Increasing either $b$ or $\beta$ generally improves generalization until overly large values degrade it, particularly under CE. |
| Fan-out effect | Increasing fan-out generally reduces required iterations under both MSE and CE. |
| Sensitivity split | Convergence is more sensitive to batch size; generalization is more sensitive to fan-out size. |
| Throughput | Larger batch size improves throughput; larger fan-out reduces throughput. |
| Practical threshold | For sparse graphs, the paper recommends keeping batch size below half of training nodes and fan-out under 15. |
| Hardware-agnostic validation | Iteration-to-accuracy is more reliable than time-to-accuracy for early configuration decisions. |

### 6.3 CAGRA and PANORAMA
Source: [summary_performance_papers.md](summary_performance_papers.md)

| System paper | Main contribution and mechanics | Reported limits and regimes |
| --- | --- | --- |
| CAGRA | GPU-oriented graph construction and approximate nearest-neighbor search; 2.2x to 27x faster graph construction than HNSW and 33x to 77x faster large-batch query throughput in the reported range; uses warp splitting, fixed out-degree 32 to 96, internal top-$M$ search, and rank-based reordering with reverse-edge addition | Memory-bandwidth bound, device-memory limited, sensitive to dimensionality and register pressure, and capped at $2^{31}-1$ nodes on the reported implementation path |
| PANORAMA | Learned orthogonal transform through a Cayley transform plus progressive pruning and accretive refinement; 2x to 30x end-to-end speedups with no recall loss in the reported setup; strongest gains reach 2x to 40x on IVFFlat and 2x to 30x on IVFPQ under AVX-512 execution | Additional memory overhead $O(nL)$, diminishing returns after too many refinement levels, smaller gains for non-contiguous layouts, and L3-cache or data-movement limits in graph and tree indexes |

## 7. Open Problems Reported Across the Sources
Source: [summary_survey_papers_4.md](../../../docs/paper_summaries/summary_survey_papers_4.md); [summary_by_paper_10.md](../../../docs/paper_summaries/summary_by_paper_10.md); [summary_performance_papers.md](summary_performance_papers.md)

| Problem | Source-backed description |
| --- | --- |
| Causal discovery | All four surveys identify the transition from expert-defined graphs to learned causal relations as a major open direction. |
| Propensity verification | Propensity correctness remains hard to validate directly on observational data because no direct quantitative method is available. |
| Dynamic feedback and online deployment | The 2023 and 2025 surveys emphasize feedback loops, the online gap, and causality-supported simulators. |
| Spill-over effects in social networks | The 2023 and 2023-01 surveys identify cross-user social effects as unresolved and as pressure on SUTVA and positivity. |
| Out-of-distribution robustness | The surveys repeatedly frame stable recommendation under domain or temporal shift as an open problem. |
| Multimodal noise and depth sensitivity | MGCE, MCLN, and FMMRec all report quality or stability limits tied to modality noise, sparsity, dense relation mining, or deeper multimodal stacks. |
| Fine-grained causal factors | DICE and CaDCR both identify coarse causal partitions as a limitation and call for finer-grained causal structure. |
| Fairness-accuracy trade-off | FMMRec explicitly reports that reducing sensitive-attribute leakage can reduce recommendation accuracy. |
| Causal privacy and federated learning | The 2025 survey raises data minimization and privacy-preserving causal inference as future directions. |
| Causal-LLM integration | The 2025 and 2024 surveys identify LLM-based recommendation and causal integration as an emerging direction. |