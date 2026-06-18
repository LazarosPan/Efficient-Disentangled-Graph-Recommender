# PropCare Technical Audit (Evaluator-Centered, Theory-to-Code Blueprint)

## Mechanical Summary Table

| Spec | Extracted implementation | Evidence |
|---|---|---|
| Framework/runtime | TensorFlow 2.x for propensity estimation, NumPy/Pandas for ranking and evaluation | `external/PropCare/README.md:15-22`, `external/PropCare/models.py:12-164`, `external/PropCare/evaluator.py:1-170` |
| Overall pipeline | Two-stage pipeline: PropCare propensity model -> thresholded exposure/propensity rewrite -> DLMF ranker -> causal evaluator | `external/PropCare/main.py:37-128`, `external/PropCare/train.py:40-140`, `external/PropCare/baselines.py:609-721` |
| Primary evaluation object | Pre-scored candidate dataframe with user, item, prediction, outcome, treatment, propensity, and causal effect columns | `external/PropCare/evaluator.py:5-18`, `external/PropCare/evaluator.py:53-112`, `external/PropCare/train.py:80-88` |
| Main reported metrics | `CPrec@10`, `CPrec@100`, and `CDCG` only | `external/PropCare/README.md:9-10`, `external/PropCare/main.py:106-116` |
| Evaluator surface beyond README | Also implements IPS-based variants, rank-based causal metrics, hit/NDCG/AUC, and causal AUC variants | `external/PropCare/evaluator.py:63-112`, `external/PropCare/evaluator.py:115-170` |
| Causal estimate used for IPS variants | $\hat{\tau}_{u,i} = y_{u,i}\left(\frac{z_{u,i}}{p_{u,i}} - \frac{1-z_{u,i}}{1-p_{u,i}}\right)$ computed row-wise in the evaluator | `external/PropCare/evaluator.py:62-65` |
| Propensity stabilization | Capping is active in evaluator and DLMF training; clipping helper exists but is disabled in evaluator | `external/PropCare/evaluator.py:31-50`, `external/PropCare/evaluator.py:56-58`, `external/PropCare/baselines.py:614-621` |
| Downstream ranker actually used | `DLMF`, despite README shorthand about DLCE/other baselines | `external/PropCare/main.py:4`, `external/PropCare/main.py:93-100`, `external/PropCare/baselines.py:609-721` |
| Local thesis evaluator contrast | Current repository evaluator is GPU-batched PyG link-prediction evaluation, not causal-uplift evaluation | `src/training/evaluator.py:21-89` |

## 1. End-to-End Evaluation Flow

### 1.1 What the evaluator receives

`Evaluator` is not an end-to-end recommender evaluator that generates candidates itself. It expects a dataframe whose rows already represent scored user-item candidates. The constructor fixes the schema names for:

- user and item identifiers
- observed outcome
- model prediction
- treatment indicator
- propensity
- ground-truth causal effect
- optional estimated causal effect

Evidence: `external/PropCare/evaluator.py:5-18`

This matters for replication: the evaluator is only the last stage of the method. Everything upstream must first produce:

1. a candidate set per user,
2. a ranking score in `pred`,
3. treatment and propensity columns, and
4. a semi-simulated `causal_effect` target if you want paper-style causal evaluation.

The upstream data preparation path preserves `propensity` and `treated` in the training dataframe and leaves validation/test data loaded from the semi-simulated CSVs, while `main.py` later attaches `pred` to each test slice before calling the evaluator.

Evidence: `external/PropCare/train.py:40-88`, `external/PropCare/main.py:101-116`

### 1.2 Evaluation control flow in code

The evaluator pipeline is mechanically simple:

```python
def evaluate(self, df_origin, measure, num_rec, mode = 'ASIS', cap_prop=0.0):
	df = df_origin.copy(deep=True)
	df = self.capping(df, cap_prop)
	# df = self.clip(df, cap_prop)
	df = self.get_sorted(df)
	self.rank_k = num_rec

	if 'IPS' in measure:
		df.loc[:, self.colname_estimate] = df.loc[:, self.colname_outcome] * \
									(df.loc[:, self.colname_treatment] / df.loc[:,self.colname_propensity] - \
									 (1 - df.loc[:, self.colname_treatment]) / (1 - df.loc[:, self.colname_propensity]))
```

Evidence: `external/PropCare/evaluator.py:53-65`

The sequence is:

1. deep-copy the input dataframe,
2. optionally cap propensity values depending on treatment status,
3. sort rows by `(user, pred)` descending,
4. set the global `rank_k`,
5. optionally construct an IPS-style causal estimate column,
6. dispatch to a metric-specific reducer.

Two small implementation details matter for a faithful port:

- `mode='ASIS'` is accepted but unused.
- `clip()` exists but is commented out in favor of asymmetric treated/control capping.

Evidence: `external/PropCare/evaluator.py:45-50`, `external/PropCare/evaluator.py:53-58`

## 2. From Paper Formula to Evaluator Code

### 2.1 Paper semantics

The paper summary defines the central causal metrics as:

$$
CP@K = \frac{1}{U}\sum_{u=1}^{U}\sum_{i=1}^{I}\frac{\mathbb{1}(rank_u(\hat{s}_{u,i}) \le K)\tau_{u,i}}{K}
$$

and

$$
CDCG = \frac{1}{U}\sum_{u=1}^{U}\sum_{i=1}^{I}\frac{\tau_{u,i}}{\log_2(1 + rank_u(\hat{s}_{u,i}))}
$$

with an IPS-style causal estimate

$$
\hat{\tau}_{u,i} = \frac{Z_{u,i}Y_{u,i}}{p_{u,i}} - \frac{(1-Z_{u,i})Y_{u,i}}{1-p_{u,i}}.
$$

Evidence: `docs/paper_summaries/summary_propcore.md:93-133`

### 2.2 IPS estimate path

The evaluator implements the IPS estimator almost literally. If the metric name contains `IPS`, it constructs:

```python
df.loc[:, self.colname_estimate] = df.loc[:, self.colname_outcome] * \
							(df.loc[:, self.colname_treatment] / df.loc[:,self.colname_propensity] - \
							 (1 - df.loc[:, self.colname_treatment]) / (1 - df.loc[:, self.colname_propensity]))
```

Evidence: `external/PropCare/evaluator.py:62-65`

This is the evaluator’s row-wise estimate of causal effect. It is later aggregated either as a top-$K$ mean (`CPrecIPS`), a discounted cumulative score (`CDCGIPS`), or rank-sensitive variants (`CARIPS`, `CARPIPS`, `CARNIPS`).

Important constraint: the implementation uses whatever `treated` and `propensity` values are present in the dataframe at evaluation time. In this repository, those columns are partly generated by the PropCare stage in `main.py` for training, but test evaluation still depends on the semi-simulated dataset carrying the right causal columns.

Evidence: `external/PropCare/main.py:66-80`, `external/PropCare/evaluator.py:71-91`

### 2.3 Causal Precision in code

The code path for `CPrec` is:

```python
elif measure == 'CPrec':
	df_ranking = self.get_ranking(df, num_rec=num_rec)
	return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
```

Evidence: `external/PropCare/evaluator.py:71-73`

Mechanically this means:

1. sort all candidate rows per user by predicted score,
2. keep the first `K` rows per user,
3. average the `causal_effect` values of those retained rows.

This is close to the paper’s $CP@K$ intention, but the exact weighting is implementation-specific:

- if every user contributes exactly $K$ ranked candidates, the code is equivalent to averaging the top-$K$ causal effects across users;
- if users contribute different numbers of candidate rows, users with more retained rows contribute more mass to the final `np.nanmean`.

In the original experimental setting this likely stays aligned because candidate-set construction is standardized upstream, but this weighting detail matters in a later PyTorch replication.

### 2.4 Causal DCG in code

The code path for `CDCG` is:

```python
elif measure == 'CDCG':
	return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.dcg_at_k})))
```

with

```python
def dcg_at_k(self, x):
	k = min(self.rank_k, len(x))
	return np.sum(x[:k] / np.log2(np.arange(k) + 2))
```

Evidence: `external/PropCare/evaluator.py:77-80`, `external/PropCare/evaluator.py:119-121`

This is the cleanest theory-to-code match in the file:

- rows are already sorted by predicted score,
- `dcg_at_k` applies the standard logarithmic discount,
- the outer `groupby(...).agg(...)` computes the per-user discounted causal gain,
- `np.nanmean` averages those user-level scores.

In `main.py`, `CDCG` is always called with `num_rec=100000`, which effectively means “use the full ranked list unless a user has more than 100000 candidates.”

Evidence: `external/PropCare/main.py:108`, `external/PropCare/main.py:116`

## 3. What the Evaluator Actually Implements

### 3.1 Metric families

Although the README describes the evaluator as computing only `CP@10`, `CP@100`, and `CDCG`, the file exposes a much broader metric surface:

- relevance-only: `precision`, `Prec`, `DCG`, `NDCG`, `hit`, `AUC`
- causal with ground-truth effect: `CPrec`, `CDCG`, `CAR`, `CARP`, `CARN`, `CAUC`, `CAUCP`, `CAUCN`
- causal with IPS estimate: `CPrecIPS`, `CDCGIPS`, `CARIPS`, `CARPIPS`, `CARNIPS`

Evidence: `external/PropCare/README.md:9-10`, `external/PropCare/evaluator.py:66-112`

This is a strong sign that `evaluator.py` is partly inherited from a broader causal-ranking codebase rather than written only for the three metrics mentioned in the paper-facing README.

### 3.2 Rank-based and AUC-based helpers

The helper functions are simple array reducers:

- `prec_at_k()` computes top-$K$ mean of binary outcomes,
- `dcg_at_k()` computes discounted cumulative gain,
- `ndcg_at_k()` normalizes by sorted ideal DCG,
- `auc()` computes pairwise ranking quality from positive positions,
- `gauc()` generalizes AUC to ternary effect labels by splitting positive and negative causal outcomes,
- `ave_rank()` computes the average rank weighted by signed values.

Evidence: `external/PropCare/evaluator.py:115-170`

For the causal AUC path, the code interprets positive causal effects and negative causal effects separately:

```python
def gauc(self, x):
	x_p = x > 0
	x_n = x < 0
	...
	if num_p > 0:
		gauc += self.auc(x_p) * (num_p/(num_p + num_n))
	if num_n > 0:
		gauc += (1.0 - self.auc(x_n)) * (num_n/(num_p + num_n))
```

Evidence: `external/PropCare/evaluator.py:146-156`

This reflects the causal-ranking objective well: a good ranking should elevate positive-effect items and bury negative-effect items.

## 4. Upstream Pipeline Context Required to Interpret Evaluation

### 4.1 Data ingestion and schema

`prepare_data()` hard-codes dataset paths, loads three CSV splits, remaps user and item IDs jointly across train/validation/test, and keeps the core training schema:

```python
train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
```

Evidence: `external/PropCare/train.py:40-88`, `external/PropCare/train.py:80`

The evaluator needs more than this training schema at test time, because it also expects:

- `pred`, which is added later by the downstream ranker,
- `causal_effect`, which must already exist in the semi-simulated evaluation data.

The repository README explicitly states that the user must generate semi-simulated datasets from external sources before this pipeline works.

Evidence: `external/PropCare/README.md:24-30`

### 4.2 Propensity estimation stage

The TensorFlow model predicts click, propensity, and relevance jointly:

- `call()` returns `(click, propensity, relevance, film_reg_loss)`
- `propensity_train()` combines click BCE, popularity-guided pairwise separation, and Beta-distribution KL regularization.

Evidence: `external/PropCare/models.py:88-132`, `external/PropCare/models.py:134-164`

Validation of the propensity stage is based on Kendall-$\tau$ between predicted and true propensities, not on downstream ranking metrics.

Evidence: `external/PropCare/train.py:91-137`, `external/PropCare/train.py:129`

### 4.3 Exposure thresholding and score generation

After propensity prediction, `main.py` post-processes it into estimated exposure and stabilized propensities:

```python
if flag.dataset == "d" or "p":
	flag.thres = 0.70
elif flag.dataset == "ml":
	flag.thres = 0.65
t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
...
train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
train_df["treated"] = t_pred
```

Evidence: `external/PropCare/main.py:70-80`

Then the downstream `DLMF` model trains on positive-outcome rows, forms an IPS-style `ITE` column internally, and later produces scalar ranking scores via matrix-factor dot products:

Evidence: `external/PropCare/baselines.py:609-709`

This is the direct handoff into evaluation: `main.py` slices the test dataframe by time, calls `recommender.predict(test_df_t)`, writes the result into `pred`, and then hands that dataframe to `Evaluator.evaluate()`.

Evidence: `external/PropCare/main.py:101-116`

## 5. Evaluation-Relevant Implementation Anomalies

### 5.1 Always-true dataset branches

The conditions

```python
if flag.dataset == "d" or "p":
```

appear in multiple places in `main.py`. In Python, this expression is always truthy because the string `"p"` is truthy. That means the Dunn/personalized branch logic is effectively always taken.

Evidence: `external/PropCare/main.py:70`, `external/PropCare/main.py:75`

For evaluator interpretation, this matters because the thresholding and propensity scaling logic feeding the downstream ranker are not behaving as the author likely intended.

### 5.2 Capping is used, clipping is not

The evaluator contains both `capping()` and `clip()` helpers, but only `capping()` is active:

```python
df = self.capping(df, cap_prop)
# df = self.clip(df, cap_prop)
```

Evidence: `external/PropCare/evaluator.py:56-58`

This matches the paper’s intent to stabilize inverse-propensity terms without symmetrically clipping every row. A faithful port should preserve that asymmetry first, then optionally expose generic clipping as a separate experiment.

### 5.3 Candidate-set semantics are external to the evaluator

Nothing in `evaluator.py` generates a full item ranking universe or filters previously seen items. It ranks only the rows already present in the input dataframe.

That means the evaluator’s meaning is inseparable from the experimental protocol used to construct test candidates. In the PropCare setting this is acceptable because the semi-simulated benchmark defines the candidate rows externally, but it would not be a drop-in replacement for the current repository’s full-item PyG evaluator.

### 5.4 README simplification vs actual code

The README frames the evaluator as if it only computes `CP@10`, `CP@100`, and `CDCG`, but the file implements a broader metric layer. For later replication work, the code should be treated as the source of truth, not the README summary.

Evidence: `external/PropCare/README.md:9-10`, `external/PropCare/evaluator.py:66-170`

## 6. PyTorch Port Blueprint for `evaluator.py`

### 6.1 Minimal correctness-first contract

For a first PyTorch replication, do not start by tensorizing everything. Keep the evaluator contract explicit and close to the original semantics.

Recommended minimal interface:

| Component | Current implementation | PyTorch-port recommendation |
|---|---|---|
| Input container | Pandas dataframe with named columns | Keep a dataframe or dataclass-backed table first; tensorize later |
| Sorting/ranking | `sort_values` + `groupby().head(K)` | Preserve exact grouped sorting semantics before optimizing |
| IPS estimate builder | Row-wise dataframe expression | Implement as a pure function on tensors or series, then attach as a column/view |
| Metric reducers | Per-user `groupby(...).agg(...)` with NumPy reducers | Port as isolated reducers; verify equality on toy data before batching |
| Stabilization | Asymmetric `capping()` | Preserve this exactly first; expose clipping only as an optional alternative |

### 6.2 Recommended module decomposition

For later translation, split the evaluator into four pieces:

1. `stabilize_propensity(rows, cap_prop)`
2. `build_effect_estimate(rows)` for IPS metrics
3. `rank_rows(rows, num_rec)` for grouped sorting and top-$K$ extraction
4. metric reducers:
   - top-$K$ row means: `Prec`, `CPrec`, `CPrecIPS`
   - per-user discounted reducers: `DCG`, `CDCG`, `CDCGIPS`, `NDCG`
   - rank-sensitive reducers: `AR`, `CAR`, `CARIPS`, `CARP`, `CARN`
   - pairwise-order reducers: `AUC`, `CAUC`, `CAUCP`, `CAUCN`

This decomposition mirrors the file’s actual organization and makes unit testing straightforward.

### 6.3 What can later be tensorized

Once correctness is verified against the original code, the following parts can be tensorized:

- row-wise IPS estimate construction,
- discounted gain computation,
- hit/precision reducers,
- grouped AUC if candidate counts are padded or segmented.

The least attractive part to tensorize early is segmented sort/top-$K$ over variable candidate counts. That is exactly where a correctness-first dataframe port is the safer first move.

### 6.4 Validation strategy for the later port

The safest replication sequence is:

1. port `capping()` and the IPS estimate formula,
2. port `get_ranking()` and `get_sorted()` semantics,
3. port `CPrec`, `CDCG`, and `CPrecIPS` first,
4. verify outputs on synthetic toy data and on one cached PropCare dataframe,
5. only then port the secondary metrics.

## 7. Comparison with the Current EDGRec Evaluator

The current repository’s evaluator has a completely different contract:

- it works over a PyG graph object, not a candidate dataframe,
- it asks the model for `get_all_scores(...)` over all items for a batch of users,
- it computes only PyG link-prediction metrics at configured $K$,
- it has no treatment, propensity, or causal-effect columns.

Evidence: `src/training/evaluator.py:21-89`

This means PropCare’s evaluator is useful as a conceptual template for a future causal-uplift evaluation layer, but it is not interchangeable with the current runtime evaluator. Integrating it here would require a different data interface and a causal benchmark protocol, not just a metric-name swap.

## 8. Bottom Line

The most interesting part of the PropCare repository is indeed `evaluator.py`, because it is where the paper’s causal-ranking thesis becomes mechanically testable.

What the file does well:

- it directly operationalizes causal uplift rather than raw relevance,
- it keeps the IPS estimate close to the paper’s formula,
- it separates ground-truth-effect metrics from IPS-estimated variants,
- it is compact enough to port faithfully.

What must be handled carefully in a later PyTorch replication:

- candidate-set construction lives outside the evaluator,
- branch bugs in `main.py` affect the meaning of evaluation inputs,
- row-level averaging in `CPrec` is only equivalent to the paper’s user-level formula under standardized candidate counts,
- the evaluator is designed for semi-simulated causal labels, not ordinary implicit-feedback test sets.

For a later translation effort, the correct target is not “rewrite the current file line by line in torch.” The correct target is “preserve its dataframe-level semantics exactly, validate on toy and cached data, and only then optimize the hot paths.”
