# Benchmark Dataset Summary

Generated from `src/data_exploration/explore_all_datasets.py` using the same
canonical statistics that power the benchmark figures.

## How to Read the Figures

- `benchmark_overview.png` is the cross-dataset scale and feedback-semantics figure. Dataset colors identify datasets only. Green is reserved for `label > 0` graph/relevance rows; teal/red/slate-gray show `sign > 0`, `sign < 0`, and `sign = 0` feedback rows.
- Each `*_profile.png` uses the same evidence order: entity/row scale, user long tail, item long tail, binary label versus graded sign, response signal or split, and the most relevant dataset-specific context.
- `label > 0` is the binary relevance signal used for positive graph edges and top-K relevance. `sign` is a separate graded feedback descriptor, drawn in a different color family. They often overlap, but KuaiRec and KuaiRand show why they should not be treated as identical.
- Constant or unavailable signals are not forced into one-bar charts. Exact counts and omitted-signal explanations are kept here and in `benchmark_summary.json`.

## Committee Questions Covered

| Likely question | Evidence to cite |
| --- | --- |
| How large and sparse is each dataset? | Overview scale/density panels plus the profile scale panel. |
| Are the datasets long-tailed? | User-activity and item-popularity CCDF panels in each profile. |
| What is a positive edge for the GCN? | Binary label row in each profile and the Chapter 3 graph-construction description. |
| Are negative and neutral feedback available? | Sign row in each profile and the sign panel in the overview. |
| Are label and sign the same thing? | Label/sign stacked bars and the overlap counts below; mismatches are explicitly reported as `label > 0 AND sign < 0`. |
| Why are side features used only for some datasets? | Feature availability/context panels and the safe-feature policy in Chapter 3. |
| Which plots are descriptive rather than performance evidence? | All figures in this directory are setup evidence; ranking claims require the results tables. |

## Dataset Table

| Dataset | Interactions | Pair reuse | Positive share | Timestamp coverage | Randomized share | User feat | Item feat | Split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Amazon-Book | 2,984,108 | 0.00% | 100.00% | 0.0% | - | 0 | 0 | provided train / test split with validation carved from train |
| MovieLens 1M | 1,000,209 | 0.00% | 57.52% | 100.0% | - | 3 | 18 | per-user temporal split derived from timestamps |
| MovieLens 20M | 20,000,263 | 0.00% | 49.98% | 100.0% | - | 0 | 20 | per-user temporal split derived from timestamps |
| KuaiRec v2 | 10,300,969 | 0.00% | 66.84% | 0.0% | - | 0 | 44 | per-user loader-order split (no timestamps) |
| Taobao | 75,969,639 | 0.00% | 12.64% | 100.0% | - | 0 | 1 | per-user temporal split derived from timestamps |
| KuaiRand-1K | 11,421,447 | 0.00% | 38.99% | 100.0% | 0.4% | 0 | 68 | per-user temporal split derived from timestamps |

## Amazon-Book

- Interaction semantics: implicit observed interactions only
- Plotted context: implicit interaction graph and split structure
- Distinct user-item pairs: 2,984,108
- Repeated-pair share: 0.00%
- Split counts: train=2,093,875, val=234,212, test=603,378
- Response summary (label): mean=1.0000, std=0.0000, min=1.0000, max=1.0000
- Label distribution: label > 0=2,984,108, label <= 0=0.
- Sign distribution: sign > 0=0, sign = 0=2,984,108, sign < 0=0.

## MovieLens 1M

- Interaction semantics: explicit 1-to-5 star ratings
- Plotted context: ratings over time plus user and item metadata
- Distinct user-item pairs: 1,000,209
- Repeated-pair share: 0.00%
- Split counts: train=797,758, val=99,692, test=102,759
- Response summary (raw_target): mean=3.5816, std=1.1171, min=1.0000, max=5.0000
- Label distribution: label > 0=575,281, label <= 0=424,928.
- Sign distribution: sign > 0=575,281, sign = 0=261,197, sign < 0=163,731.

## MovieLens 20M

- Interaction semantics: explicit 1-to-5 star ratings
- Plotted context: ratings over time plus rich item metadata
- Distinct user-item pairs: 20,000,263
- Repeated-pair share: 0.00%
- Split counts: train=15,945,812, val=1,991,736, test=2,062,715
- Response summary (raw_target): mean=3.5255, std=1.0520, min=0.5000, max=5.0000
- Label distribution: label > 0=9,995,410, label <= 0=10,004,853.
- Sign distribution: sign > 0=12,195,566, sign = 0=4,291,193, sign < 0=3,513,504.

## KuaiRec v2

- Interaction semantics: watch-ratio feedback from short-video viewing
- Plotted context: watch ratio plus item-side content descriptors
- Distinct user-item pairs: 10,300,969
- Repeated-pair share: 0.00%
- Split counts: train=8,237,940, val=1,029,713, test=1,033,316
- Response summary (raw_target): mean=0.9363, std=0.8354, min=0.0000, max=5.0000
- Label distribution: label > 0=6,885,007, label <= 0=3,415,962.
- Sign distribution: sign > 0=3,731,149, sign = 0=677, sign < 0=6,569,143.
- Overlap rows (`label > 0` and `sign < 0`): 3,153,181.

## Taobao

- Interaction semantics: multi-behavior shopping interactions
- Plotted context: interaction types from page view to purchase
- Distinct user-item pairs: 75,969,639
- Repeated-pair share: 0.00%
- Split counts: train=60,380,165, val=7,548,302, test=8,041,172
- Response summary (raw_target): mean=0.2434, std=0.6848, min=0.0000, max=3.0000
- Label distribution: label > 0=9,601,560, label <= 0=66,368,079.
- Sign distribution: sign > 0=9,601,560, sign = 0=0, sign < 0=66,368,079.
- Top behavior mix: pv=66,368,079, cart=5,039,405, fav=2,636,128, buy=1,926,027

## KuaiRand-1K

- Interaction semantics: short-video engagement logs with randomized exposure
- Plotted context: watch time, engagement labels, and exposure policy
- Distinct user-item pairs: 11,421,447
- Repeated-pair share: 0.00%
- Split counts: train=9,136,755, val=1,142,098, test=1,142,594
- Response summary (raw_target): mean=0.3844, std=0.6392, min=0.0000, max=5.0000
- Label distribution: label > 0=4,453,651, label <= 0=6,967,796.
- Sign distribution: sign > 0=3,142,603, sign = 0=8,269,070, sign < 0=9,774.
- Overlap rows (`label > 0` and `sign < 0`): 1,651.
- Exposure policy: randomized_share=0.38%, positive_rate_randomized=17.71%, positive_rate_standard=39.07%
- Top behavior mix: neutral=6,958,852, long_view=2,935,267, click=1,308,484, like=181,371, comment=23,009, other=14,464
- Source domains: random: n=43,026, pos=17.71%, standard: n=11,378,421, pos=39.07%
