# Benchmark Dataset Summary

Generated from `src/data_exploration/explore_all_datasets.py` using the same
canonical statistics that power the benchmark figures.

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

## MovieLens 1M

- Interaction semantics: explicit 1-to-5 star ratings
- Plotted context: ratings over time plus user and item metadata
- Distinct user-item pairs: 1,000,209
- Repeated-pair share: 0.00%
- Split counts: train=797,758, val=99,692, test=102,759
- Response summary (raw_target): mean=3.5816, std=1.1171, min=1.0000, max=5.0000

## MovieLens 20M

- Interaction semantics: explicit 1-to-5 star ratings
- Plotted context: ratings over time plus rich item metadata
- Distinct user-item pairs: 20,000,263
- Repeated-pair share: 0.00%
- Split counts: train=15,945,812, val=1,991,736, test=2,062,715
- Response summary (raw_target): mean=3.5255, std=1.0520, min=0.5000, max=5.0000

## KuaiRec v2

- Interaction semantics: watch-ratio feedback from short-video viewing
- Plotted context: watch ratio plus item-side content descriptors
- Distinct user-item pairs: 10,300,969
- Repeated-pair share: 0.00%
- Split counts: train=8,237,940, val=1,029,713, test=1,033,316
- Response summary (raw_target): mean=0.9363, std=0.8354, min=0.0000, max=5.0000

## Taobao

- Interaction semantics: multi-behavior shopping interactions
- Plotted context: interaction types from page view to purchase
- Distinct user-item pairs: 75,969,639
- Repeated-pair share: 0.00%
- Split counts: train=60,380,165, val=7,548,302, test=8,041,172
- Response summary (raw_target): mean=0.2434, std=0.6848, min=0.0000, max=3.0000
- Top behavior mix: pv=66,368,079, cart=5,039,405, fav=2,636,128, buy=1,926,027

## KuaiRand-1K

- Interaction semantics: short-video engagement logs with randomized exposure
- Plotted context: watch time, engagement labels, and exposure policy
- Distinct user-item pairs: 11,421,447
- Repeated-pair share: 0.00%
- Split counts: train=9,136,755, val=1,142,098, test=1,142,594
- Response summary (raw_target): mean=0.3844, std=0.6392, min=0.0000, max=5.0000
- Exposure policy: randomized_share=0.38%, positive_rate_randomized=17.71%, positive_rate_standard=39.07%
- Top behavior mix: neutral=6,958,852, long_view=2,935,267, click=1,308,484, like=181,371, comment=23,009, other=14,464
- Source domains: random: n=43,026, pos=17.71%, standard: n=11,378,421, pos=39.07%
