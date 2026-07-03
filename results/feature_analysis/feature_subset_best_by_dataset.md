# Feature Subset Best By Dataset

Ranking: ValidationCRRU@20_40 within each dataset.
Positive side_feature_gain means side features helped.
Positive single_group_gain means that group alone beat no features.
Positive drop_group_effect means removing that group hurt.
Positive pair/triple gain means that combination beat no features.

## CRRU Reporting Utility

**CRRU@K - Composite Resource-aware Recommendation Utility at K**
- Formulation: absolute_weighted_geometric_crru_with_log_normalized_raw_arp.
- Family: CRRU@K(m; theta) is parameterized by explicitly stated weights.
- Direction: higher is better; bounded in [0, 1] for valid inputs.
- Scope: absolute per-run utility; independent of other experiments.
- Adding or removing experiments cannot change an already computed CRRU value.
- No row-set, report-row, dataset, trial, or completed-experiment min-max normalization is used.
- RankingAccuracy@K keeps ranking quality dominant.
- RankingAccuracy@K = NDCG@K^0.50 * Recall@K^0.35 * HitRatio@K^0.15
- HitRatio has the smallest ranking-accuracy weight because it is coarser than NDCG and Recall.
- PopularityAwarePersonalization@K is personalized recommendation with reduced popularity concentration.
- Popular items are not inherently bad; CRRU only reflects a thesis preference against excessive concentration.
- PyG AveragePopularity@K is logged from raw train-only item interaction counts.
- Reconstructing LargestTrainingItemInteractionCount supplies only the CRRU denominator; it does not convert a legacy non-raw AveragePopularity@K value into raw PyG ARP.
- CRRU normalizes raw ARP only inside the utility: CRRUNormalizedAveragePopularity@K = log(1 + AveragePopularity@K) / log(1 + LargestTrainingItemInteractionCount).
- PopularityAwarePersonalization@K = Personalization@K^0.40 * InverseRecommendationPopularity@K^0.60
- InverseRecommendationPopularity@K = 1 - CRRUNormalizedAveragePopularity@K
- Peak GPU memory is treated as a capacity cost; epoch duration is a throughput cost.
- Average GPU memory may be reported as a diagnostic but is not used in CRRU.
- TrainingResourceUtility = PeakGpuMemoryCapacityScore^0.50 * EpochDurationEfficiencyScore^0.50
- PeakGpuMemoryCapacityScore = 1 / (1 + log(1 + PeakGpuMemoryMegabytes)).
- EpochDurationEfficiencyScore = 1 / (1 + log(1 + EpochDurationSeconds)).
- CRRU@K = RankingAccuracy@K^0.55 * PopularityAwarePersonalization@K^0.30 * TrainingResourceUtility^0.15
- ValidationCRRU@20And40 = arithmetic_mean(CRRU@20, CRRU@40).
- CRRU_EPSILON=1e-08 is only a numerical lower bound, not normalization.
- Missing, NaN, infinite, or out-of-domain inputs raise an error.
- CRRU is a thesis comparison utility, not a causal estimator or standard recommender metric.
- CRRU is not a fairness metric, debiasing proof, or universal cross-dataset quality score.

## amazonbook

Best completed profile: `graph_only` (ValidationCRRU@20_40=0.144116, ValidationAccuracy@20_40=0.034895, NDCG@20=0.024389, Recall@20=0.044113, AvgPop@20=0.082221, time/epoch=1.92, VRAM=14940.0).
Pending required profiles: 0.

## kuairand1k

Best completed profile: `triple_item_author_music__item_upload_time__item_category` (ValidationCRRU@20_40=0.062106, ValidationAccuracy@20_40=0.007737, NDCG@20=0.004747, Recall@20=0.010046, AvgPop@20=0.309704, time/epoch=0.12, VRAM=4469.0).
Pending required profiles: 0.

| Effect | Delta ValidationCRRU@20_40 |
|---|---:|
| side_feature_gain | 0.020024 |
| drop_group_effect:item_author_music | -0.001287 |
| drop_group_effect:item_category | 0.011115 |
| drop_group_effect:item_resolution | -0.004227 |
| drop_group_effect:item_upload_time | 0.003025 |
| drop_group_effect:item_video_metadata | -0.002977 |
| pair_gain:item_author_music__item_category | 0.021188 |
| pair_gain:item_author_music__item_resolution | 0.009823 |
| pair_gain:item_author_music__item_upload_time | 0.017813 |
| pair_gain:item_author_music__item_video_metadata | 0.011007 |
| pair_gain:item_resolution__item_category | 0.013256 |
| pair_gain:item_upload_time__item_category | 0.019074 |
| pair_gain:item_upload_time__item_resolution | 0.003017 |
| pair_gain:item_video_metadata__item_category | 0.014720 |
| pair_gain:item_video_metadata__item_resolution | 0.000120 |
| pair_gain:item_video_metadata__item_upload_time | 0.012293 |
| single_group_gain:item_author_music | 0.011687 |
| single_group_gain:item_category | 0.018246 |
| single_group_gain:item_resolution | -0.000910 |
| single_group_gain:item_upload_time | 0.007890 |
| single_group_gain:item_video_metadata | 0.011565 |
| triple_gain:item_author_music__item_resolution__item_category | 0.021027 |
| triple_gain:item_author_music__item_upload_time__item_category | 0.025027 |
| triple_gain:item_author_music__item_upload_time__item_resolution | 0.011395 |
| triple_gain:item_author_music__item_video_metadata__item_category | 0.018616 |
| triple_gain:item_author_music__item_video_metadata__item_resolution | 0.006219 |
| triple_gain:item_author_music__item_video_metadata__item_upload_time | 0.013348 |
| triple_gain:item_upload_time__item_resolution__item_category | 0.017949 |
| triple_gain:item_video_metadata__item_resolution__item_category | 0.015218 |
| triple_gain:item_video_metadata__item_upload_time__item_category | 0.023391 |
| triple_gain:item_video_metadata__item_upload_time__item_resolution | 0.003632 |

## kuairec_v2

Best completed profile: `single_item_resolution` (ValidationCRRU@20_40=0.215856, ValidationAccuracy@20_40=0.079564, NDCG@20=0.103866, Recall@20=0.031134, AvgPop@20=0.502449, time/epoch=3.58, VRAM=9234.0).
Pending required profiles: 0.

| Effect | Delta ValidationCRRU@20_40 |
|---|---:|
| side_feature_gain | 0.002253 |
| drop_group_effect:item_author_music | -0.012681 |
| drop_group_effect:item_category | 0.032549 |
| drop_group_effect:item_resolution | 0.024464 |
| drop_group_effect:item_upload_time | 0.018950 |
| drop_group_effect:item_video_metadata | 0.013860 |
| pair_gain:item_author_music__item_category | 0.002055 |
| pair_gain:item_author_music__item_resolution | -0.012036 |
| pair_gain:item_author_music__item_upload_time | -0.014476 |
| pair_gain:item_author_music__item_video_metadata | -0.027379 |
| pair_gain:item_category__item_upload_time | 0.000945 |
| pair_gain:item_resolution__item_category | 0.018766 |
| pair_gain:item_resolution__item_upload_time | 0.029885 |
| pair_gain:item_video_metadata__item_category | 0.019255 |
| pair_gain:item_video_metadata__item_resolution | 0.049288 |
| pair_gain:item_video_metadata__item_upload_time | -0.024602 |
| single_group_gain:item_author_music | -0.014100 |
| single_group_gain:item_category | 0.001977 |
| single_group_gain:item_resolution | 0.051471 |
| single_group_gain:item_upload_time | 0.021339 |
| single_group_gain:item_video_metadata | 0.048024 |
| triple_gain:item_author_music__item_category__item_upload_time | -0.013013 |
| triple_gain:item_author_music__item_resolution__item_category | -0.000512 |
| triple_gain:item_author_music__item_resolution__item_upload_time | -0.013823 |
| triple_gain:item_author_music__item_video_metadata__item_category | -0.017598 |
| triple_gain:item_author_music__item_video_metadata__item_resolution | -0.026078 |
| triple_gain:item_author_music__item_video_metadata__item_upload_time | -0.030886 |
| triple_gain:item_resolution__item_category__item_upload_time | 0.000169 |
| triple_gain:item_video_metadata__item_category__item_upload_time | -0.017784 |
| triple_gain:item_video_metadata__item_resolution__item_category | -0.013587 |
| triple_gain:item_video_metadata__item_resolution__item_upload_time | -0.024188 |

## movielens1m

Best completed profile: `none` (ValidationCRRU@20_40=0.262050, ValidationAccuracy@20_40=0.096454, NDCG@20=0.075484, Recall@20=0.107588, AvgPop@20=0.179336, time/epoch=1.12, VRAM=7016.0).
Pending required profiles: 0.

| Effect | Delta ValidationCRRU@20_40 |
|---|---:|
| side_feature_gain | -0.012534 |
| drop_group_effect:item_genre | -0.008488 |
| single_group_gain:item_genre | -0.013897 |
