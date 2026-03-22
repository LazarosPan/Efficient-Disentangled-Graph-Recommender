# Dataset Information Report

- Generated at: 2026-03-22T22:31:05
- Scan root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data`
- Datasets scanned: 17
- Files scanned: 114
- Total size: 19,483,255,075 B (18,580.68 MiB | 18.15 GiB)

## Dataset Summary

| Dataset | Status | Files | Total Size | Pairwise | Sign | Preprocessing | Raw | Processed |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| KuaiRand-1K | PARTIAL | 12 | 4,404.30 MiB | yes | yes | medium | no | no |
| KuaiSAR_v2 | PARTIAL | 6 | 3,546.29 MiB | yes | no | medium | no | no |
| Taobao | PARTIAL (raw only) | 3 | 3,502.23 MiB | yes | yes | medium | yes | no |
| AmazonProducts | PARTIAL (raw only) | 4 | 2,414.98 MiB | no | no | high | yes | no |
| Yelp | PARTIAL (raw only) | 4 | 2,036.25 MiB | no | no | high | yes | no |
| KuaiRec_v2 | PARTIAL | 14 | 1,507.51 MiB | yes | yes | medium | no | no |
| MovieLens20M | PARTIAL (raw only) | 7 | 835.03 MiB | yes | yes | low | yes | no |
| netflix | PARTIAL (raw only) | 20 | 142.56 MiB | no | yes | medium | yes | no |
| Douban_Book | PARTIAL | 14 | 122.08 MiB | no | no | high | no | no |
| MovieLens1M | PARTIAL (raw only) | 4 | 23.75 MiB | yes | yes | low | yes | no |
| AmazonBook | PARTIAL (raw only) | 4 | 19.65 MiB | yes | no | medium | yes | no |
| AmazonCDs | PARTIAL (raw only) | 4 | 13.03 MiB | no | no | low | yes | no |
| KuaiRand_SIGformer | PARTIAL (raw only) | 4 | 3.52 MiB | no | yes | medium | yes | no |
| MovieLens | PARTIAL (raw only) | 5 | 3.15 MiB | yes | yes | low | yes | no |
| KuaiRec_SIGformer | PARTIAL (raw only) | 4 | 3.08 MiB | no | yes | medium | yes | no |
| Douban | PARTIAL (raw only) | 1 | 2.65 MiB | no | no | high | yes | no |
| AmazonMusic | PARTIAL (raw only) | 4 | 0.63 MiB | no | no | low | yes | no |

## Causal Audit Summary

| Dataset | Experiment Path | Strongest Asset | Primary Risk | Priority | Next Step |
| --- | --- | --- | --- | --- | --- |
| KuaiRand-1K | kuairand1k | randomized exposure metadata | behavioral aggregate features may leak post-exposure outcomes | highest | audit randomized vs non-random exposure slices before adding new causal objectives |
| KuaiSAR_v2 | not in current loader registry | item-side covariates available | search and recommendation signals are mixed and need causal separation | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| Taobao | taobao | graded sign or explicit negative signal | no major risk flagged | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| AmazonProducts | not in current loader registry | interaction-only baseline | preprocessing cost is high before the dataset can join the formal matrix | low | implement a canonical loader before considering formal experiments |
| Yelp | not in current loader registry | interaction-only baseline | preprocessing cost is high before the dataset can join the formal matrix | low | implement a canonical loader before considering formal experiments |
| KuaiRec_v2 | kuairec_v2 | graded sign or explicit negative signal | behavioral aggregate features may leak post-exposure outcomes | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| MovieLens20M | movielens20m | graded sign or explicit negative signal | textual descriptors need encoding and leakage checks before promotion | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| netflix | not in current loader registry | graded sign or explicit negative signal | no major risk flagged | low | implement a canonical loader before considering formal experiments |
| Douban_Book | not in current loader registry | user-side covariates available | preprocessing cost is high before the dataset can join the formal matrix | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| MovieLens1M | movielens1m | graded sign or explicit negative signal | textual descriptors need encoding and leakage checks before promotion | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| AmazonBook | amazonbook | temporal ordering available | no major risk flagged | medium | use as an interaction-only baseline unless new covariates are engineered |
| AmazonCDs | not in current loader registry | temporal ordering available | no major risk flagged | low | implement a canonical loader before considering formal experiments |
| KuaiRand_SIGformer | not in current loader registry | graded sign or explicit negative signal | no major risk flagged | low | implement a canonical loader before considering formal experiments |
| MovieLens | not in current loader registry | graded sign or explicit negative signal | textual descriptors need encoding and leakage checks before promotion | high | separate pre-treatment descriptors from post-treatment aggregates and promote only safe features |
| KuaiRec_SIGformer | not in current loader registry | graded sign or explicit negative signal | no major risk flagged | low | implement a canonical loader before considering formal experiments |
| Douban | not in current loader registry | temporal ordering available | preprocessing cost is high before the dataset can join the formal matrix | low | implement a canonical loader before considering formal experiments |
| AmazonMusic | not in current loader registry | temporal ordering available | no major risk flagged | low | implement a canonical loader before considering formal experiments |

## KuaiRand-1K

- Status: PARTIAL
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/KuaiRand-1K`
- Total files: 12
- Dataset size: 4,618,247,460 B (4,404.30 MiB | 4.30 GiB)
- Raw files present: no
- Processed files present: no
- Extension breakdown: .csv: 6, .md: 1, .png: 3, .py: 1, <no_ext>: 1

### U-CaGNN Suitability

- Registry dataset: KuaiRand-1K
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: yes
- Preprocessing cost: medium
- Inferred fields: user=user_id, item=video_id, label=is_click, timestamp=time_ms
- File-level semantic inspection errors: 0
- Assessment: Most valuable for randomized exposure, but heavier than KuaiRec because of sequential logs and large feature tables.

### Causal Feature Audit

- Current experiment path: kuairand1k
- Strongest causal asset: randomized exposure metadata
- Highest-priority audit level: highest
- Main opportunities: temporal ordering available, graded sign or explicit negative signal, randomized exposure metadata, user-side covariates available, item-side covariates available
- Main risks: behavioral aggregate features may leak post-exposure outcomes, textual descriptors need encoding and leakage checks before promotion
- Recommended next step: audit randomized vs non-random exposure slices before adding new causal objectives

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | data/log_random_4_22_to_5_08_1k.csv, data/log_standard_4_08_to_4_21_1k.csv, data/log_standard_4_22_to_5_08_1k.csv | native loader | consumed by training and evaluation | contains randomized exposure indicator; supports label/sign construction; supports temporal splitting |
| User Features | data/user_features_1k.csv | available in files only | retained in canonical/graph objects, not used by model | candidate pre-treatment user covariates |
| Item Features | data/video_features_basic_1k.csv, data/video_features_statistic_1k.csv | thesis-default canonical item_features from safe video_features_basic_1k descriptor columns only | used in Module A when canonical item_features exist | mix of item descriptors and post-treatment exposure aggregates |
| Metadata / Context | data/log_random_4_22_to_5_08_1k.csv, data/log_standard_4_08_to_4_21_1k.csv, data/log_standard_4_22_to_5_08_1k.csv | preserved in canonical metadata: is_rand | retained only for analysis; not used by model | supports exposure-aware causal evaluation |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | date | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | hourmin | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | time_ms | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_click | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_like | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_follow | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_comment | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_forward | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_hate | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | long_view | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | play_time_ms | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | duration_ms | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | profile_stay_time | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | comment_stay_time | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_profile_enter | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | is_rand | pre_treatment | analysis_retained | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/log_random_4_22_to_5_08_1k.csv | metadata | context | tab | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | date | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | hourmin | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | time_ms | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_click | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_like | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_follow | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_comment | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_forward | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_hate | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | long_view | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | play_time_ms | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | duration_ms | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | profile_stay_time | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | comment_stay_time | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_profile_enter | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | is_rand | pre_treatment | analysis_retained | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/log_standard_4_08_to_4_21_1k.csv | metadata | context | tab | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | date | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | hourmin | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | time_ms | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_click | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_like | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_follow | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_comment | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_forward | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_hate | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | long_view | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | play_time_ms | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | duration_ms | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | profile_stay_time | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | comment_stay_time | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_profile_enter | unknown | raw_only | review | needs manual causal review before promotion |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | is_rand | pre_treatment | analysis_retained | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/log_standard_4_22_to_5_08_1k.csv | metadata | context | tab | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | user_active_degree | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features_1k.csv | user_features | user | is_lowactive_period | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features_1k.csv | user_features | user | is_live_streamer | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features_1k.csv | user_features | user | is_video_author | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features_1k.csv | user_features | user | follow_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/user_features_1k.csv | user_features | user | follow_user_num_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | fans_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/user_features_1k.csv | user_features | user | fans_user_num_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | friend_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/user_features_1k.csv | user_features | user | friend_user_num_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | register_days | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features_1k.csv | user_features | user | register_days_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat0 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat1 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat2 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat3 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat4 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat5 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat6 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat7 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat8 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat9 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat10 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat11 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat12 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat13 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat14 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat15 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat16 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features_1k.csv | user_features | user | onehot_feat17 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/video_features_basic_1k.csv | item_features | item | author_id | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | video_type | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | upload_dt | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | upload_type | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | visible_status | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | video_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_basic_1k.csv | item_features | item | server_width | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | server_height | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | music_id | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | music_type | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/video_features_basic_1k.csv | item_features | item | tag | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/video_features_statistic_1k.csv | item_features | item | counts | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_features_statistic_1k.csv | item_features | item | show_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | show_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | play_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | complete_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | complete_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | valid_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | valid_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | long_time_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | long_time_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | short_time_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | short_time_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | play_progress | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | comment_stay_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | like_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | click_like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | double_click_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | cancel_like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | cancel_like_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | comment_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | direct_comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | reply_comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | delete_comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | delete_comment_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | comment_like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | comment_like_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | follow_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | follow_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | cancel_follow_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | cancel_follow_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | share_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | share_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | download_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | download_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | report_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | report_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | reduce_similar_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | reduce_similar_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | collect_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | collect_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | cancel_collect_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | cancel_collect_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | direct_comment_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | reply_comment_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | share_all_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | share_all_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/video_features_statistic_1k.csv | item_features | item | outsite_share_all_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| data/video_features_statistic_1k.csv | .csv | 3,214.48 MiB | 4371869 | 4371869 | 52 | video_id, counts, show_cnt, show_user_num, play_cnt, play_user_num, play_duration, complete_play_cnt, complete_play_user_num, valid_play_cnt, valid_play_user_num, long_time_play_cnt, long_time_play_user_num, short_time_play_cnt, short_time_play_user_num, play_progress, comment_stay_duration, like_cnt, like_user_num, click_like_cnt, double_click_cnt, cancel_like_cnt, cancel_like_user_num, comment_cnt, comment_user_num, direct_comment_cnt, reply_comment_cnt, delete_comment_cnt, delete_comment_user_num, comment_like_cnt, comment_like_user_num, follow_cnt, follow_user_num, cancel_follow_cnt, cancel_follow_user_num, share_cnt, share_user_num, download_cnt, download_user_num, report_cnt, report_user_num, reduce_similar_cnt, reduce_similar_user_num, collect_cnt, collect_user_num, cancel_collect_cnt, cancel_collect_user_num, direct_comment_user_num, reply_comment_user_num, share_all_cnt, share_all_user_num, outsite_share_all_cnt | columns_detected=0; delimiter_guess=,; has_header=False |
| data/log_standard_4_22_to_5_08_1k.csv | .csv | 469.50 MiB | 6657062 | 6657061 | 19 | user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab | columns_detected=19; delimiter_guess=,; has_header=True |
| data/video_features_basic_1k.csv | .csv | 359.43 MiB | 4371869 | 4371868 | 12 | video_id, author_id, video_type, upload_dt, upload_type, visible_status, video_duration, server_width, server_height, music_id, music_type, tag | columns_detected=12; delimiter_guess=,; has_header=True |
| data/log_standard_4_08_to_4_21_1k.csv | .csv | 356.36 MiB | 5055985 | 5055984 | 19 | user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab | columns_detected=19; delimiter_guess=,; has_header=True |
| data/log_random_4_22_to_5_08_1k.csv | .csv | 2.94 MiB | 43029 | 43028 | 19 | user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab | columns_detected=19; delimiter_guess=,; has_header=True |
| figs/kuaishou-app.png | .png | 1.03 MiB | - | - | - | - | - |
| figs/three-version.png | .png | 0.31 MiB | - | - | - | - | - |
| data/user_features_1k.csv | .csv | 0.12 MiB | 1001 | 1000 | 31 | user_id, user_active_degree, is_lowactive_period, is_live_streamer, is_video_author, follow_user_num, follow_user_num_range, fans_user_num, fans_user_num_range, friend_user_num, friend_user_num_range, register_days, register_days_range, onehot_feat0, onehot_feat1, onehot_feat2, onehot_feat3, onehot_feat4, onehot_feat5, onehot_feat6, onehot_feat7, onehot_feat8, onehot_feat9, onehot_feat10, onehot_feat11, onehot_feat12, onehot_feat13, onehot_feat14, onehot_feat15, onehot_feat16, onehot_feat17 | columns_detected=31; delimiter_guess=,; has_header=True |
| figs/KuaiRand.png | .png | 0.09 MiB | - | - | - | - | - |
| README.md | .md | 0.02 MiB | 308 | 308 | - | - | - |
| LICENSE | <no_ext> | 0.02 MiB | 426 | 426 | - | - | - |
| load_data_1k.py | .py | 0.00 MiB | - | - | - | - | - |

## KuaiSAR_v2

- Status: PARTIAL
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/KuaiSAR_v2`
- Total files: 6
- Dataset size: 3,718,550,329 B (3,546.29 MiB | 3.46 GiB)
- Raw files present: no
- Processed files present: no
- Extension breakdown: .csv: 5, .md: 1

### U-CaGNN Suitability

- Registry dataset: KuaiSAR_v2
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: yes
- Preprocessing cost: medium
- Inferred fields: user=user_id, item=item_id, label=-, timestamp=timestamp
- File-level semantic inspection errors: 0
- Assessment: Rich dataset, but search and recommendation are mixed, which makes it less aligned with the first U-CaGNN benchmark phase.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: item-side covariates available
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, user-side covariates available, item-side covariates available
- Main risks: search and recommendation signals are mixed and need causal separation, social links may act as confounders and need explicit treatment, textual descriptors need encoding and leakage checks before promotion
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | rec_inter.csv, src_inter.csv | available in files only | not reachable by current experiment path | supports temporal splitting |
| User Features | user_features.csv | available in files only | not reachable by current experiment path | candidate pre-treatment user covariates |
| Item Features | item_features.csv | available in files only | not reachable by current experiment path | descriptor-rich item covariates that need encoding |
| Metadata / Context | rec_inter.csv, social_network.csv, src_inter.csv (+1 more) | available in files only | not reachable by current experiment path | context mixes search and recommendation behavior |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| item_features.csv | item_features | item | first_level_category_id | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| item_features.csv | item_features | item | first_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | second_level_category_id | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| item_features.csv | item_features | item | second_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | third_level_category_id | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| item_features.csv | item_features | item | third_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | fourth_level_category_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| item_features.csv | item_features | item | fourth_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | caption | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| item_features.csv | item_features | item | author_id | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| item_features.csv | item_features | item | item_type | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | upload_time | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| item_features.csv | item_features | item | upload_type | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| item_features.csv | item_features | item | first_level_category_name_en | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | second_level_category_name_en | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | third_level_category_name_en | unknown | raw_only | review | needs manual causal review before promotion |
| item_features.csv | item_features | item | fourth_level_category_name_en | unknown | raw_only | review | needs manual causal review before promotion |
| rec_inter.csv | metadata | context | duration_ms | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| rec_inter.csv | metadata | context | playing_time | unknown | raw_only | review | needs manual causal review before promotion |
| rec_inter.csv | metadata | context | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| rec_inter.csv | metadata | context | forward | unknown | raw_only | review | needs manual causal review before promotion |
| rec_inter.csv | metadata | context | like | unknown | raw_only | review | needs manual causal review before promotion |
| rec_inter.csv | metadata | context | follow | unknown | raw_only | review | needs manual causal review before promotion |
| rec_inter.csv | metadata | context | search_photo_related | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| rec_inter.csv | metadata | context | search | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| rec_inter.csv | metadata | context | click | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| rec_inter.csv | metadata | context | time | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| social_network.csv | metadata | context | user_follow_id | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| src_inter.csv | metadata | context | search_session_id | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| src_inter.csv | metadata | context | search_session_timestamp | unknown | raw_only | review | needs manual causal review before promotion |
| src_inter.csv | metadata | context | search_session_source | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| src_inter.csv | metadata | context | keyword | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| src_inter.csv | metadata | context | click_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| src_inter.csv | metadata | context | item_type | unknown | raw_only | review | needs manual causal review before promotion |
| src_inter.csv | metadata | context | time | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| user_features.csv | user_features | user | onehot_feat1 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| user_features.csv | user_features | user | onehot_feat2 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| user_features.csv | user_features | user | search_active_level | unknown | raw_only | review | needs manual causal review before promotion |
| user_features.csv | user_features | user | rec_active_level | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| item_features.csv | .csv | 1,915.49 MiB | 6949991 | 6949990 | 18 | item_id, first_level_category_id, first_level_category_name, second_level_category_id, second_level_category_name, third_level_category_id, third_level_category_name, fourth_level_category_id, fourth_level_category_name, caption, author_id, item_type, upload_time, upload_type, first_level_category_name_en, second_level_category_name_en, third_level_category_name_en, fourth_level_category_name_en | columns_detected=18; delimiter_guess=,; has_header=True |
| rec_inter.csv | .csv | 1,102.64 MiB | 14605717 | 14605716 | 12 | user_id, item_id, duration_ms, playing_time, timestamp, forward, like, follow, search_photo_related, search, click, time | columns_detected=12; delimiter_guess=,; has_header=True |
| src_inter.csv | .csv | 527.75 MiB | 5059170 | 5059169 | 9 | user_id, search_session_id, search_session_timestamp, search_session_source, keyword, item_id, click_cnt, item_type, time | columns_detected=9; delimiter_guess=,; has_header=True |
| user_features.csv | .csv | 0.39 MiB | 25878 | 25877 | 5 | user_id, onehot_feat1, onehot_feat2, search_active_level, rec_active_level | columns_detected=5; delimiter_guess=,; has_header=True |
| social_network.csv | .csv | 0.01 MiB | 613 | 612 | 2 | user_id, user_follow_id | columns_detected=2; delimiter_guess=,; has_header=True |
| README.md | .md | 0.01 MiB | 118 | 118 | - | - | - |

## Taobao

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/Taobao`
- Total files: 3
- Dataset size: 3,672,350,226 B (3,502.23 MiB | 3.42 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .csv: 1, .md: 1, .md5: 1

### U-CaGNN Suitability

- Registry dataset: Taobao
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: medium
- Inferred fields: user=user_id, item=item_id, label=-, timestamp=timestamp
- File-level semantic inspection errors: 0
- Assessment: Large-scale multi-behavior dataset; good for implicit sign derivation and scaling experiments.

### Causal Feature Audit

- Current experiment path: taobao
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, graded sign or explicit negative signal, item-side covariates available
- Main risks: no major risk flagged
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/UserBehavior.csv | native loader | consumed by training and evaluation | supports label/sign construction; supports temporal splitting |
| Item Features | raw/UserBehavior.csv | thesis-default canonical item_features derived from category_id | used in Module A when canonical item_features exist | candidate item-side descriptors for feature fusion |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/UserBehavior.csv | item_features | item | category_id | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| raw/UserBehavior.csv | item_features | item | behavior_type | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/UserBehavior.csv | item_features | item | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/UserBehavior.csv | .csv | 3,502.22 MiB | 100150807 | 100150807 | 5 | user_id, item_id, category_id, behavior_type, timestamp | columns_detected=5; delimiter_guess=,; has_header=False |
| raw/README.md | .md | 0.00 MiB | 51 | 51 | - | - | - |
| raw/UserBehavior.csv.zip.md5 | .md5 | 0.00 MiB | - | - | - | - | - |

## AmazonProducts

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/AmazonProducts`
- Total files: 4
- Dataset size: 2,532,295,056 B (2,414.98 MiB | 2.36 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .json: 2, .npy: 1, .npz: 1

### U-CaGNN Suitability

- Registry dataset: AmazonProducts
- Pairwise triplets ready: no
- Timestamp split ready: no
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: yes
- Preprocessing cost: high
- Inferred fields: user=-, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Likely off-scope for phase 1 because the structure is not a simple user-item recommendation table.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: interaction-only baseline
- Highest-priority audit level: low
- Main opportunities: interaction-only baseline
- Main risks: preprocessing cost is high before the dataset can join the formal matrix
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Metadata / Context | raw/class_map.json, raw/role.json | available in files only | not reachable by current experiment path | context or auxiliary metadata |

### Candidate Column Audit

- No candidate columns detected.

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/feats.npy | .npy | 1,197.78 MiB | - | - | - | - | numpy_dtype=float32; numpy_shape=(1569960, 200) |
| raw/adj_full.npz | .npz | 705.77 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['indices', 'indptr', 'format', 'shape', 'data'] |
| raw/class_map.json | .json | 499.02 MiB | 0 | 0 | - | - | json_keys=1569960; json_keys_preview=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']; json_type=dict |
| raw/role.json | .json | 12.42 MiB | 0 | 0 | - | - | json_keys=3; json_keys_preview=['tr', 'va', 'te']; json_type=dict |

## Yelp

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/Yelp`
- Total files: 4
- Dataset size: 2,135,160,769 B (2,036.25 MiB | 1.99 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .json: 2, .npy: 1, .npz: 1

### U-CaGNN Suitability

- Registry dataset: Yelp
- Pairwise triplets ready: no
- Timestamp split ready: no
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: yes
- Preprocessing cost: high
- Inferred fields: user=-, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Likely off-scope for phase 1 because the structure is not a simple user-item recommendation table.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: interaction-only baseline
- Highest-priority audit level: low
- Main opportunities: interaction-only baseline
- Main risks: preprocessing cost is high before the dataset can join the formal matrix
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Metadata / Context | raw/class_map.json, raw/role.json | available in files only | not reachable by current experiment path | context or auxiliary metadata |

### Candidate Column Audit

- No candidate columns detected.

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/feats.npy | .npy | 1,640.73 MiB | - | - | - | - | numpy_dtype=float64; numpy_shape=(716847, 300) |
| raw/class_map.json | .json | 349.92 MiB | 0 | 0 | - | - | json_keys=716847; json_keys_preview=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']; json_type=dict |
| raw/adj_full.npz | .npz | 40.24 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['indices', 'indptr', 'format', 'shape', 'data'] |
| raw/role.json | .json | 5.36 MiB | 0 | 0 | - | - | json_keys=3; json_keys_preview=['tr', 'va', 'te']; json_type=dict |

## KuaiRec_v2

- Status: PARTIAL
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/KuaiRec_v2`
- Total files: 14
- Dataset size: 1,580,742,417 B (1,507.51 MiB | 1.47 GiB)
- Raw files present: no
- Processed files present: no
- Extension breakdown: .csv: 8, .ipynb: 1, .md: 1, .png: 1, .py: 1, .svg: 1, <no_ext>: 1

### U-CaGNN Suitability

- Registry dataset: KuaiRec_v2
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: yes
- Side or multimodal features: yes
- Preprocessing cost: medium
- Inferred fields: user=user_id, item=video_id, label=watch_ratio, timestamp=timestamp
- File-level semantic inspection errors: 0
- Assessment: Strong candidate for richer feedback and side features; likely easiest Kuai dataset to adapt to U-CaGNN.

### Causal Feature Audit

- Current experiment path: kuairec_v2
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, graded sign or explicit negative signal, user-side covariates available, item-side covariates available
- Main risks: behavioral aggregate features may leak post-exposure outcomes, social links may act as confounders and need explicit treatment, textual descriptors need encoding and leakage checks before promotion
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | data/big_matrix.csv, data/small_matrix.csv | native loader | consumed by training and evaluation | supports label/sign construction; supports temporal splitting |
| User Features | data/user_features.csv | available in files only | retained in canonical/graph objects, not used by model | candidate pre-treatment user covariates |
| Item Features | data/big_matrix.csv, data/item_categories.csv, data/item_daily_features.csv (+3 more) | thesis-default canonical item_features from safe item_daily_features columns, item_categories, and caption category IDs | used in Module A when canonical item_features exist | mix of item descriptors and post-treatment exposure aggregates |
| Metadata / Context | data/social_network.csv | available in files only | retained only for analysis; not used by model | social context may act as confounding metadata |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data/big_matrix.csv | item_features | item | play_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/big_matrix.csv | item_features | item | video_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/big_matrix.csv | item_features | item | time | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/big_matrix.csv | item_features | item | date | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/big_matrix.csv | item_features | item | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/big_matrix.csv | item_features | item | watch_ratio | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_categories.csv | item_features | item | feat | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | date | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/item_daily_features.csv | item_features | item | author_id | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | video_type | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | upload_dt | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | upload_type | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | visible_status | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | video_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | video_width | unknown | raw_only | review | needs manual causal review before promotion |
| data/item_daily_features.csv | item_features | item | video_height | unknown | raw_only | review | needs manual causal review before promotion |
| data/item_daily_features.csv | item_features | item | music_id | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| data/item_daily_features.csv | item_features | item | video_tag_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| data/item_daily_features.csv | item_features | item | video_tag_name | unknown | raw_only | review | needs manual causal review before promotion |
| data/item_daily_features.csv | item_features | item | show_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | show_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | play_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | complete_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | complete_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | valid_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | valid_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | long_time_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | long_time_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | short_time_play_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | short_time_play_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | play_progress | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | comment_stay_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | like_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | click_like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | double_click_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | cancel_like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | cancel_like_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | comment_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | direct_comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | reply_comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | delete_comment_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | delete_comment_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | comment_like_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | comment_like_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | follow_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | follow_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | cancel_follow_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | cancel_follow_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | share_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | share_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | download_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | download_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | report_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | report_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | reduce_similar_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | reduce_similar_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | collect_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | collect_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | cancel_collect_cnt | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/item_daily_features.csv | item_features | item | cancel_collect_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/kuairec_caption_category.csv | item_features | item | manual_cover_text | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/kuairec_caption_category.csv | item_features | item | caption | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/kuairec_caption_category.csv | item_features | item | topic_tag | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/kuairec_caption_category.csv | item_features | item | first_level_category_id | pre_treatment | model_consumed | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/kuairec_caption_category.csv | item_features | item | first_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| data/kuairec_caption_category.csv | item_features | item | second_level_category_id | pre_treatment | model_consumed | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/kuairec_caption_category.csv | item_features | item | second_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| data/kuairec_caption_category.csv | item_features | item | third_level_category_id | pre_treatment | model_consumed | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| data/kuairec_caption_category.csv | item_features | item | third_level_category_name | unknown | raw_only | review | needs manual causal review before promotion |
| data/small_matrix.csv | item_features | item | play_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/small_matrix.csv | item_features | item | video_duration | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/small_matrix.csv | item_features | item | time | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/small_matrix.csv | item_features | item | date | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/small_matrix.csv | item_features | item | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/small_matrix.csv | item_features | item | watch_ratio | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/social_network.csv | metadata | context | friend_list | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| data/user_features.csv | user_features | user | user_active_degree | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features.csv | user_features | user | is_lowactive_period | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features.csv | user_features | user | is_live_streamer | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features.csv | user_features | user | is_video_author | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features.csv | user_features | user | follow_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/user_features.csv | user_features | user | follow_user_num_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | fans_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/user_features.csv | user_features | user | fans_user_num_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | friend_user_num | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| data/user_features.csv | user_features | user | friend_user_num_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | register_days | unknown | raw_only | review | needs manual causal review before promotion |
| data/user_features.csv | user_features | user | register_days_range | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat0 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat1 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat2 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat3 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat4 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat5 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat6 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat7 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat8 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat9 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat10 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat11 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat12 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat13 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat14 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat15 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat16 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/user_features.csv | user_features | user | onehot_feat17 | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/video_raw_categories_multi.csv | item_features | item | category_name | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/video_raw_categories_multi.csv | item_features | item | category_id | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| data/video_raw_categories_multi.csv | item_features | item | category_level | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | prob | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | source | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | upload_date | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | category_online | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | root_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| data/video_raw_categories_multi.csv | item_features | item | root_name | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | parent_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| data/video_raw_categories_multi.csv | item_features | item | parent_name | unknown | raw_only | review | needs manual causal review before promotion |
| data/video_raw_categories_multi.csv | item_features | item | type | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| data/big_matrix.csv | .csv | 1,033.33 MiB | 12530807 | 12530806 | 8 | user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio | columns_detected=8; delimiter_guess=,; has_header=True |
| data/small_matrix.csv | .csv | 387.34 MiB | 4676571 | 4676570 | 8 | user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio | columns_detected=8; delimiter_guess=,; has_header=True |
| data/item_daily_features.csv | .csv | 81.88 MiB | 343342 | 343341 | 58 | video_id, date, author_id, video_type, upload_dt, upload_type, visible_status, video_duration, video_width, video_height, music_id, video_tag_id, video_tag_name, show_cnt, show_user_num, play_cnt, play_user_num, play_duration, complete_play_cnt, complete_play_user_num, valid_play_cnt, valid_play_user_num, long_time_play_cnt, long_time_play_user_num, short_time_play_cnt, short_time_play_user_num, play_progress, comment_stay_duration, like_cnt, like_user_num, click_like_cnt, double_click_cnt, cancel_like_cnt, cancel_like_user_num, comment_cnt, comment_user_num, direct_comment_cnt, reply_comment_cnt, delete_comment_cnt, delete_comment_user_num, comment_like_cnt, comment_like_user_num, follow_cnt, follow_user_num, cancel_follow_cnt, cancel_follow_user_num, share_cnt, share_user_num, download_cnt, download_user_num, report_cnt, report_user_num, reduce_similar_cnt, reduce_similar_user_num, collect_cnt, collect_user_num, cancel_collect_cnt, cancel_collect_user_num | columns_detected=58; delimiter_guess=,; has_header=True |
| data/kuairec_caption_category.csv | .csv | 1.87 MiB | 10729 | 10728 | 10 | video_id, manual_cover_text, caption, topic_tag, first_level_category_id, first_level_category_name, second_level_category_id, second_level_category_name, third_level_category_id, third_level_category_name | columns_detected=10; delimiter_guess=,; has_header=True |
| data/video_raw_categories_multi.csv | .csv | 1.64 MiB | 26827 | 26826 | 13 | video_id, category_name, category_id, category_level, prob, source, upload_date, category_online, root_id, root_name, parent_id, parent_name, type | columns_detected=13; delimiter_guess=,; has_header=True |
| data/user_features.csv | .csv | 0.71 MiB | 7177 | 7176 | 31 | user_id, user_active_degree, is_lowactive_period, is_live_streamer, is_video_author, follow_user_num, follow_user_num_range, fans_user_num, fans_user_num_range, friend_user_num, friend_user_num_range, register_days, register_days_range, onehot_feat0, onehot_feat1, onehot_feat2, onehot_feat3, onehot_feat4, onehot_feat5, onehot_feat6, onehot_feat7, onehot_feat8, onehot_feat9, onehot_feat10, onehot_feat11, onehot_feat12, onehot_feat13, onehot_feat14, onehot_feat15, onehot_feat16, onehot_feat17 | columns_detected=31; delimiter_guess=,; has_header=True |
| Statistics_KuaiRec.ipynb | .ipynb | 0.30 MiB | - | - | - | - | - |
| figs/KuaiRec.png | .png | 0.29 MiB | - | - | - | - | - |
| data/item_categories.csv | .csv | 0.11 MiB | 10729 | 10728 | 2 | video_id, feat | columns_detected=2; delimiter_guess=,; has_header=True |
| LICENSE | <no_ext> | 0.02 MiB | 426 | 426 | - | - | - |
| data/README.md | .md | 0.02 MiB | 289 | 289 | - | - | - |
| data/social_network.csv | .csv | 0.01 MiB | 473 | 472 | 2 | user_id, friend_list | columns_detected=2; delimiter_guess=,; has_header=True |
| figs/colab-badge.svg | .svg | 0.00 MiB | - | - | - | - | - |
| loaddata.py | .py | 0.00 MiB | - | - | - | - | - |

## MovieLens20M

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/MovieLens20M`
- Total files: 7
- Dataset size: 875,588,784 B (835.03 MiB | 0.82 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .csv: 6, .md: 1

### U-CaGNN Suitability

- Registry dataset: MovieLens20M
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: low
- Inferred fields: user=userId, item=movieId, label=rating, timestamp=timestamp
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: movielens20m
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, graded sign or explicit negative signal, item-side covariates available
- Main risks: textual descriptors need encoding and leakage checks before promotion
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/ratings.csv, raw/tags.csv | native loader | consumed by training and evaluation | supports label/sign construction; supports temporal splitting |
| Item Features | raw/genome-scores.csv, raw/genome-tags.csv, raw/movies.csv (+1 more) | thesis-default canonical item_features from genres and genome relevance scores | used in Module A when canonical item_features exist | descriptor-rich item covariates that need encoding |
| Metadata / Context | raw/genome-tags.csv, raw/links.csv, raw/tags.csv | available in files only | retained only for analysis; not used by model | context or auxiliary metadata |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/genome-scores.csv | item_features | item | tagid | unknown | raw_only | review | needs manual causal review before promotion |
| raw/genome-scores.csv | item_features | item | relevance | unknown | model_consumed | review | needs manual causal review before promotion |
| raw/genome-tags.csv | item_features | item | tagid | unknown | raw_only | review | needs manual causal review before promotion |
| raw/genome-tags.csv | item_features | item | tag | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| raw/links.csv | metadata | context | imdbid | non_causal | raw_only | exclude | identifier or bookkeeping field |
| raw/links.csv | metadata | context | tmdbid | non_causal | raw_only | exclude | identifier or bookkeeping field |
| raw/movies.csv | item_features | item | title | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| raw/movies.csv | item_features | item | genres | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| raw/ratings.csv | interactions | interaction | rating | post_treatment | model_consumed | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/ratings.csv | interactions | context | timestamp | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| raw/tags.csv | item_features | item | tag | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| raw/tags.csv | item_features | item | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/ratings.csv | .csv | 508.73 MiB | 20000264 | 20000263 | 4 | userId, movieId, rating, timestamp | columns_detected=4; delimiter_guess=,; has_header=True |
| raw/genome-scores.csv | .csv | 308.56 MiB | 11709769 | 11709768 | 3 | movieId, tagId, relevance | columns_detected=3; delimiter_guess=,; has_header=True |
| raw/tags.csv | .csv | 15.83 MiB | 465565 | 465564 | 4 | userId, movieId, tag, timestamp | columns_detected=4; delimiter_guess=,; has_header=True |
| raw/movies.csv | .csv | 1.33 MiB | 27279 | 27278 | 3 | movieId, title, genres | columns_detected=3; delimiter_guess=,; has_header=True |
| raw/links.csv | .csv | 0.54 MiB | 27279 | 27278 | 3 | movieId, imdbId, tmdbId | columns_detected=3; delimiter_guess=,; has_header=True |
| raw/genome-tags.csv | .csv | 0.02 MiB | 1129 | 1128 | 2 | tagId, tag | columns_detected=2; delimiter_guess=,; has_header=True |
| raw/README.md | .md | 0.01 MiB | 183 | 183 | - | - | - |

## netflix

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/netflix`
- Total files: 20
- Dataset size: 149,483,220 B (142.56 MiB | 0.14 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .csv: 5, .json: 2, .npy: 4, .npz: 8, <no_ext>: 1

### U-CaGNN Suitability

- Registry dataset: Netflix
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: medium
- Inferred fields: user=-, item=iid, label=rating, timestamp=ts
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: low
- Main opportunities: temporal ordering available, graded sign or explicit negative signal
- Main risks: no major risk flagged
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/output/record.csv, raw/output/test_record.csv, raw/output/train_record.csv (+2 more) | available in files only | not reachable by current experiment path | supports label/sign construction; supports temporal splitting |
| Metadata / Context | raw/output/popularity.npy, raw/output/popularity_all.npy, raw/output/popularity_blend.npy (+1 more) | available in files only | not reachable by current experiment path | context or auxiliary metadata |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/output/record.csv | interactions | interaction | unnamed: 0 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/record.csv | interactions | interaction | rating | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/output/record.csv | interactions | interaction | ts | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/test_record.csv | interactions | interaction | unnamed: 0 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/test_record.csv | interactions | interaction | rating | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/output/test_record.csv | interactions | interaction | ts | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/train_record.csv | interactions | interaction | unnamed: 0 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/train_record.csv | interactions | interaction | rating | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/output/train_record.csv | interactions | interaction | ts | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/train_skew_record.csv | interactions | interaction | unnamed: 0 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/train_skew_record.csv | interactions | interaction | rating | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/output/train_skew_record.csv | interactions | interaction | ts | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/val_record.csv | interactions | interaction | unnamed: 0 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/output/val_record.csv | interactions | interaction | rating | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/output/val_record.csv | interactions | interaction | ts | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/output/record.csv | .csv | 59.67 MiB | 2212691 | 2212690 | 5 | Unnamed: 0, uid, iid, rating, ts | columns_detected=5; delimiter_guess=,; has_header=True |
| raw/output/train_record.csv | .csv | 35.17 MiB | 1327615 | 1327614 | 5 | Unnamed: 0, uid, iid, rating, ts | columns_detected=5; delimiter_guess=,; has_header=True |
| raw/output/test_record.csv | .csv | 11.72 MiB | 442539 | 442538 | 5 | Unnamed: 0, uid, iid, rating, ts | columns_detected=5; delimiter_guess=,; has_header=True |
| raw/output/train_blend_coo_adj_graph.npz | .npz | 6.78 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/val_record.csv | .csv | 5.81 MiB | 221270 | 221269 | 5 | Unnamed: 0, uid, iid, rating, ts | columns_detected=5; delimiter_guess=,; has_header=True |
| raw/output/train_skew_record.csv | .csv | 5.81 MiB | 221270 | 221269 | 5 | Unnamed: 0, uid, iid, rating, ts | columns_detected=5; delimiter_guess=,; has_header=True |
| raw/output/train_coo_adj_graph.npz | .npz | 4.85 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/coo_record.npz | .npz | 3.67 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/train_coo_record.npz | .npz | 2.43 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/train_skew_coo_adj_graph.npz | .npz | 1.92 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/test_coo_record.npz | .npz | 1.92 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/val_coo_record.npz | .npz | 0.96 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/train_skew_coo_record.npz | .npz | 0.96 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['row', 'col', 'format', 'shape', 'data'] |
| raw/output/user_reindex.json | .json | 0.51 MiB | 0 | 0 | - | - | json_keys=32450; json_keys_preview=['102653', '31580', '6769', '20609', '171416', '350473', '64121', '423', '190438', '165551']; json_type=dict |
| raw/output/item_reindex.json | .json | 0.11 MiB | 0 | 0 | - | - | json_keys=8432; json_keys_preview=['1238', '2628', '6736', '787', '1195', '3970', '1958', '907', '953', '320']; json_type=dict |
| raw/output/popularity.npy | .npy | 0.06 MiB | - | - | - | - | numpy_dtype=int64; numpy_shape=(8432,) |
| raw/output/popularity_all.npy | .npy | 0.06 MiB | - | - | - | - | numpy_dtype=int64; numpy_shape=(8432,) |
| raw/output/popularity_blend.npy | .npy | 0.06 MiB | - | - | - | - | numpy_dtype=int64; numpy_shape=(8432,) |
| raw/output/popularity_skew.npy | .npy | 0.06 MiB | - | - | - | - | numpy_dtype=int64; numpy_shape=(8432,) |
| raw/.DS_Store | <no_ext> | 0.01 MiB | 0 | 0 | - | - | - |

## Douban_Book

- Status: PARTIAL
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/Douban_Book`
- Total files: 14
- Dataset size: 128,011,290 B (122.08 MiB | 0.12 GiB)
- Raw files present: no
- Processed files present: no
- Extension breakdown: .npz: 4, .txt: 9, <no_ext>: 1

### U-CaGNN Suitability

- Registry dataset: Douban_Book
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: yes
- Preprocessing cost: high
- Inferred fields: user=-, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 2
- Assessment: Likely off-scope for phase 1 because the structure is not a simple user-item recommendation table.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: user-side covariates available
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, user-side covariates available
- Main risks: preprocessing cost is high before the dataset can join the formal matrix
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | test.txt, train.txt | available in files only | not reachable by current experiment path | primary interaction source |
| User Features | user.txt | available in files only | not reachable by current experiment path | candidate pre-treatment user covariates |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| author.txt | interactions | context | 23789 | unknown | raw_only | review | needs manual causal review before promotion |
| author.txt | interactions | context | 128 | unknown | raw_only | review | needs manual causal review before promotion |
| item_list.txt | interactions | context | org_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| item_list.txt | interactions | context | remap_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| location.txt | interactions | context | 13225 | unknown | raw_only | review | needs manual causal review before promotion |
| location.txt | interactions | context | 128 | unknown | raw_only | review | needs manual causal review before promotion |
| publisher.txt | interactions | context | 14796 | unknown | raw_only | review | needs manual causal review before promotion |
| publisher.txt | interactions | context | 128 | unknown | raw_only | review | needs manual causal review before promotion |
| user.txt | user_features | user | 25679 | unknown | raw_only | review | needs manual causal review before promotion |
| user.txt | user_features | user | 128 | unknown | raw_only | review | needs manual causal review before promotion |
| user_list.txt | interactions | context | org_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| user_list.txt | interactions | context | remap_id | non_causal | raw_only | exclude | identifier or bookkeeping field |
| year.txt | interactions | context | 12996 | unknown | raw_only | review | needs manual causal review before promotion |
| year.txt | interactions | context | 128 | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| user.txt | .txt | 29.93 MiB | 25680 | 25680 | 2 | 25679, 128 | delimiter_guess=  |
| author.txt | .txt | 27.75 MiB | 23790 | 23790 | 2 | 23789, 128 | delimiter_guess=  |
| publisher.txt | .txt | 17.24 MiB | 14797 | 14797 | 2 | 14796, 128 | delimiter_guess=  |
| location.txt | .txt | 15.42 MiB | 13226 | 13226 | 2 | 13225, 128 | delimiter_guess=  |
| year.txt | .txt | 15.18 MiB | 12997 | 12997 | 2 | 12996, 128 | delimiter_guess=  |
| s_pre_adj_mat.npz | .npz | 5.24 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['indices', 'indptr', 'format', 'shape', 'data'] |
| train.txt | .txt | 2.76 MiB | 9433 | 9433 | - | - | schema_error=Expected 72 fields in line 5, saw 89. Error could possibly be due to quotes being ignored when a multi-char delimiter is used. |
| s_norm_adj_mat.npz | .npz | 2.59 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['indices', 'indptr', 'format', 'shape', 'data'] |
| s_mean_adj_mat.npz | .npz | 2.46 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['indices', 'indptr', 'format', 'shape', 'data'] |
| s_adj_mat.npz | .npz | 2.39 MiB | - | - | - | - | npz_arrays=5; npz_keys_preview=['indices', 'indptr', 'format', 'shape', 'data'] |
| test.txt | .txt | 0.74 MiB | 9433 | 9433 | - | - | schema_error=Expected 19 fields in line 5, saw 24. Error could possibly be due to quotes being ignored when a multi-char delimiter is used. |
| item_list.txt | .txt | 0.23 MiB | 21773 | 21773 | 2 | org_id, remap_id | delimiter_guess=  |
| user_list.txt | .txt | 0.13 MiB | 13024 | 13024 | 2 | org_id, remap_id | delimiter_guess=  |
| .DS_Store | <no_ext> | 0.01 MiB | 3 | 3 | - | - | - |

## MovieLens1M

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/MovieLens1M`
- Total files: 4
- Dataset size: 24,905,384 B (23.75 MiB | 0.02 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .dat: 3, .md: 1

### U-CaGNN Suitability

- Registry dataset: MovieLens1M
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: low
- Inferred fields: user=user_id, item=movie_id, label=rating, timestamp=timestamp
- File-level semantic inspection errors: 0
- Assessment: Strong baseline for fairness, timestamps, and rating-derived positive-negative splits.

### Causal Feature Audit

- Current experiment path: movielens1m
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, graded sign or explicit negative signal, user-side covariates available, item-side covariates available
- Main risks: textual descriptors need encoding and leakage checks before promotion, user features are loaded today but remain unused by the model
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/ratings.dat | native loader | consumed by training and evaluation | supports label/sign construction; supports temporal splitting |
| User Features | raw/users.dat | loaded into canonical user_features | retained in canonical/graph objects, not used by model | candidate pre-treatment user covariates |
| Item Features | raw/movies.dat | loaded into canonical item_features | used in Module A when canonical item_features exist | descriptor-rich item covariates that need encoding |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/movies.dat | item_features | item | title | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| raw/movies.dat | item_features | item | genres | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| raw/ratings.dat | interactions | interaction | rating | post_treatment | model_consumed | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/ratings.dat | interactions | context | timestamp | pre_treatment | model_consumed | safe_candidate | eligible for quick utility probes in the current feature-aware path |
| raw/users.dat | user_features | user | gender | pre_treatment | graph_retained | model_extension_needed | already loaded but not consumed by the current model |
| raw/users.dat | user_features | user | age | pre_treatment | graph_retained | model_extension_needed | already loaded but not consumed by the current model |
| raw/users.dat | user_features | user | occupation | pre_treatment | graph_retained | model_extension_needed | already loaded but not consumed by the current model |
| raw/users.dat | user_features | user | zip_code | pre_treatment | graph_retained | model_extension_needed | already loaded but not consumed by the current model |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/ratings.dat | .dat | 23.45 MiB | 1000209 | 1000209 | 4 | user_id, movie_id, rating, timestamp | delimiter_guess=:: |
| raw/movies.dat | .dat | 0.16 MiB | 3883 | 3883 | 3 | movie_id, title, genres | delimiter_guess=:: |
| raw/users.dat | .dat | 0.13 MiB | 6040 | 6040 | 5 | user_id, gender, age, occupation, zip_code | delimiter_guess=:: |
| raw/README.md | .md | 0.01 MiB | 170 | 170 | - | - | - |

## AmazonBook

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/AmazonBook`
- Total files: 4
- Dataset size: 20,601,125 B (19.65 MiB | 0.02 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .txt: 4

### U-CaGNN Suitability

- Registry dataset: AmazonBook
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: medium
- Inferred fields: user=user_id, item=item_ids..., label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: amazonbook
- Strongest causal asset: temporal ordering available
- Highest-priority audit level: medium
- Main opportunities: temporal ordering available
- Main risks: no major risk flagged
- Recommended next step: use as an interaction-only baseline unless new covariates are engineered

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/test.txt, raw/train.txt | native loader | consumed by training and evaluation | primary interaction source |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/item_list.txt | interactions | context | org_id | non_causal | model_consumed | exclude | identifier or bookkeeping field |
| raw/item_list.txt | interactions | context | remap_id | non_causal | model_consumed | exclude | identifier or bookkeeping field |
| raw/test.txt | interactions | interaction | item_ids... | unknown | model_consumed | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | item_ids... | unknown | model_consumed | review | needs manual causal review before promotion |
| raw/user_list.txt | interactions | context | org_id | non_causal | model_consumed | exclude | identifier or bookkeeping field |
| raw/user_list.txt | interactions | context | remap_id | non_causal | model_consumed | exclude | identifier or bookkeeping field |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/train.txt | .txt | 13.47 MiB | 52643 | 2380730 | 2 | user_id, item_ids... | rows_with_user=52643 |
| raw/test.txt | .txt | 3.67 MiB | 52643 | 603378 | 2 | user_id, item_ids... | rows_with_user=52643 |
| raw/item_list.txt | .txt | 1.47 MiB | 91600 | 91600 | 2 | org_id, remap_id | delimiter_guess=  |
| raw/user_list.txt | .txt | 1.03 MiB | 52644 | 52644 | 2 | org_id, remap_id | delimiter_guess=  |

## AmazonCDs

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/AmazonCDs`
- Total files: 4
- Dataset size: 13,662,802 B (13.03 MiB | 0.01 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .txt: 4

### U-CaGNN Suitability

- Registry dataset: AmazonCDs
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: low
- Inferred fields: user=users, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: temporal ordering available
- Highest-priority audit level: low
- Main opportunities: temporal ordering available
- Main risks: no major risk flagged
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/test.txt, raw/train.txt, raw/valid.txt | available in files only | not reachable by current experiment path | primary interaction source |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/info.txt | interactions | context | users | unknown | raw_only | review | needs manual causal review before promotion |
| raw/info.txt | interactions | context | 51267 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 127 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 30 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 1333 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 1683 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 2831 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 43024 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/train.txt | .txt | 9.12 MiB | 626692 | 626692 | 3 | 1333, 1683, 1.00 | delimiter_guess=  |
| raw/test.txt | .txt | 2.61 MiB | 179051 | 179051 | 3 | 127, 30, 1.00 | delimiter_guess=  |
| raw/valid.txt | .txt | 1.30 MiB | 89523 | 89523 | 3 | 2831, 43024, 1.00 | delimiter_guess=  |
| raw/info.txt | .txt | 0.00 MiB | 20 | 20 | 2 | users, 51267 | delimiter_guess=  |

## KuaiRand_SIGformer

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/KuaiRand_SIGformer`
- Total files: 4
- Dataset size: 3,689,226 B (3.52 MiB | 0.00 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .txt: 4

### U-CaGNN Suitability

- Registry dataset: KuaiRand_SIGformer
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: medium
- Inferred fields: user=users, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: low
- Main opportunities: temporal ordering available, graded sign or explicit negative signal
- Main risks: no major risk flagged
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/test.txt, raw/train.txt, raw/valid.txt | available in files only | not reachable by current experiment path | primary interaction source |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/info.txt | interactions | context | users | unknown | raw_only | review | needs manual causal review before promotion |
| raw/info.txt | interactions | context | 16974 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 10 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1622 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 0.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 86 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 3956 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 0.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 14939 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 208 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 0.00 | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/train.txt | .txt | 2.46 MiB | 184172 | 184172 | 3 | 86, 3956, 0.00 | delimiter_guess=  |
| raw/test.txt | .txt | 0.70 MiB | 52619 | 52619 | 3 | 10, 1622, 0.00 | delimiter_guess=  |
| raw/valid.txt | .txt | 0.35 MiB | 26309 | 26309 | 3 | 14939, 208, 0.00 | delimiter_guess=  |
| raw/info.txt | .txt | 0.00 MiB | 8 | 8 | 2 | users, 16974 | delimiter_guess=  |

## MovieLens

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/MovieLens`
- Total files: 5
- Dataset size: 3,303,135 B (3.15 MiB | 0.00 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .csv: 4, .txt: 1

### U-CaGNN Suitability

- Registry dataset: MovieLensSmall
- Pairwise triplets ready: yes
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: low
- Inferred fields: user=userId, item=movieId, label=rating, timestamp=timestamp
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: high
- Main opportunities: temporal ordering available, graded sign or explicit negative signal, item-side covariates available
- Main risks: textual descriptors need encoding and leakage checks before promotion
- Recommended next step: separate pre-treatment descriptors from post-treatment aggregates and promote only safe features

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/ml-latest-small/ratings.csv, raw/ml-latest-small/tags.csv | available in files only | not reachable by current experiment path | supports label/sign construction; supports temporal splitting |
| Item Features | raw/ml-latest-small/movies.csv, raw/ml-latest-small/tags.csv | available in files only | not reachable by current experiment path | descriptor-rich item covariates that need encoding |
| Metadata / Context | raw/ml-latest-small/links.csv, raw/ml-latest-small/tags.csv | available in files only | not reachable by current experiment path | context or auxiliary metadata |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/ml-latest-small/links.csv | metadata | context | imdbid | non_causal | raw_only | exclude | identifier or bookkeeping field |
| raw/ml-latest-small/links.csv | metadata | context | tmdbid | non_causal | raw_only | exclude | identifier or bookkeeping field |
| raw/ml-latest-small/movies.csv | item_features | item | title | pre_treatment | raw_only | encode_then_test | pre-treatment descriptor but needs encoding and leakage review |
| raw/ml-latest-small/movies.csv | item_features | item | genres | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| raw/ml-latest-small/ratings.csv | interactions | interaction | rating | post_treatment | raw_only | defer | likely downstream of exposure or outcome; keep out of default causal features |
| raw/ml-latest-small/ratings.csv | interactions | context | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |
| raw/ml-latest-small/tags.csv | item_features | item | tag | proxy | raw_only | ablation_only | potentially useful but entangled with exposure, search, or social context |
| raw/ml-latest-small/tags.csv | item_features | item | timestamp | pre_treatment | raw_only | load_then_test | looks safe enough to prototype after adding loader support |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/ml-latest-small/ratings.csv | .csv | 2.37 MiB | 100837 | 100836 | 4 | userId, movieId, rating, timestamp | columns_detected=4; delimiter_guess=,; has_header=True |
| raw/ml-latest-small/movies.csv | .csv | 0.47 MiB | 9743 | 9742 | 3 | movieId, title, genres | columns_detected=3; delimiter_guess=,; has_header=True |
| raw/ml-latest-small/links.csv | .csv | 0.19 MiB | 9743 | 9742 | 3 | movieId, imdbId, tmdbId | columns_detected=3; delimiter_guess=,; has_header=True |
| raw/ml-latest-small/tags.csv | .csv | 0.11 MiB | 3684 | 3683 | 4 | userId, movieId, tag, timestamp | columns_detected=4; delimiter_guess=,; has_header=True |
| raw/ml-latest-small/README.txt | .txt | 0.01 MiB | 153 | 153 | - | - | schema_error=Expected 1 fields in line 6, saw 5 |

## KuaiRec_SIGformer

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/KuaiRec_SIGformer`
- Total files: 4
- Dataset size: 3,226,953 B (3.08 MiB | 0.00 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .txt: 4

### U-CaGNN Suitability

- Registry dataset: KuaiRec_SIGformer
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: yes
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: medium
- Inferred fields: user=users, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: graded sign or explicit negative signal
- Highest-priority audit level: low
- Main opportunities: temporal ordering available, graded sign or explicit negative signal
- Main risks: no major risk flagged
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/test.txt, raw/train.txt, raw/valid.txt | available in files only | not reachable by current experiment path | primary interaction source |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/info.txt | interactions | context | users | unknown | raw_only | review | needs manual causal review before promotion |
| raw/info.txt | interactions | context | 1411 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1188 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1541 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 543 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 1953 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 939 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 189 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/train.txt | .txt | 2.15 MiB | 177789 | 177789 | 3 | 543, 1953, 1.00 | delimiter_guess=  |
| raw/test.txt | .txt | 0.62 MiB | 50796 | 50796 | 3 | 1188, 1541, 1.00 | delimiter_guess=  |
| raw/valid.txt | .txt | 0.31 MiB | 25398 | 25398 | 3 | 939, 189, 1.00 | delimiter_guess=  |
| raw/info.txt | .txt | 0.00 MiB | 11 | 11 | 2 | users, 1411 | delimiter_guess=  |

## Douban

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/Douban`
- Total files: 1
- Dataset size: 2,778,554 B (2.65 MiB | 0.00 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .mat: 1

### U-CaGNN Suitability

- Registry dataset: Douban
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: high
- Inferred fields: user=-, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Likely off-scope for phase 1 because the structure is not a simple user-item recommendation table.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: temporal ordering available
- Highest-priority audit level: low
- Main opportunities: temporal ordering available
- Main risks: preprocessing cost is high before the dataset can join the formal matrix
- Recommended next step: implement a canonical loader before considering formal experiments

- No causal feature sources detected.

### Candidate Column Audit

- No candidate columns detected.

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/training_test_dataset.mat | .mat | 2.65 MiB | - | - | - | - | - |

## AmazonMusic

- Status: PARTIAL (raw only)
- Root: `/home/lazar/Documents/MSc_Data_Science/MSc_Thesis/Causal-Embeddings-for-Recommendations/data/AmazonMusic`
- Total files: 4
- Dataset size: 658,345 B (0.63 MiB | 0.00 GiB)
- Raw files present: yes
- Processed files present: no
- Extension breakdown: .txt: 4

### U-CaGNN Suitability

- Registry dataset: AmazonMusic
- Pairwise triplets ready: no
- Timestamp split ready: yes
- Sign-aware split ready: no
- Popularity signal ready: no
- Side or multimodal features: no
- Preprocessing cost: low
- Inferred fields: user=users, item=-, label=-, timestamp=-
- File-level semantic inspection errors: 0
- Assessment: Potentially usable if transformed into the unified U-CaGNN ingestion contract.

### Causal Feature Audit

- Current experiment path: not in current loader registry
- Strongest causal asset: temporal ordering available
- Highest-priority audit level: low
- Main opportunities: temporal ordering available
- Main risks: no major risk flagged
- Recommended next step: implement a canonical loader before considering formal experiments

| Aspect | Source Files | Current Loader Coverage | Current Model Use | Causal Notes |
| --- | --- | --- | --- | --- |
| Interactions | raw/test.txt, raw/train.txt, raw/valid.txt | available in files only | not reachable by current experiment path | primary interaction source |

### Candidate Column Audit

| File | Aspect | Entity | Column | Causal Role | Pipeline Stage | Quick Check | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/info.txt | interactions | context | users | unknown | raw_only | review | needs manual causal review before promotion |
| raw/info.txt | interactions | context | 3472 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 152 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1769 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/test.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 2687 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 782 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/train.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 419 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 108 | unknown | raw_only | review | needs manual causal review before promotion |
| raw/valid.txt | interactions | interaction | 1.00 | unknown | raw_only | review | needs manual causal review before promotion |

### Files

| Path | Extension | Size | Line Count | Parsed Count | Column Count | Column Names | Details |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| raw/train.txt | .txt | 0.44 MiB | 34915 | 34915 | 3 | 2687, 782, 1.00 | delimiter_guess=  |
| raw/test.txt | .txt | 0.13 MiB | 9974 | 9974 | 3 | 152, 1769, 1.00 | delimiter_guess=  |
| raw/valid.txt | .txt | 0.06 MiB | 4986 | 4986 | 3 | 419, 108, 1.00 | delimiter_guess=  |
| raw/info.txt | .txt | 0.00 MiB | 14 | 14 | 2 | users, 3472 | delimiter_guess=  |
