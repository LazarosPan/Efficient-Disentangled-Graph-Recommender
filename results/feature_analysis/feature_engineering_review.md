# Feature Engineering Review

Purpose: explain which dataset features are used for EDGRec training, which are left outside the thesis-default path, and what evidence still needs a full-data test rerun before becoming a thesis claim.

Primary tables/reports: `dataset_feature_decision_map.md`, `feature_subset_evidence_matrix.md`, `feature_subset_best_by_dataset.md`, and `feature_subset_results.csv`.

## Dataset Decisions

| Dataset / feedback context | Feature input | Left outside training | Evidence | Decision |
|---|---|---|---|---|
| Amazon Book<br/>3.0M interactions<br/>0.0619% density<br/>implicit observed interactions only<br/>label/sign: label>0 100.0%; sign>0 0.0%, sign=0 100.0%, sign<0 0.0% | No side-feature source; interaction graph only. | Nothing feature-bearing was omitted. | Best: graph only<br/>ValCRRU 0.144116<br/>side gain n/a | Train graph-only; no side-feature claim. |
| MovieLens-1M<br/>1.0M interactions<br/>4.47% density<br/>explicit 1-to-5 star ratings<br/>label/sign: label>0 57.5%; sign>0 57.5%, sign=0 26.1%, sign<0 16.4% | item genre (18)<br/>Not searched: user demographics (3) | Zip code is proxy-only; demographics are not searched/trained in current EDGRec. | Best: no side features<br/>ValCRRU 0.262050<br/>side gain -0.013 | Prefer no side features in current EDGRec basin; keep feature result as negative evidence. |
| MovieLens-20M<br/>20.0M interactions<br/>0.54% density<br/>explicit 1-to-5 star ratings<br/>label/sign: label>0 50.0%; sign>0 61.0%, sign=0 21.5%, sign<0 17.6% | Genres are available; no current feature-subset search or EDGRec test row. | Genome/tag text evidence is outside the current thesis-default path. | No completed feature-subset evidence. | Context dataset only for now; run feature-subset search before using feature claims. |
| KuaiRec v2<br/>10.3M interactions<br/>13.4% density<br/>watch-ratio feedback from short-video viewing<br/>label/sign: label>0 66.8%; sign>0 36.2%, sign=0 0.0%, sign<0 63.8%; label>0 & sign<0 3,153,181 | author/music (2)<br/>category/tags (35)<br/>resolution (2)<br/>upload time (1)<br/>video metadata (4) | User profiles, captions/free text, and engagement counts are excluded or proxy-only. | Best: resolution<br/>ValCRRU 0.215856<br/>side gain +0.002<br/>strongest: resolution (+0.051) | Use resolution/video-metadata feature reruns as candidates; final claim needs matching test rows. |
| Taobao<br/>76.0M interactions<br/>0.00185% density<br/>multi-behavior shopping interactions<br/>label/sign: label>0 12.6%; sign>0 12.6%, sign=0 0.0%, sign<0 87.4% | Category id is available; no current feature-subset search or EDGRec test row. | Behavior labels and timestamps are outcomes/context, not side features for the current model. | No completed feature-subset evidence. | Context dataset only for now; run feature-subset search before using feature claims. |
| KuaiRand-1K<br/>11.4M interactions<br/>0.261% density<br/>short-video engagement logs with randomized exposure<br/>label/sign: label>0 39.0%; sign>0 27.5%, sign=0 72.4%, sign<0 0.1%; label>0 & sign<0 1,651 | author/music (3)<br/>category/tags (58)<br/>resolution (2)<br/>upload time (1)<br/>video metadata (4) | Statistic engagement file is excluded; show_cnt is only a propensity target when IPW is explicit. | Best: author/music + upload time + category/tags<br/>ValCRRU 0.062106<br/>side gain +0.020<br/>strongest: author/music + upload time + category/tags (+0.025) | Use category/triple features as candidates; test outside compact diagnostic regime before a headline. |

## What This Means

- AmazonBook is a graph-only recommendation dataset in the current code path; no side-feature engineering claim should be made.
- MovieLens-1M genre features are negative validation evidence in the current EDGRec basin; user demographics are loaded as metadata but not used by the item-only context head.
- KuaiRec v2 has the strongest validation signal for side features, especially resolution and video metadata, but feature-specific full-data test rows are needed before a test-set claim.
- KuaiRand-1K has positive validation signal for category/triple features, but current compact randomized-exposure results are diagnostic; standard-regime feature reruns are needed for a headline.
- MovieLens-20M and Taobao are dataset-analysis context unless they receive matching EDGRec feature-subset searches and test rows.

## Not Done Yet

- Full-data test reruns for KuaiRec `item_resolution`, `item_video_metadata`, and `item_video_metadata + item_resolution` candidates.
- Full-data KuaiRand rerun for `item_author_music + item_upload_time + item_category`, preferably outside the ultra-compact diagnostic-only setup.
- Explicit statement in slides that user-side features are not part of the current EDGRec scorer; otherwise a committee may ask why demographics were loaded but not trained.
- If Taobao or MovieLens-20M become headline datasets, add their own feature-subset searches rather than extrapolating from the four current feature-search datasets.
