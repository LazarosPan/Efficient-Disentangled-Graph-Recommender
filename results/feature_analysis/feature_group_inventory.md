# Feature Group Inventory

Loaded thesis-default feature columns grouped by dataset and entity.
Feature-effect metrics are intentionally absent from this inventory.
`search_candidate` means the safe item group is eligible for the feature-subset search; it is not a pending experiment status.

| Dataset | Entity | Group | LoadedColumns | FeatureSubsetStatus |
|---|---|---|---:|---|
| amazonbook | item | graph_only | 1 | not_applicable |
| kuairand1k | item | item_author_music | 3 | search_candidate |
| kuairand1k | item | item_category | 58 | search_candidate |
| kuairand1k | item | item_resolution | 2 | search_candidate |
| kuairand1k | item | item_upload_time | 1 | search_candidate |
| kuairand1k | item | item_video_metadata | 4 | search_candidate |
| kuairec_v2 | item | item_author_music | 2 | search_candidate |
| kuairec_v2 | item | item_category | 35 | search_candidate |
| kuairec_v2 | item | item_resolution | 2 | search_candidate |
| kuairec_v2 | item | item_upload_time | 1 | search_candidate |
| kuairec_v2 | item | item_video_metadata | 4 | search_candidate |
| movielens1m | item | item_genre | 18 | search_candidate |
| movielens1m | user | user_demographic | 3 | not_searched |
