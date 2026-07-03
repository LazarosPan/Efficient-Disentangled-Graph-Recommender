# Feature Subset Evidence Matrix

Cells are validation/search deltas, not test-set claims. Drop importance = all-features score minus score after removing the group.

| Dataset | All side vs none | Best single group | Best pair/triple | Best drop importance | Best validation profile | ValCRRU | Thesis use |
|---|---:|---:|---:|---:|---|---:|---|
| Amazon Book | n/a | n/a | n/a | n/a | graph only | 0.144116 | graph-only |
| MovieLens-1M | -0.013 | -0.014 item genre | n/a | -0.008 item genre | no side features | 0.262050 | negative evidence |
| KuaiRec v2 | +0.002 | +0.051 resolution | +0.049 video metadata + resolution | +0.033 category/tags | single: resolution | 0.215856 | rerun before claim |
| KuaiRand-1K | +0.020 | +0.018 category/tags | +0.025 author/music + upload time + category/tags | +0.011 category/tags | triple: author/music + upload time + category/tags | 0.062106 | rerun before claim |
