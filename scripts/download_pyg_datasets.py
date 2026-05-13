#!/usr/bin/env python
"""Download selected PyTorch-Geometric datasets to data/<dataset>/ in repository root."""

from pathlib import Path

import torch_geometric.datasets as pyg_datasets


def main():
    # new layout: data/<dataset>/{raw,processed,...}
    base_data = Path(__file__).parent.parent / "data"
    base_data.mkdir(parents=True, exist_ok=True)

    def load_if_needed(dataset_name, cls, **kwargs):
        dataset_root = base_data / dataset_name
        raw = dataset_root / "raw"
        proc = dataset_root / "processed" / "data.pt"
        dataset_root.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        proc.parent.mkdir(parents=True, exist_ok=True)

        exists = proc.exists() or (raw.exists() and any(raw.iterdir()))
        if exists:
            print(
                f"Skipping {dataset_name}; data already present (processed or raw files)",
            )
            return None
        try:
            return cls(root=str(dataset_root), **kwargs)
        except ModuleNotFoundError as e:
            print(f"⚠ Skipping {dataset_name}; missing dependency: {e.name}")
            return None
        except Exception as e:
            print(f"⚠ Error loading {dataset_name}: {e}")
            return None

    datasets = {
        # "Yelp": load_if_needed("Yelp", pyg_datasets.Yelp),
        # "AmazonProducts": load_if_needed("AmazonProducts", pyg_datasets.AmazonProducts),
        # "MovieLens": load_if_needed("MovieLens", pyg_datasets.MovieLens),
        # "MovieLens100K": load_if_needed("MovieLens100K", pyg_datasets.MovieLens100K),
        "MovieLens1M": load_if_needed("MovieLens1M", pyg_datasets.MovieLens1M),
        # "Taobao": load_if_needed("Taobao", pyg_datasets.Taobao), downloaded from https://tianchi.aliyun.com/dataset/649?lang=en-us,
        # IGMCDataset needs a name argument; download only Douban,
        # "Douban": load_if_needed("Douban", pyg_datasets.IGMCDataset, name="Douban"),
        "AmazonBook": load_if_needed("AmazonBook", pyg_datasets.AmazonBook),
        # "KuaiRec_v2": https://kuairec.com/,
        # "KuaiRand1K": https://kuairec.com/,
        # Movielens20M https://grouplens.org/datasets/movielens/20m/
    }

    print("\nDownloaded the following datasets (None = skipped/failure):")
    for name, ds in datasets.items():
        if ds is None:
            print(f" - {name}: <skipped>")
        else:
            try:
                count = len(ds)
            except Exception:
                count = "?"
            print(f" - {name}: {count} graph(s)")


if __name__ == "__main__":
    main()
