from .canonical import CanonicalInteractions

# Heavy modules (graph_builder, negative_sampler, subgraph_sampler) are
# imported lazily via __getattr__ to avoid loading PyG/CUDA at package import.
__all__ = [
    "CanonicalInteractions",
    "NegativeSampler",
    "SubgraphBatch",
    "SubgraphSampler",
    "build_graph",
]


def __getattr__(name: str):
    if name == "build_graph":
        from .graph_builder import build_graph

        return build_graph
    if name == "NegativeSampler":
        from .negative_sampler import NegativeSampler

        return NegativeSampler
    if name in {"SubgraphSampler", "SubgraphBatch"}:
        from .subgraph_sampler import SubgraphBatch, SubgraphSampler

        return {"SubgraphSampler": SubgraphSampler, "SubgraphBatch": SubgraphBatch}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
