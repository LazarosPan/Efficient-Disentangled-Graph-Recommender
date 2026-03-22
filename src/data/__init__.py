from .canonical import CanonicalInteractions

__all__ = [
    "CanonicalInteractions",
    "build_graph",
    "NegativeSampler",
    "SubgraphSampler",
    "SubgraphBatch",
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
