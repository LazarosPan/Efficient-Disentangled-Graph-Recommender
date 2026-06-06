# `torch_geometric.sampler`

## Overview

- `BaseSampler`
- `NodeSamplerInput`
- `EdgeSamplerInput`
- `SamplerOutput`
- `HeteroSamplerOutput`
- `NumNeighbors`
- `NegativeSampling`
- `NeighborSampler`
- `BidirectionalNeighborSampler`
- `HGTSampler`

## `BaseSampler`

An abstract base class that initializes a graph sampler and provides `sample_from_nodes()` and `sample_from_edges()` routines.

> **Note**
>
> Any data stored in the sampler will be replicated across data loading workers that use the sampler since each data loading worker holds its own instance of a sampler. As such, it is recommended to limit the amount of information stored in the sampler.

### `sample_from_nodes(index: NodeSamplerInput, **kwargs) -> Union[HeteroSamplerOutput, SamplerOutput]`

Performs sampling from the nodes specified in `index`, returning a sampled subgraph in the specified output format.

The index is a tuple holding the following information:

- The example indices of the seed nodes
- The node indices to start sampling from
- The timestamps of the given seed nodes (optional)

**Parameters:**

- `index` (`NodeSamplerInput`) — The node sampler input object.
- `**kwargs` (optional) — Additional keyword arguments.

**Return type:**

- `Union[HeteroSamplerOutput, SamplerOutput]`

### `sample_from_edges(index: EdgeSamplerInput, neg_sampling: Optional[NegativeSampling] = None) -> Union[HeteroSamplerOutput, SamplerOutput]`

Performs sampling from the edges specified in `index`, returning a sampled subgraph in the specified output format.

The index is a tuple holding the following information:

- The example indices of the seed links
- The source node indices to start sampling from
- The destination node indices to start sampling from
- The labels of the seed links (optional)
- The timestamps of the given seed nodes (optional)

**Parameters:**

- `index` (`EdgeSamplerInput`) — The edge sampler input object.
- `neg_sampling` (`NegativeSampling`, optional) — The negative sampling configuration. (default: `None`)

**Return type:**

- `Union[HeteroSamplerOutput, SamplerOutput]`

### `edge_permutation`

If the sampler performs any modification of edge ordering in the original graph, this function is expected to return the permutation tensor that defines the permutation from the edges in the original graph and the edges used in the sampler. If no such permutation was applied, `None` is returned. For heterogeneous graphs, the expected return type is a permutation tensor for each edge type.

**Return type:**

- `Union[Tensor, None, Dict[Tuple[str, str, str], Optional[Tensor]]]`

Graph sampler package.

## `NodeSamplerInput`

```python
class NodeSamplerInput(input_id: Optional[Tensor], node: Tensor, time: Optional[Tensor] = None, input_type: Optional[str] = None)
```

The sampling input of `sample_from_nodes()`.

**Parameters:**

- `input_id` (`torch.Tensor`, optional) — The indices of the data loader input of the current mini-batch.
- `node` (`torch.Tensor`) — The indices of seed nodes to start sampling from.
- `time` (`torch.Tensor`, optional) — The timestamp for the seed nodes. (default: `None`)
- `input_type` (`str`, optional) — The input node type (in case of sampling in a heterogeneous graph). (default: `None`)

## `EdgeSamplerInput`

```python
class EdgeSamplerInput(input_id: Optional[Tensor], row: Tensor, col: Tensor, label: Optional[Tensor] = None, time: Optional[Tensor] = None, input_type: Optional[Tuple[str, str, str]] = None)
```

The sampling input of `sample_from_edges()`.

**Parameters:**

- `input_id` (`torch.Tensor`, optional) — The indices of the data loader input of the current mini-batch.
- `row` (`torch.Tensor`) — The source node indices of seed links to start sampling from.
- `col` (`torch.Tensor`) — The destination node indices of seed links to start sampling from.
- `label` (`torch.Tensor`, optional) — The label for the seed links. (default: `None`)
- `time` (`torch.Tensor`, optional) — The timestamp for the seed links. (default: `None`)
- `input_type` (`Tuple[str, str, str]`, optional) — The input edge type (in case of sampling in a heterogeneous graph). (default: `None`)

## `SamplerOutput`

```python
class SamplerOutput(node: Tensor, row: Tensor, col: Tensor, edge: Optional[Tensor], batch: Optional[Tensor] = None, num_sampled_nodes: Optional[List[int]] = None, num_sampled_edges: Optional[List[int]] = None, orig_row: Optional[Tensor] = None, orig_col: Optional[Tensor] = None, metadata: Optional[Any] = None, _seed_node: Optional[Tensor] = None)
```

The sampling output of a `BaseSampler` on homogeneous graphs.

**Parameters:**

- `node` (`torch.Tensor`) — The sampled nodes in the original graph.
- `row` (`torch.Tensor`) — The source node indices of the sampled subgraph. Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }` corresponding to the nodes in the `node` tensor.
- `col` (`torch.Tensor`) — The destination node indices of the sampled subgraph. Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }` corresponding to the nodes in the `node` tensor.
- `edge` (`torch.Tensor`, optional) — The sampled edges in the original graph. This tensor is used to obtain edge features from the original graph. If no edge attributes are present, it may be omitted.
- `batch` (`torch.Tensor`, optional) — The vector to identify the seed node for each sampled node. Can be present in case of disjoint subgraph sampling per seed node. (default: `None`)
- `num_sampled_nodes` (`List[int]`, optional) — The number of sampled nodes per hop. (default: `None`)
- `num_sampled_edges` (`List[int]`, optional) — The number of sampled edges per hop. (default: `None`)
- `orig_row` (`torch.Tensor`, optional) — The original source node indices returned by the sampler. Filled in case `to_bidirectional()` is called with the `keep_orig_edges` option. (default: `None`)
- `orig_col` (`torch.Tensor`, optional) — The original destination node indices returned by the sampler. Filled in case `to_bidirectional()` is called with the `keep_orig_edges` option. (default: `None`)
- `metadata` (`Optional[Any]`, default: `None`) — Additional metadata information.

### `to_bidirectional(keep_orig_edges: bool = False) -> SamplerOutput`

Converts the sampled subgraph into a bidirectional variant, in which all sampled edges are guaranteed to be bidirectional.

**Parameters:**

- `keep_orig_edges` (`bool`, optional) — If specified, directional edges are still maintained. (default: `False`)

### `collate(outputs: List[SamplerOutput], replace: bool = True) -> SamplerOutput`

Collate a list of `SamplerOutput` objects into a single `SamplerOutput` object. Requires that they all have the same fields.

### `merge_with(other: SamplerOutput, replace: bool = True) -> SamplerOutput`

Merges two `SamplerOutput` objects. If `replace` is `True`, `self`’s nodes and edges take precedence.

## `HeteroSamplerOutput`

```python
class HeteroSamplerOutput(node: Dict[str, Tensor], row: Dict[Tuple[str, str, str], Tensor], col: Dict[Tuple[str, str, str], Tensor], edge: Dict[Tuple[str, str, str], Optional[Tensor]], batch: Optional[Dict[str, Tensor]] = None, num_sampled_nodes: Optional[Dict[str, List[int]]] = None, num_sampled_edges: Optional[Dict[Tuple[str, str, str], List[int]]] = None, orig_row: Optional[Dict[Tuple[str, str, str], Tensor]] = None, orig_col: Optional[Dict[Tuple[str, str, str], Tensor]] = None, metadata: Optional[Any] = None)
```

The sampling output of a `BaseSampler` on heterogeneous graphs.

**Parameters:**

- `node` (`Dict[str, torch.Tensor]`) — The sampled nodes in the original graph for each node type.
- `row` (`Dict[Tuple[str, str, str], torch.Tensor]`) — The source node indices of the sampled subgraph for each edge type. Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }` corresponding to the nodes in the `node` tensor of the source node type.
- `col` (`Dict[Tuple[str, str, str], torch.Tensor]`) — The destination node indices of the sampled subgraph for each edge type. Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }` corresponding to the nodes in the `node` tensor of the destination node type.
- `edge` (`Dict[Tuple[str, str, str], torch.Tensor]`, optional) — The sampled edges in the original graph for each edge type. This tensor is used to obtain edge features from the original graph. If no edge attributes are present, it may be omitted.
- `batch` (`Dict[str, torch.Tensor]`, optional) — The vector to identify the seed node for each sampled node for each node type. Can be present in case of disjoint subgraph sampling per seed node. (default: `None`)
- `num_sampled_nodes` (`Dict[str, List[int]]`, optional) — The number of sampled nodes for each node type and each layer. (default: `None`)
- `num_sampled_edges` (`Dict[EdgeType, List[int]]`, optional) — The number of sampled edges for each edge type and each layer. (default: `None`)
- `orig_row` (`Dict[EdgeType, torch.Tensor]`, optional) — The original source node indices returned by the sampler. Filled in case `to_bidirectional()` is called with the `keep_orig_edges` option. (default: `None`)
- `orig_col` (`Dict[EdgeType, torch.Tensor]`, optional) — The original destination node indices returned by the sampler. Filled in case `to_bidirectional()` is called with the `keep_orig_edges` option. (default: `None`)
- `metadata` (`Optional[Any]`, default: `None`) — Additional metadata information.

### `to_bidirectional(keep_orig_edges: bool = False) -> SamplerOutput`

Converts the sampled subgraph into a bidirectional variant, in which all sampled edges are guaranteed to be bidirectional.

### `collate(outputs: List[HeteroSamplerOutput], replace: bool = True) -> HeteroSamplerOutput`

Collate a list of `HeteroSamplerOutput` objects. Requires that they all have the same fields.

### `merge_with(other: HeteroSamplerOutput, replace: bool = True) -> HeteroSamplerOutput`

Merges two `HeteroSamplerOutput` objects. If `replace` is `True`, `self`’s nodes and edges take precedence.

## `NumNeighbors`

```python
class NumNeighbors(values: Union[List[int], Dict[Tuple[str, str, str], List[int]]], default: Optional[List[int]] = None)
```

The number of neighbors to sample in a homogeneous or heterogeneous graph. In heterogeneous graphs, may also take in a dictionary denoting the amount of neighbors to sample for individual edge types.

**Parameters:**

- `values` (`List[int]` or `Dict[Tuple[str, str, str], List[int]]`) — The number of neighbors to sample. If an entry is set to `-1`, all neighbors will be included. In heterogeneous graphs, may also take in a dictionary denoting the amount of neighbors to sample for individual edge types.
- `default` (`List[int]`, optional) — The default number of neighbors for edge types not specified in `values`. (default: `None`)

### `get_values(edge_types: Optional[List[Tuple[str, str, str]]] = None) -> Union[List[int], Dict[Tuple[str, str, str], List[int]]]`

Returns the number of neighbors.

### `get_mapped_values(edge_types: Optional[List[Tuple[str, str, str]]] = None) -> Union[List[int], Dict[str, List[int]]]`

Returns the number of neighbors. For heterogeneous graphs, a dictionary is returned in which edge type tuples are converted to strings.

### `num_hops`

Returns the number of hops.

## `NegativeSampling`

```python
class NegativeSampling(mode: Union[NegativeSamplingMode, str], amount: Union[int, float] = 1, src_weight: Optional[Tensor] = None, dst_weight: Optional[Tensor] = None)
```

The negative sampling configuration of a `BaseSampler` when calling `sample_from_edges()`.

**Parameters:**

- `mode` (`str`) — The negative sampling mode (`"binary"` or `"triplet"`). If set to `"binary"`, will randomly sample negative links from the graph. If set to `"triplet"`, will randomly sample negative destination nodes for each positive source node.
- `amount` (`int` or `float`, optional) — The ratio of sampled negative edges to the number of positive edges. (default: `1`)
- `src_weight` (`torch.Tensor`, optional) — A node-level vector determining the sampling of source nodes. Does not necessarily need to sum up to one. If not given, negative nodes will be sampled uniformly. (default: `None`)
- `dst_weight` (`torch.Tensor`, optional) — A node-level vector determining the sampling of destination nodes. Does not necessarily need to sum up to one. If not given, negative nodes will be sampled uniformly. (default: `None`)

### `sample(num_samples: int, endpoint: Literal['src', 'dst'], num_nodes: Optional[int] = None) -> Tensor`

Generates `num_samples` negative samples.

## Sampler Implementations

- `NeighborSampler(data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]], num_neighbors: Union[NumNeighbors, List[int], Dict[Tuple[str, str, str], List[int]]], subgraph_type: Union[SubgraphType, str] = 'directional', replace: bool = False, disjoint: bool = False, temporal_strategy: str = 'uniform', time_attr: Optional[str] = None, weight_attr: Optional[str] = None, is_sorted: bool = False, share_memory: bool = False, directed: bool = True, sample_direction: Literal['forward', 'backward'] = 'forward')` — An implementation of an in-memory (heterogeneous) neighbor sampler used by `NeighborLoader`.
- `BidirectionalNeighborSampler(data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]], num_neighbors: Union[NumNeighbors, List[int], Dict[Tuple[str, str, str], List[int]]], subgraph_type: Union[SubgraphType, str] = 'directional', replace: bool = False, disjoint: bool = False, temporal_strategy: str = 'uniform', time_attr: Optional[str] = None, weight_attr: Optional[str] = None, is_sorted: bool = False, share_memory: bool = False, directed: bool = True)` — A sampler that allows for both upstream and downstream sampling.
- `HGTSampler(data: HeteroData, num_samples: Union[List[int], Dict[str, List[int]]], is_sorted: bool = False, share_memory: bool = False)` — An implementation of an in-memory heterogeneous layer-wise sampler used by `HGTLoader`.
