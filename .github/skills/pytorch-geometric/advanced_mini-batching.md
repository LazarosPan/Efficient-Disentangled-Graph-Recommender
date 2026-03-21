# Advanced Mini-Batching

The creation of mini-batching is crucial for letting the training of a deep learning model scale to huge amounts of data. Instead of processing examples one-by-one, a mini-batch groups a set of examples into a unified representation where it can efficiently be processed in parallel. In the image or language domain, this procedure is typically achieved by rescaling or padding each example into a set to equally-sized shapes, and examples are then grouped in an additional dimension. The length of this dimension is then equal to the number of examples grouped in a mini-batch and is typically referred to as the `batch_size`.

Since graphs are one of the most general data structures that can hold any number of nodes or edges, the two approaches described above are either not feasible or may result in a lot of unnecessary memory consumption. In PyG, we opt for another approach to achieve parallelization across a number of examples. Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple isolated subgraphs), and node and target features are simply concatenated in the node dimension, i.e.

$$
\begin{split}
\mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix},
\qquad
\mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix},
\qquad
\mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}.
\end{split}
$$

This procedure has some crucial advantages over other batching procedures:

- GNN operators that rely on a message passing scheme do not need to be modified since messages still cannot be exchanged between two nodes that belong to different graphs.
- There is no computational or memory overhead. For example, this batching procedure works completely without any padding of node or edge features. Note that there is no additional memory overhead for adjacency matrices since they are saved in a sparse fashion holding only non-zero entries, i.e., the edges.

PyG automatically takes care of batching multiple graphs into a single giant graph with the help of the `torch_geometric.loader.DataLoader` class. Internally, `DataLoader` is just a regular PyTorch `torch.utils.data.DataLoader` that overwrites its `collate()` functionality, i.e., the definition of how a list of examples should be grouped together. Therefore, all arguments that can be passed to a PyTorch `DataLoader` can also be passed to a PyG `DataLoader`, e.g., the number of workers `num_workers`.

In its most general form, the PyG `DataLoader` will automatically increment the `edge_index` tensor by the cumulated number of nodes of all graphs that got collated before the currently processed graph, and will concatenate `edge_index` tensors (that are of shape `[2, num_edges]`) in the second dimension. The same is true for `face` tensors, i.e., face indices in meshes. All other tensors will just get concatenated in the first dimension without any further increasement of their values.

However, there are a few special use-cases (as outlined below) where the user actively wants to modify this behavior to its own needs. PyG allows modification to the underlying batching procedure by overwriting the `torch_geometric.data.Data.__inc__()` and `torch_geometric.data.Data.__cat_dim__()` functionalities. Without any modifications, these are defined as follows in the `Data` class:

```python
def __inc__(self, key, value, *args, **kwargs):
    if 'index' in key:
        return self.num_nodes
    else:
        return 0


def __cat_dim__(self, key, value, *args, **kwargs):
    if 'index' in key:
        return 1
    else:
        return 0
```

We can see that `__inc__()` defines the incremental count between two consecutive graph attributes. By default, PyG increments attributes by the number of nodes whenever their attribute names contain the substring `index` (for historical reasons), which comes in handy for attributes such as `edge_index` or `node_index`. However, note that this may lead to unexpected behavior for attributes whose names contain the substring `index` but should not be incremented. To make sure, it is best practice to always double-check the output of batching. Furthermore, `__cat_dim__()` defines in which dimension graph tensors of the same attribute should be concatenated together. Both functions are called for each attribute stored in the `Data` class, and get passed their specific `key` and `value` item as arguments.

In what follows, we present a few use-cases where the modification of `__inc__()` and `__cat_dim__()` might be absolutely necessary.

## Pairs of Graphs

In case you want to store multiple graphs in a single `Data` object, e.g., for applications such as graph matching, you need to ensure correct batching behavior across all those graphs. For example, consider storing two graphs, a source graph and a target graph in a `Data`, e.g.:

```python
from torch_geometric.data import Data


class PairData(Data):
    pass


data = PairData(x_s=x_s, edge_index_s=edge_index_s,  # Source graph.
                x_t=x_t, edge_index_t=edge_index_t)  # Target graph.
```

In this case, `edge_index_s` should be increased by the number of nodes in the source graph, e.g., `x_s.size(0)`, and `edge_index_t` should be increased by the number of nodes in the target graph, e.g., `x_t.size(0)`:

```python
class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)
```

We can test our `PairData` batching behavior by setting up a simple test script:

```python
from torch_geometric.loader import DataLoader

x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])

x_t = torch.randn(4, 16)  # 4 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])

data = PairData(x_s=x_s, edge_index_s=edge_index_s,
                x_t=x_t, edge_index_t=edge_index_t)

data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
>>> PairDataBatch(x_s=[10, 16], edge_index_s=[2, 8],
                  x_t=[8, 16], edge_index_t=[2, 6])

print(batch.edge_index_s)
>>> tensor([[0, 0, 0, 0, 5, 5, 5, 5],
            [1, 2, 3, 4, 6, 7, 8, 9]])

print(batch.edge_index_t)
>>> tensor([[0, 0, 0, 4, 4, 4],
            [1, 2, 3, 5, 6, 7]])
```

Everything looks good so far. `edge_index_s` and `edge_index_t` get correctly batched together, even when using different numbers of nodes for the source and target graph. However, the `batch` attribute (that maps each node to its respective graph) is missing since PyG fails to identify the actual graph in the `PairData` object. That is where the `follow_batch` argument of the `DataLoader` comes into play. Here, we can specify for which attributes we want to maintain the batch information:

```python
loader = DataLoader(data_list, batch_size=2, follow_batch=['x_s', 'x_t'])
batch = next(iter(loader))

print(batch)
>>> PairDataBatch(x_s=[10, 16], edge_index_s=[2, 8], x_s_batch=[10],
                  x_t=[8, 16], edge_index_t=[2, 6], x_t_batch=[8])

print(batch.x_s_batch)
>>> tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

print(batch.x_t_batch)
>>> tensor([0, 0, 0, 0, 1, 1, 1, 1])
```

As one can see, `follow_batch=['x_s', 'x_t']` now successfully creates assignment vectors `x_s_batch` and `x_t_batch` for the node features `x_s` and `x_t`, respectively. That information can now be used to perform reduce operations, e.g., global pooling, on multiple graphs in a single `Batch` object.

## Bipartite Graphs

The adjacency matrix of a bipartite graph defines the relationship between nodes of two different node types. In general, the number of nodes for each node type do not need to match, resulting in a non-quadratic adjacency matrix. In a mini-batching procedure of bipartite graphs, the source nodes of edges in `edge_index` should get increased differently than the target nodes of edges in `edge_index`. To achieve this, consider a bipartite graph between two node types with corresponding node features `x_s` and `x_t`, respectively:

```python
from torch_geometric.data import Data


class BipartiteData(Data):
    pass


data = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
```

For a correct mini-batching procedure in bipartite graphs, we need to tell PyG that it should increment source and target nodes of edges in `edge_index` independently:

```python
class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
```

Here, `edge_index[0]` (the source nodes of edges) get incremented by `x_s.size(0)` while `edge_index[1]` (the target nodes of edges) get incremented by `x_t.size(0)`. We can again test our implementation by running a simple test script:

```python
from torch_geometric.loader import DataLoader

x_s = torch.randn(2, 16)  # 2 nodes.
x_t = torch.randn(3, 16)  # 3 nodes.
edge_index = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
])

data = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)

data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
>>> BipartiteDataBatch(x_s=[4, 16], x_t=[6, 16], edge_index=[2, 8])

print(batch.edge_index)
>>> tensor([[0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 1, 2, 3, 4, 4, 5]])
```

Again, this is exactly the behavior we aimed for.

## Batching Along New Dimensions

Sometimes, attributes of data objects should be batched by gaining a new batch dimension (as in classical mini-batching), e.g., for graph-level properties or targets. Specifically, a list of attributes of shape `[num_features]` should be returned as `[num_examples, num_features]` rather than `[num_examples * num_features]`. PyG achieves this by returning a concatenation dimension of `None` in `__cat_dim__()`:

```python
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'foo':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


edge_index = torch.tensor([
   [0, 1, 1, 2],
   [1, 0, 2, 1],
])
foo = torch.randn(16)

data = MyData(num_nodes=3, edge_index=edge_index, foo=foo)

data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
>>> MyDataBatch(num_nodes=6, edge_index=[2, 8], foo=[2, 16])
```

As desired, `batch.foo` is now described by two dimensions: the batch dimension and the feature dimension.
