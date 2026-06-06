# Memory-Efficient Aggregations

The `MessagePassing` interface of PyG relies on a gather-scatter scheme to aggregate messages from neighboring nodes. For example, consider the message passing layer

$$
\mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \textrm{MLP}(\mathbf{x}_j - \mathbf{x}_i),
$$

that can be implemented as:

```python
from torch_geometric.nn import MessagePassing

x = ...           # Node features of shape [num_nodes, num_features]
edge_index = ...  # Edge indices of shape [2, num_edges]


class MyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return MLP(x_j - x_i)
```

Under the hood, the `MessagePassing` implementation produces code that looks as follows:

```python
from torch_geometric.utils import scatter

x = ...           # Node features of shape [num_nodes, num_features]
edge_index = ...  # Edge indices of shape [2, num_edges]

x_j = x[edge_index[0]]  # Source node features [num_edges, num_features]
x_i = x[edge_index[1]]  # Target node features [num_edges, num_features]

msg = MLP(x_j - x_i)  # Compute message for each edge

# Aggregate messages based on target node indices
out = scatter(msg, edge_index[1], dim=0, dim_size=x.size(0), reduce='sum')
```

While the gather-scatter formulation generalizes to a lot of useful GNN implementations, it has the disadvantage of explicitly materializing `x_j` and `x_i`, resulting in a high memory footprint on large and dense graphs.

Luckily, not all GNNs need to be implemented by explicitly materializing `x_j` and/or `x_i`. In some cases, GNNs can also be implemented as a simple sparse matrix multiplication. As a general rule of thumb, this holds true for GNNs that do not make use of the central node features `x_i` or multi-dimensional edge features when computing messages. For example, the `GINConv` layer

$$
\mathbf{x}^{\prime}_i = \textrm{MLP} \left( (1 + \epsilon) \cdot \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),
$$

is equivalent to computing

$$
\mathbf{X}^{\prime} = \textrm{MLP} \left( (1 + \epsilon) \cdot \mathbf{X} + \mathbf{A}\mathbf{X} \right),
$$

where $\mathbf{A}$ denotes a sparse adjacency matrix of shape `[num_nodes, num_nodes]`. This formulation allows us to leverage dedicated and fast sparse-matrix multiplication implementations.

In PyG `>= 1.6.0`, we officially introduce better support for sparse-matrix multiplication GNNs, resulting in a lower memory footprint and a faster execution time. As a result, we introduce the `SparseTensor` class (from the `torch_sparse` package), which implements fast forward and backward passes for sparse-matrix multiplication based on the *“Design Principles for Sparse Matrix Multiplication on the GPU”* paper.

Using the `SparseTensor` class is straightforward and similar to the way `scipy` treats sparse matrices:

```python
from torch_sparse import SparseTensor

adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=...,
                   sparse_sizes=(num_nodes, num_nodes))
# value is optional and can be None

# Obtain different representations (COO, CSR, CSC):
row,    col, value = adj.coo()
rowptr, col, value = adj.csr()
colptr, row, value = adj.csc()

adj = adj[:100, :100]  # Slicing, indexing and masking support
adj = adj.set_diag()   # Add diagonal entries
adj_t = adj.t()        # Transpose
out = adj.matmul(x)    # Sparse-dense matrix multiplication
adj = adj.matmul(adj)  # Sparse-sparse matrix multiplication

# Creating SparseTensor instances:
adj = SparseTensor.from_dense(mat)
adj = SparseTensor.eye(100, 100)
adj = SparseTensor.from_scipy(mat)
```

Our `MessagePassing` interface can handle both `torch.Tensor` and `SparseTensor` as input for propagating messages. However, when holding a directed graph in `SparseTensor`, you need to make sure to input the transposed sparse matrix to `propagate()`:

```python
conv = GCNConv(16, 32)
out1 = conv(x, edge_index)
out2 = conv(x, adj.t())
assert torch.allclose(out1, out2)

conv = GINConv(nn=Sequential(Linear(16, 32), ReLU(), Linear(32, 32)))
out1 = conv(x, edge_index)
out2 = conv(x, adj.t())
assert torch.allclose(out1, out2)
```

To leverage sparse-matrix multiplications, the `MessagePassing` interface introduces the `message_and_aggregate()` function (which fuses the `message()` and `aggregate()` functions into a single computation step), which gets called whenever it is implemented and receives a `SparseTensor` as input for `edge_index`. With it, the `GINConv` layer can now be implemented as follows:

```python
import torch_sparse


class GINConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return MLP((1 + eps) * x + out)

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)
```

Playing around with the new `SparseTensor` format is straightforward since all of our GNNs work with it out-of-the-box. To convert the `edge_index` format to the newly introduced `SparseTensor` format, you can make use of the `torch_geometric.transforms.ToSparseTensor` transform:

```python
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

dataset = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())
data = dataset[0]
>>> Data(adj_t=[2708, 2708, nnz=10556], x=[2708, 1433], y=[2708], ...)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)


model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return float(loss)


for epoch in range(1, 201):
    loss = train(data)
```

All code remains the same as before, except for the data transform via `T.ToSparseTensor()`. As an additional advantage, `MessagePassing` implementations that utilize the `SparseTensor` class are deterministic on the GPU since aggregations no longer rely on atomic operations.

Notably, the GNN layer execution slightly changes in case GNNs incorporate single or multi-dimensional edge information `edge_weight` or `edge_attr` into their message passing formulation, respectively. In particular, it is now expected that these attributes are directly added as values to the `SparseTensor` object. Instead of calling the GNN as

```python
conv = GMMConv(16, 32, dim=3)
out = conv(x, edge_index, edge_attr)
```

we now execute our GNN operator as

```python
conv = GMMConv(16, 32, dim=3)
adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr)
out = conv(x, adj.t())
```

> **Note**
>
> Since this feature is still experimental, some operations, e.g., graph pooling methods, may still require you to input the `edge_index` format. You can convert `adj_t` back to `(edge_index, edge_attr)` via:

```python
row, col, edge_attr = adj_t.t().coo()
edge_index = torch.stack([row, col], dim=0)
```

Please let us know what you think of `SparseTensor`, how we can improve it, and whenever you encounter any unexpected behavior.
