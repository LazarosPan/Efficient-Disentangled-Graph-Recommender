# TorchScript Support

TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency. If you are unfamiliar with TorchScript, we recommend reading the official *“Introduction to TorchScript”* tutorial first.

## Converting GNN Models

> **Note**
>
> From PyG 2.5 (and onwards), GNN layers are now fully compatible with `torch.jit.script()` without any modification needed. If you are on an earlier version of PyG, consider converting your GNN layers into “jittable” instances first by calling `jittable()`.

Converting your PyG model to a TorchScript program is straightforward and requires only a few code changes. Let’s consider the following model:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


model = GNN(dataset.num_features, dataset.num_classes)
```

The instantiated model can now be directly passed into `torch.jit.script()`:

```python
model = torch.jit.script(model)
```

That is all you need to know on how to convert your PyG models to TorchScript programs. You can have a further look at our JIT examples that show-case how to obtain TorchScript programs for node and graph classification models.

## Creating Jittable GNN Operators

All PyG `MessagePassing` operators are tested to be convertible to a TorchScript program. However, if you want your own GNN module to be compatible with `torch.jit.script()`, you need to account for the following two things:

1. As one would expect, your `forward()` code may need to be adjusted so that it passes the TorchScript compiler requirements, e.g., by adding type notations.
2. You need to tell the `MessagePassing` module the types that you pass to its `propagate()` function. This can be achieved in two different ways.

### Declaring the Type of Propagation Arguments in a Dictionary Called `propagate_type`

```python
from typing import Optional
from torch import Tensor
from torch_geometric.nn import MessagePassing


class MyConv(MessagePassing):
    propagate_type = {'x': Tensor, 'edge_weight': Optional[Tensor]}

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
```

### Declaring the Type of Propagation Arguments as a Comment Inside Your Module

```python
from typing import Optional
from torch import Tensor
from torch_geometric.nn import MessagePassing


class MyConv(MessagePassing):
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: Optional[Tensor])
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
```

If none of these options are given, the `MessagePassing` module will infer the arguments of `propagate()` to be of type `torch.Tensor` (mimicking the default type that TorchScript is inferring for non-annotated arguments).
