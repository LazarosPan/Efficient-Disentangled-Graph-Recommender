# Compiled Graph Neural Networks

`torch.compile()` is the latest method to speed up your PyTorch code in `torch >= 2.0.0`. `torch.compile()` makes PyTorch code run faster by JIT-compiling it into optimized kernels, all while requiring minimal code changes.

Under the hood, `torch.compile()` captures PyTorch programs via TorchDynamo, canonicalizes over 2,000 PyTorch operators via PrimTorch, and finally generates fast code out of it across multiple accelerators and backends via the deep learning compiler TorchInductor.

> **Note**
>
> See here for a general tutorial on how to leverage `torch.compile()`, and here for a description of its interface.

In this tutorial, we show how to optimize your custom PyG model via `torch.compile()`.

> **Note**
>
> From PyG 2.5 (and onwards), `torch.compile()` is now fully compatible with all PyG GNN layers. If you are on an earlier version of PyG, consider using `torch_geometric.compile()` instead.

## Basic Usage

Once you have a PyG model defined, simply wrap it with `torch.compile()` to obtain its optimized version:

```python
import torch
from torch_geometric.nn import GraphSAGE

model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels)
model = model.to(device)

model = torch.compile(model)
```

And execute it as usual:

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root, name="Cora")
data = dataset[0].to(device)

out = model(data.x, data.edge_index)
```

## Maximizing Performance

The `torch.compile()` method provides two important arguments to be aware of:

Most of the mini-batches observed in PyG are dynamic by nature, meaning that their shape varies across different mini-batches. For these scenarios, we can enforce dynamic shape tracing in PyTorch via the `dynamic=True` argument:

```python
torch.compile(model, dynamic=True)
```

With this, PyTorch will up-front attempt to generate a kernel that is as dynamic as possible to avoid recompilations when sizes change across mini-batches. Note that when `dynamic` is set to `False`, PyTorch will never generate dynamic kernels, and thus only work when graph sizes are guaranteed to never change (e.g., in full-batch training on small graphs). By default, `dynamic` is set to `None` in PyTorch `>= 2.1.0`, and PyTorch will automatically detect if dynamism has occurred. Note that support for dynamic shape tracing requires PyTorch `>= 2.1.0` to be installed.

In order to maximize speedup, graph breaks in the compiled model should be limited. We can force compilation to raise an error upon the first graph break encountered by using the `fullgraph=True` argument:

```python
torch.compile(model, fullgraph=True)
```

It is generally a good practice to confirm that your written model does not contain any graph breaks. Importantly, there exist a few operations in PyG that will currently lead to graph breaks (but workarounds exist), e.g.:

- `global_mean_pool()` (and other pooling operators) perform device synchronization in case the batch size `size` is not passed, leading to a graph break.
- `remove_self_loops()` and `add_remaining_self_loops()` mask the given `edge_index`, leading to a device synchronization to compute its final output shape. As such, we recommend augmenting your graph before inputting it into your GNN, e.g., via the `AddSelfLoops` or `GCNNorm` transformations, and setting `add_self_loops=False`/`normalize=False` when initializing layers such as `GCNConv`.

## Example Scripts

We have incorporated multiple examples in `examples/compile` that further show the practical usage of `torch.compile()`:

- Node Classification via `GCN` (`dynamic=False`)
- Graph Classification via `GIN` (`dynamic=True`)

If you notice that `torch.compile()` fails for a certain PyG model, do not hesitate to reach out either on GitHub or Slack. We are very eager to improve `torch.compile()` support across the whole PyG code base.

## Benchmark

`torch.compile()` works fantastically well for many PyG models. Overall, we observe runtime improvements of up to 300%.

Specifically, we benchmark `GCN`, `GraphSAGE`, and `GIN` and compare runtimes obtained from traditional eager mode and `torch.compile()`. We use a synthetic graph with 10,000 nodes and 200,000 edges, and a hidden feature dimensionality of 64. We report runtimes over 500 optimization steps:

| Model | Mode | Forward | Backward | Total | Speedup |
|---|---|---:|---:|---:|---:|
| GCN | Eager | 2.6396s | 2.1697s | 4.8093s | |
| GCN | Compiled | 1.1082s | 0.5896s | 1.6978s | 2.83x |
| GraphSAGE | Eager | 1.6023s | 1.6428s | 3.2451s | |
| GraphSAGE | Compiled | 0.7033s | 0.7465s | 1.4498s | 2.24x |
| GIN | Eager | 1.6701s | 1.6990s | 3.3690s | |
| GIN | Compiled | 0.7320s | 0.7407s | 1.4727s | 2.29x |

To reproduce these results, run:

```python
python test/nn/models/test_basic_gnn.py
```

from the root folder of your checked out PyG repository from GitHub.
