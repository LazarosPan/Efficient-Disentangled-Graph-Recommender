# torch_geometric.profile

GNN profiling and performance analysis utilities.

| Function/Class | Description |
|--------|-------------|
| `profileit` | A decorator to facilitate profiling a function, e.g., obtaining training runtime and memory statistics. Returns a GPUStats if device is xpu or CUDAStats if device is cuda. |
| `timeit` | A context decorator to facilitate timing a function, e.g., obtaining the runtime of a specific model on a specific dataset. |
| `get_stats_summary` | Creates a summary of collected runtime and memory statistics. |
| `trace_handler` | Internal handler for trace collection. |
| `print_time_total` | Internal function to print total timing information. |
| `rename_profile_file` | Internal function to rename profile output files. |
| `torch_profile` | PyTorch native profiler wrapper with Chrome trace export. |
| `xpu_profile` | Intel XPU profiler wrapper with Chrome trace export. |
| `count_parameters` | Counts the trainable parameters in a torch.nn.Module. |
| `get_model_size` | Gets the actual disk size of a torch.nn.Module in bytes. |
| `get_data_size` | Gets the theoretical memory usage of a torch_geometric.data.Data object in bytes. |
| `get_cpu_memory_from_gc` | Returns the used CPU memory in bytes from Python garbage collector. |
| `get_gpu_memory_from_gc` | Returns the used GPU memory in bytes from Python garbage collector. |
| `get_gpu_memory_from_nvidia_smi` | Returns free and used GPU memory from nvidia-smi. |
| `get_gpu_memory_from_ipex` | Returns XPU memory statistics. |
| `benchmark` | Benchmarks a list of functions with the same arguments. |
| `nvtxit` | Enables NVTX profiling for a function. |

## Core Profiling Functions

### profileit(device: str)[source]

A decorator to facilitate profiling a function, e.g., obtaining training runtime and memory statistics of a specific model on a specific dataset. Returns a `GPUStats` if device is xpu or extended object `CUDAStats`, if device is cuda.

**Parameters:**

- `device` (str) – Target device for profiling. Options are: `cuda` and `xpu`.

**Example:**

```python
@profileit("cuda")
def train(model, optimizer, x, edge_index, y):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return float(loss)

loss, stats = train(model, x, edge_index, y)
```

---

### class timeit(log: bool = True, avg_time_divisor: int = 0)[source]

A context decorator to facilitate timing a function, e.g., obtaining the runtime of a specific model on a specific dataset.

**Parameters:**

- `log` (bool, optional) – If set to False, will not log any runtime to the console. (default: `True`)
- `avg_time_divisor` (int, optional) – If set to a value greater than 1, will divide the total time by this value. Useful for calculating the average of runtimes within a for-loop. (default: `0`)

**Methods:**

- `reset()` – Prints the duration and resets current timer.

**Example:**

```python
@torch.no_grad()
def test(model, x, edge_index):
    return model(x, edge_index)

with timeit() as t:
    z = test(model, x, edge_index)
time = t.duration
```

---

### get_stats_summary(stats_list: Union[List[GPUStats], List[CUDAStats]]) → Union[GPUStatsSummary, CUDAStatsSummary][source]

Creates a summary of collected runtime and memory statistics. Returns a `GPUStatsSummary` if list of `GPUStats` was passed, otherwise (list of `CUDAStats` was passed), returns a `CUDAStatsSummary`.

**Parameters:**

- `stats_list` (Union[List[GPUStats], List[CUDAStats]]) – A list of GPUStats or CUDAStats objects, as returned by `profileit()`.

**Return type:**

- Union[GPUStatsSummary, CUDAStatsSummary]

---

## Memory and Size Functions

### count_parameters(model: Module) → int[source]

Given a `torch.nn.Module`, count its trainable parameters.

**Parameters:**

- `model` (torch.nn.Module) – The model.

**Return type:**

- int

---

### get_model_size(model: Module) → int[source]

Given a `torch.nn.Module`, get its actual disk size in bytes.

**Parameters:**

- `model` (torch.nn.Module) – The model.

**Return type:**

- int

---

### get_data_size(data: BaseData) → int[source]

Given a `torch_geometric.data.Data` object, get its theoretical memory usage in bytes.

**Parameters:**

- `data` (torch_geometric.data.Data or torch_geometric.data.HeteroData) – The Data or HeteroData graph object.

**Return type:**

- int

---

### get_cpu_memory_from_gc() → int[source]

Returns the used CPU memory in bytes, as reported by the Python garbage collector.

**Return type:**

- int

---

### get_gpu_memory_from_gc(device: int = 0) → int[source]

Returns the used GPU memory in bytes, as reported by the Python garbage collector.

**Parameters:**

- `device` (int, optional) – The GPU device identifier. (default: `0`)

**Return type:**

- int

---

### get_gpu_memory_from_nvidia_smi(device: int = 0, digits: int = 2) → Tuple[float, float][source]

Returns the free and used GPU memory in megabytes, as reported by nvidia-smi.

**Note:** nvidia-smi will generally overestimate the amount of memory used by the actual program.

**Parameters:**

- `device` (int, optional) – The GPU device identifier. (default: `0`)
- `digits` (int, optional) – The number of decimals to use for megabytes. (default: `2`)

**Return type:**

- Tuple[float, float]

---

### get_gpu_memory_from_ipex(device: int = 0, digits: int = 2) → Tuple[float, float, float][source]

Returns the XPU memory statistics.

**Parameters:**

- `device` (int, optional) – The XPU device identifier. (default: `0`)
- `digits` (int, optional) – The number of decimals to use for megabytes. (default: `2`)

**Return type:**

- Tuple[float, float, float]

---

## Benchmarking Functions

### benchmark(funcs: List[Callable], args: Union[Tuple[Any], List[Tuple[Any]]], num_steps: int, func_names: Optional[List[str]] = None, num_warmups: int = 10, backward: bool = False, per_step: bool = False, progress_bar: bool = False)[source]

Benchmark a list of functions `funcs` that receive the same set of arguments `args`.

**Parameters:**

- `funcs` (List[Callable]) – The list of functions to benchmark.
- `args` (Tuple[Any] or List[Tuple[Any]]) – The arguments to pass to the functions. Can be a list of arguments for each function in case their headers differ. Alternatively, can pass functions that generate arguments on-the-fly.
- `num_steps` (int) – The number of steps to run the benchmark.
- `func_names` (List[str], optional) – The names of the functions. If not given, will try to infer the name from the function itself. (default: `None`)
- `num_warmups` (int, optional) – The number of warmup steps. (default: `10`)
- `backward` (bool, optional) – If set to True, will benchmark both forward and backward passes. (default: `False`)
- `per_step` (bool, optional) – If set to True, will report runtimes per step. (default: `False`)
- `progress_bar` (bool, optional) – If set to True, will print a progress bar during benchmarking. (default: `False`)

---

## NVTX Profiling

### nvtxit(name: Optional[str] = None, n_warmups: int = 0, n_iters: Optional[int] = None)[source]

Enables NVTX profiling for a function.

**Parameters:**

- `name` (Optional[str], optional) – Name to give the reference frame for the function being wrapped. Defaults to the name of the function in code.
- `n_warmups` (int, optional) – Number of iterations to call that function before starting. Defaults to 0.
- `n_iters` (Optional[int], optional) – Number of iterations of that function to record. Defaults to all of them.
