# Performance Tuning Guide

Created On: Sep 21, 2020 | Last Updated: Jul 09, 2025 | Last Verified: Nov 05, 2024

Author: Szymon Migacz

Performance Tuning Guide is a set of optimizations and best practices which can accelerate training and inference of deep learning models in PyTorch. Presented techniques often can be implemented by changing only a few lines of code and can be applied to a wide range of deep learning models across all domains.

## What You Will Learn

- General optimization techniques for PyTorch models
- CPU-specific performance optimizations
- GPU acceleration strategies
- Distributed training optimizations

## Prerequisites

- PyTorch 2.0 or later
- Python 3.8 or later
- CUDA-capable GPU (recommended for GPU optimizations)
- Linux, macOS, or Windows operating system

## Overview

Performance optimization is crucial for efficient deep learning model training and inference. This tutorial covers a comprehensive set of techniques to accelerate PyTorch workloads across different hardware configurations and use cases.

## General Optimizations

```python
import torch
import torchvision
```

### Enable Asynchronous Data Loading and Augmentation

`torch.utils.data.DataLoader` supports asynchronous data loading and data augmentation in separate worker subprocesses. The default setting for `DataLoader` is `num_workers=0`, which means that the data loading is synchronous and done in the main process. As a result the main training process has to wait for the data to be available to continue the execution.

Setting `num_workers > 0` enables asynchronous data loading and overlap between the training and data loading. `num_workers` should be tuned depending on the workload, CPU, GPU, and location of training data.

`DataLoader` accepts a `pin_memory` argument, which defaults to `False`. When using a GPU it's better to set `pin_memory=True`; this instructs `DataLoader` to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.

### Disable Gradient Calculation for Validation or Inference

PyTorch saves intermediate buffers from all operations which involve tensors that require gradients. Typically gradients are not needed for validation or inference. The `torch.no_grad()` context manager can be applied to disable gradient calculation within a specified block of code; this accelerates execution and reduces the amount of required memory. `torch.no_grad()` can also be used as a function decorator.

### Disable Bias for Convolutions Directly Followed by a Batch Norm

`torch.nn.Conv2d()` has a `bias` parameter which defaults to `True` and the same is true for `Conv1d` and `Conv3d`.

If an `nn.Conv2d` layer is directly followed by an `nn.BatchNorm2d` layer, then the bias in the convolution is not needed. Instead, use `nn.Conv2d(..., bias=False, ...)`. Bias is not needed because in the first step BatchNorm subtracts the mean, which effectively cancels out the effect of bias.

This is also applicable to 1D and 3D convolutions as long as BatchNorm or another normalization layer normalizes on the same dimension as convolution bias.

Models available from `torchvision` already implement this optimization.

### Use `parameter.grad = None` Instead of `model.zero_grad()` or `optimizer.zero_grad()`

Instead of calling:

```python
model.zero_grad()
# or
optimizer.zero_grad()
```

to zero out gradients, use the following method instead:

```python
for param in model.parameters():
    param.grad = None
```

The second code snippet does not zero the memory of each individual parameter. Also, the subsequent backward pass uses assignment instead of addition to store gradients, which reduces the number of memory operations.

Setting gradients to `None` has a slightly different numerical behavior than setting them to zero. For more details, refer to the documentation.

Alternatively, call `model.zero_grad(set_to_none=True)` or `optimizer.zero_grad(set_to_none=True)`.

### Fuse Operations

Pointwise operations such as elementwise addition, multiplication, and math functions like `sin()`, `cos()`, `sigmoid()`, and similar operators can be combined into a single kernel. This fusion helps reduce memory access and kernel launch times. Typically, pointwise operations are memory-bound; PyTorch eager mode initiates a separate kernel for each operation, which involves loading data from memory, executing the operation, and writing the results back to memory.

By using a fused operator, only one kernel is launched for multiple pointwise operations, and data is loaded and stored just once. This efficiency is particularly beneficial for activation functions, optimizers, and custom RNN cells.

PyTorch 2 introduces a compile mode facilitated by TorchInductor, an underlying compiler that automatically fuses kernels. TorchInductor extends its capabilities beyond simple element-wise operations, enabling advanced fusion of eligible pointwise and reduction operations for optimized performance.

In the simplest case, fusion can be enabled by applying the `torch.compile` decorator to the function definition, for example:

```python
@torch.compile
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
```

Refer to Introduction to `torch.compile` for more advanced use cases.

### Enable `channels_last` Memory Format for Computer Vision Models

PyTorch supports `channels_last` memory format for convolutional networks. This format is meant to be used in conjunction with AMP to further accelerate convolutional neural networks with Tensor Cores.

Support for `channels_last` is experimental, but it is expected to work for standard computer vision models such as ResNet-50 and SSD. To convert models to `channels_last` format, follow Channels Last Memory Format Tutorial. The tutorial includes a section on converting existing models.

### Checkpoint Intermediate Buffers

Buffer checkpointing is a technique to mitigate the memory capacity burden of model training. Instead of storing inputs of all layers to compute upstream gradients in backward propagation, it stores the inputs of a few layers and the others are recomputed during backward pass. The reduced memory requirements enable increasing the batch size, which can improve utilization.

Checkpointing targets should be selected carefully. The best targets are not large layer outputs with small recomputation cost. Example target layers are activation functions such as `ReLU`, `Sigmoid`, and `Tanh`, plus up-sampling, down-sampling, and matrix-vector operations with small accumulation depth.

PyTorch supports a native `torch.utils.checkpoint` API to automatically perform checkpointing and recomputation.

### Disable Debugging APIs

Many PyTorch APIs are intended for debugging and should be disabled for regular training runs:

- anomaly detection: `torch.autograd.detect_anomaly` or `torch.autograd.set_detect_anomaly(True)`
- profiler related: `torch.autograd.profiler.emit_nvtx`, `torch.autograd.profiler.profile`
- autograd gradcheck: `torch.autograd.gradcheck` or `torch.autograd.gradgradcheck`

## CPU Specific Optimizations

### Utilize Non-Uniform Memory Access (NUMA) Controls

NUMA, or non-uniform memory access, is a memory layout design used in data center machines meant to take advantage of locality of memory in multi-socket machines with multiple memory controllers and blocks. Generally speaking, all deep learning workloads, training or inference, get better performance without accessing hardware resources across NUMA nodes. Thus, inference can be run with multiple instances, each instance on one socket, to raise throughput. For training tasks on a single node, distributed training is recommended to make each training process run on one socket.

In general cases the following command executes a PyTorch script on cores on the `N`th node only, and avoids cross-socket memory access to reduce memory access overhead:

```bash
numactl --cpunodebind=N --membind=N python <pytorch_script>
```

More detailed descriptions can be found here.

### Utilize OpenMP

OpenMP is utilized to bring better performance for parallel computation tasks. `OMP_NUM_THREADS` is the easiest switch that can be used to accelerate computations. It determines the number of threads used for OpenMP computations. CPU affinity settings control how workloads are distributed over multiple cores. This affects communication overhead, cache-line invalidation overhead, or page thrashing, so proper affinity settings bring performance benefits. `GOMP_CPU_AFFINITY` or `KMP_AFFINITY` determine how to bind OpenMP threads to physical processing units. Detailed information can be found here.

With the following command, PyTorch runs the task on `N` OpenMP threads:

```bash
export OMP_NUM_THREADS=N
```

Typically, the following environment variables are used for CPU affinity with the GNU OpenMP implementation. `OMP_PROC_BIND` specifies whether threads may be moved between processors. Setting it to `CLOSE` keeps OpenMP threads close to the primary thread in contiguous place partitions. `OMP_SCHEDULE` determines how OpenMP threads are scheduled. `GOMP_CPU_AFFINITY` binds threads to specific CPUs. An important tuning parameter is core pinning, which prevents the threads from migrating between multiple CPUs, enhancing data locality and minimizing inter-core communication.

```bash
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="N-M"
```

### Intel OpenMP Runtime Library (`libiomp`)

By default, PyTorch uses GNU OpenMP (`GNU libgomp`) for parallel computation. On Intel platforms, Intel OpenMP Runtime Library (`libiomp`) provides OpenMP API specification support. It sometimes brings more performance benefits compared to `libgomp`. Using the `LD_PRELOAD` environment variable can switch the OpenMP library to `libiomp`:

```bash
export LD_PRELOAD=<path>/libiomp5.so:$LD_PRELOAD
```

Similar to CPU affinity settings in GNU OpenMP, environment variables are provided in `libiomp` to control CPU affinity settings. `KMP_AFFINITY` binds OpenMP threads to physical processing units. `KMP_BLOCKTIME` sets the time, in milliseconds, that a thread should wait after completing the execution of a parallel region before sleeping. In most cases, setting `KMP_BLOCKTIME` to `1` or `0` yields good performance. The following commands show common settings with Intel OpenMP Runtime Library:

```bash
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
```

### Switch Memory Allocator

For deep learning workloads, Jemalloc or TCMalloc can get better performance by reusing memory more effectively than the default `malloc` function. Jemalloc is a general-purpose `malloc` implementation that emphasizes fragmentation avoidance and scalable concurrency support. TCMalloc also features optimizations to speed up program execution. One of them is holding memory in caches to speed up access of commonly used objects. Holding such caches even after deallocation also helps avoid costly system calls if such memory is later reallocated. Use the `LD_PRELOAD` environment variable to take advantage of one of them.

```bash
export LD_PRELOAD=<jemalloc.so/tcmalloc.so>:$LD_PRELOAD
```

### Train a Model on CPU with PyTorch `DistributedDataParallel` (DDP) Functionality

For small-scale models or memory-bound models, such as DLRM, training on CPU is also a good choice. On a machine with multiple sockets, distributed training brings efficient hardware-resource usage to accelerate the training process. `torch-ccl`, optimized with Intel oneCCL for efficient distributed deep learning training and collectives such as `allreduce`, `allgather`, and `alltoall`, implements the PyTorch C10D `ProcessGroup` API and can be dynamically loaded as an external process group. Alongside optimizations implemented in the PyTorch DDP module, `torch-ccl` accelerates communication operations and also features simultaneous computation-communication functionality.

## GPU Specific Optimizations

### Enable Tensor Cores

Tensor cores are specialized hardware designed to compute matrix-matrix multiplication operations, primarily utilized in deep learning and AI workloads. Tensor cores have specific precision requirements which can be adjusted manually or via the Automatic Mixed Precision API.

In particular, tensor operations take advantage of lower precision workloads, which can be controlled via `torch.set_float32_matmul_precision`. The default format is set to `'highest'`, which utilizes the tensor data type. However, PyTorch offers alternative precision settings: `'high'` and `'medium'`. These options prioritize computational speed over numerical precision.

### Use CUDA Graphs

When using a GPU, work first must be launched from the CPU and in some cases the context switch between CPU and GPU can lead to poor resource utilization. CUDA graphs are a way to keep computation within the GPU without paying the extra cost of kernel launches and host synchronization.

It can be enabled using:

```python
torch.compile(m, "reduce-overhead")
# or
torch.compile(m, "max-autotune")
```

Support for CUDA graphs is still in development, and its usage can incur increased device memory consumption. Some models might not compile.

### Enable cuDNN Auto-Tuner

NVIDIA cuDNN supports many algorithms to compute a convolution. The autotuner runs a short benchmark and selects the kernel with the best performance on a given hardware for a given input size.

For convolutional networks, which are the currently supported case, enable the cuDNN autotuner before launching the training loop by setting:

```python
torch.backends.cudnn.benchmark = True
```

- The autotuner decisions may be non-deterministic; different algorithms may be selected for different runs. For more details, see PyTorch reproducibility guidance.
- In some rare cases, such as highly variable input sizes, it is better to run convolutional networks with the autotuner disabled to avoid the overhead associated with algorithm selection for each input size.

### Avoid Unnecessary CPU-GPU Synchronization

Avoid unnecessary synchronizations to let the CPU run ahead of the accelerator as much as possible and make sure that the accelerator work queue contains many operations.

When possible, avoid operations which require synchronization, for example:

- `print(cuda_tensor)`
- `cuda_tensor.item()`
- memory copies such as `tensor.cuda()`, `cuda_tensor.cpu()`, and equivalent `tensor.to(device)` calls
- `cuda_tensor.nonzero()`
- Python control flow which depends on results of operations performed on CUDA tensors, for example `if (cuda_tensor != 0).all()`

### Create Tensors Directly on the Target Device

Instead of calling `torch.rand(size).cuda()` to generate a random tensor, produce the output directly on the target device: `torch.rand(size, device='cuda')`.

This is applicable to all functions which create new tensors and accept a `device` argument, such as `torch.rand()`, `torch.zeros()`, `torch.full()`, and similar functions.

### Use Mixed Precision and AMP

Mixed precision leverages Tensor Cores and offers up to 3x overall speedup on Volta and newer GPU architectures. To use Tensor Cores, AMP should be enabled and matrix or tensor dimensions should satisfy requirements for kernels that use Tensor Cores.

To use Tensor Cores:

- Set sizes to multiples of 8 to map onto Tensor Core dimensions.
- See Deep Learning Performance Documentation for more details and guidelines specific to each layer type.
- If a layer size is derived from other parameters rather than fixed, it can still be explicitly padded, for example vocabulary size in NLP models.
- Enable AMP.
- Native PyTorch AMP is available through its documentation, examples, and tutorial material.

### Preallocate Memory in Case of Variable Input Length

Models for speech recognition or NLP are often trained on input tensors with variable sequence length. Variable length can be problematic for the PyTorch caching allocator and can lead to reduced performance or unexpected out-of-memory errors. If a batch with a short sequence length is followed by another batch with longer sequence length, then PyTorch is forced to release intermediate buffers from the previous iteration and reallocate new buffers. This process is time consuming and causes fragmentation in the caching allocator, which may result in out-of-memory errors.

A typical solution is to implement preallocation. It consists of the following steps:

1. Generate a usually random batch of inputs with maximum sequence length, either corresponding to the maximum length in the training dataset or to some predefined threshold.
2. Execute a forward and a backward pass with the generated batch, but do not execute an optimizer or a learning rate scheduler. This step preallocates buffers of maximum size, which can be reused in subsequent training iterations.
3. Zero out gradients.
4. Proceed to regular training.

## Distributed Optimizations

### Use Efficient Data-Parallel Backend

PyTorch has two ways to implement data-parallel training:

- `torch.nn.DataParallel`
- `torch.nn.parallel.DistributedDataParallel`

`DistributedDataParallel` offers much better performance and scaling to multiple GPUs. For more information, refer to the relevant section of CUDA Best Practices from the PyTorch documentation.

### Skip Unnecessary All-Reduce if Training with `DistributedDataParallel` and Gradient Accumulation

By default, `torch.nn.parallel.DistributedDataParallel` executes gradient all-reduce after every backward pass to compute the average gradient over all workers participating in the training. If training uses gradient accumulation over `N` steps, then all-reduce is not necessary after every training step. It is only required after the last call to `backward`, just before execution of the optimizer.

`DistributedDataParallel` provides the `no_sync()` context manager, which disables gradient all-reduce for a particular iteration. `no_sync()` should be applied to the first `N - 1` iterations of gradient accumulation. The last iteration should follow the default execution and perform the required gradient all-reduce.

### Match the Order of Layers in Constructors and During Execution if Using `DistributedDataParallel(find_unused_parameters=True)`

`torch.nn.parallel.DistributedDataParallel` with `find_unused_parameters=True` uses the order of layers and parameters from model constructors to build buckets for DistributedDataParallel gradient all-reduce. DistributedDataParallel overlaps all-reduce with the backward pass. All-reduce for a particular bucket is asynchronously triggered only when all gradients for parameters in a given bucket are available.

To maximize the amount of overlap, the order in model constructors should roughly match the order during execution. If the order does not match, then all-reduce for the entire bucket waits for the gradient which is the last to arrive. This may reduce the overlap between backward pass and all-reduce, which slows down training.

`DistributedDataParallel` with `find_unused_parameters=False`, which is the default setting, relies on automatic bucket formation based on order of operations encountered during the backward pass. With `find_unused_parameters=False` it is not necessary to reorder layers or parameters to achieve optimal performance.

### Load-Balance Workload in a Distributed Setting

Load imbalance typically occurs for models processing sequential data such as speech recognition, translation, and language models. If one device receives a batch of data with sequence length longer than sequence lengths for the remaining devices, then all devices wait for the worker which finishes last. The backward pass functions as an implicit synchronization point in a distributed setting with the DistributedDataParallel backend.

There are multiple ways to solve the load-balancing problem. The core idea is to distribute workload over all workers as uniformly as possible within each global batch. For example, Transformer solves imbalance by forming batches with approximately constant numbers of tokens and variable numbers of sequences in a batch. Other models solve imbalance by bucketing samples with similar sequence length or even by sorting the dataset by sequence length.

## Conclusion

This tutorial covered a comprehensive set of performance optimization techniques for PyTorch models. The key takeaways include:

- General optimizations: enable async data loading, disable gradients for inference, fuse operations with `torch.compile`, and use efficient memory formats.
- CPU optimizations: leverage NUMA controls, optimize OpenMP settings, and use efficient memory allocators.
- GPU optimizations: enable Tensor Cores, use CUDA graphs, enable cuDNN autotuner, and implement mixed precision training.
- Distributed optimizations: use DistributedDataParallel, optimize gradient synchronization, and balance workloads across devices.

Many of these optimizations can be applied with minimal code changes and provide significant performance improvements across a wide range of deep learning models.

## Further Reading

- PyTorch Performance Tuning Documentation
- CUDA Best Practices
- Distributed Training Documentation
- Mixed Precision Training
- `torch.compile` Tutorial
