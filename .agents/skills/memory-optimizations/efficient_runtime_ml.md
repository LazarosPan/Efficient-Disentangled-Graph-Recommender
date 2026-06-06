Efficient Memory Management Strategies (https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-6-advanced-ml-runtime-systems/runtime-memory-management)

Efficient memory management is fundamental to the performance of any ML runtime system. Machine learning models frequently operate on large tensors, requiring substantial memory allocations. Furthermore, the dynamic nature of some models and the intermediate activations generated during inference lead to frequent allocation and deallocation requests. Naive reliance on general-purpose allocators like malloc or cudaMalloc within performance-critical loops introduces significant overhead and potential fragmentation, severely impacting execution speed. Therefore, specialized memory management strategies are indispensable.
The Challenge of ML Memory Allocation

ML workloads present unique memory management challenges:

    Large Allocations: Tensors representing model weights, activations, and inputs/outputs can consume gigabytes of memory.
    Frequent Allocation/Deallocation: Intermediate tensors within a computation graph often have short lifetimes, leading to high churn.
    Performance Sensitivity: Allocation/deallocation latency directly adds to the end-to-end model execution time. System call overhead associated with standard allocators is often unacceptable.
    Heterogeneous Memory: Systems often involve distinct memory spaces (e.g., CPU DRAM, GPU High Bandwidth Memory (HBM)) with different capacities, bandwidths, and access characteristics. Data movement between these spaces is costly.
    Dynamic Shapes: The exact size of tensors might not be known until runtime, complicating static memory planning.

Arena Allocators and Memory Pooling

The most prevalent technique to mitigate allocation overhead in ML runtimes is the use of arena allocators, also known as memory pools. The core idea is straightforward:

    Pre-allocation: The runtime pre-allocates one or more large contiguous blocks of memory (the "arena") from the underlying system (e.g., using cudaMalloc for GPU memory or mmap/VirtualAlloc for CPU memory) during initialization or ahead of executing a specific subgraph.
    Sub-allocation: Subsequent requests for tensor memory are satisfied by carving out chunks from this arena. This involves lightweight bookkeeping within the runtime, avoiding expensive system calls.
    Deallocation: When a tensor's memory is no longer needed, its corresponding chunk within the arena is marked as free, making it available for future requests. The actual memory remains allocated to the arena until the arena itself is destroyed or reset.

Allocation Requests Allocator Logic Arena (Large Pre-allocated Block) Free Tensor A (Used) Free Tensor B (Used) Tensor C (Used) Free (Internal Fragmentation) Request(Tensor A) Request(Tensor B) Arena Allocator Request(Tensor C) Allocates A Allocates B Allocates C Marks B as Free Deallocate(Tensor B)

    View of an arena allocator servicing requests by sub-allocating from a pre-allocated memory block and managing free space.

Benefits:

    Reduced Overhead: Eliminates most system call overhead for individual allocations/deallocations.
    Improved Locality: Allocations within an arena are often physically contiguous or closer in virtual address space, potentially improving cache performance (though large tensor sizes often dominate locality effects).
    Controlled Fragmentation: While internal fragmentation (unused space within an allocated chunk) can occur, external fragmentation (unusable free space between allocated chunks) is managed within the arena boundary.

Implementation Strategies:

    First-Fit/Next-Fit: Simple strategies for finding the next available block. Can lead to fragmentation over time.
    Best-Fit: Finds the smallest free block that fits the request. Can reduce fragmentation but requires searching the free list.
    Segregated Free Lists (Segregated Fits): Maintain separate free lists for different size classes. Allocation requests for a specific size check the corresponding list first. This is highly effective for workloads with common tensor sizes, significantly speeding up allocation and reducing fragmentation.
    Buddy Allocators: Manage memory in power-of-two sized blocks, simplifying merging of freed blocks but potentially causing higher internal fragmentation.

The choice of strategy depends on the expected allocation patterns, memory constraints, and performance goals. For dynamic shapes, arenas might need resizing, or multiple arenas with different growth strategies might be employed.
Static vs. Dynamic Allocation Revisited

As discussed in Chapter 3 (Graph-Level Optimizations), static memory planning analyzes the computation graph ahead-of-time to determine tensor lifetimes and identify opportunities for buffer sharing and reuse. This minimizes the peak memory footprint. However, static planning relies on knowing tensor shapes upfront.

When dynamic shapes are present, the runtime memory manager must handle allocations whose sizes are determined during execution. Even with static planning for the known parts of the graph, the dynamic portions rely heavily on efficient runtime allocation. Often, a hybrid approach is used: static planning optimizes as much as possible, and a dynamic arena allocator handles the rest, including potential overallocation based on heuristics or runtime feedback to accommodate dynamic sizes.
Explicit Memory Reuse and Liveness Tracking

Beyond the reuse provided by arena allocators returning freed blocks, runtimes can implement more aggressive explicit memory reuse. This requires tracking the liveness of each tensor buffer: knowing precisely when the data in a buffer is no longer needed by any subsequent operation.

Once a buffer is identified as "dead," the runtime can immediately alias it for a new allocation request, even before the corresponding operation that produced it has fully completed (provided synchronization ensures correctness). This requires careful integration with the runtime's execution scheduler (discussed later) to manage dependencies correctly. Liveness information, often computed by the compiler, is passed to the runtime to guide these decisions.
Optimizing Host-Device Transfers: Pinned Memory

Transferring data between CPU (host) and GPU (device) memory is a common bottleneck. Standard host memory allocated via malloc is typically pageable, meaning the operating system can move its physical location. For Direct Memory Access (DMA) engines used by GPUs to achieve high bandwidth transfers, the physical address must be fixed.

Therefore, initiating a transfer from pageable memory often involves an intermediate step: the GPU driver copies the data from the pageable source buffer to a temporary pinned (or page-locked) buffer in host RAM, whose physical address is fixed. The DMA engine then transfers data from this pinned buffer to the GPU. This extra copy adds latency and consumes bandwidth.
Pageable Memory Transfer Pinned Memory Transfer CPU App Buffer (Pageable) Driver Staging Buffer (Pinned) 1. Driver Copy (CPU) GPU Memory 2. DMA Transfer CPU App Buffer (Pinned) GPU Memory 1. DMA Transfer

    Comparison of data transfer paths using pageable vs. pinned host memory. Pinned memory allows direct DMA, eliminating the staging copy.

ML runtimes optimize this by allocating host-side buffers that will participate in GPU transfers directly as pinned memory (e.g., using cudaMallocHost or cudaHostAlloc).

Trade-offs:

    Performance: Significantly faster host-to-device and device-to-host transfers due to direct DMA access. Essential for overlapping computation and communication.
    Resource Consumption: Pinned memory is a limited system resource. Over-allocating pinned memory can negatively impact overall system performance by reducing the amount of pageable memory available to the OS and other applications.
    Allocation Overhead: Allocating pinned memory can sometimes be slightly slower than allocating standard pageable memory.

Runtimes must carefully manage pinned memory allocation, often using dedicated arenas for pinned buffers and allocating it judiciously only where transfer performance is important.
Unified Memory Systems

Unified Memory (UM) aims to simplify programming for heterogeneous systems by providing a single, coherent virtual address space accessible by both the CPU and GPU. Programmers allocate memory (e.g., using cudaMallocManaged) once, and pointers can be dereferenced from either processor.

The underlying system (GPU driver, OS, and hardware) manages data migration between physical CPU DRAM and GPU HBM automatically, typically on-demand based on page faults.

Advantages:

    Simplified Programming: Eliminates the need for explicit memory allocation in separate spaces and manual data transfers (cudaMemcpy).
    Potential for Oversubscription: Allows applications to allocate more memory than physically available on the GPU, with the system paging data in and out.

Disadvantages:

    Migration Overhead: Automatic migration triggered by page faults introduces latency. Performance can be unpredictable if access patterns cause frequent back-and-forth migrations ("thrashing").
    Control Granularity: Developers have less explicit control over data placement and movement compared to manual management. Performance tuning often involves using hints (e.g., cudaMemAdvise) to guide the driver's migration decisions or prefetching data (cudaMemPrefetchAsync).
    Hardware/Driver Dependency: Performance characteristics vary significantly across GPU generations and driver versions.

While UM simplifies development, high-performance ML runtimes often still prefer explicit memory management (using arenas for cudaMalloc and cudaMallocHost) combined with asynchronous memory copies (cudaMemcpyAsync) scheduled alongside computation kernels. This provides maximum control over data placement and movement, which is often necessary to achieve peak performance, although UM can be a viable alternative in scenarios where development simplicity is prioritized or for specific access patterns where automatic migration performs well.
Advanced Allocator Design

Building high-performance memory managers for ML runtimes involves several key factors:

    Thread Safety: Runtimes often use multiple threads for scheduling operations or managing data transfers. The memory allocator must be thread-safe, typically achieved through locking mechanisms or thread-local arenas. Lock contention can become a bottleneck, motivating designs with finer-grained locking or lock-free approaches.
    NUMA Awareness: On multi-socket CPU systems, memory access latency depends on Non-Uniform Memory Access (NUMA) domains. NUMA-aware runtimes allocate CPU memory on the node closest to the executing thread or the connected device (e.g., GPU) to minimize access latency.
    Device Affinity: Allocations should be placed on the specific device (e.g., GPU 0 vs. GPU 1) where the computation will occur, minimizing expensive cross-device communication.
    Interaction with Asynchronous Execution: When freeing memory in an asynchronous runtime, the manager must ensure that all operations using that memory have completed. This often involves synchronization primitives like events or fences, delaying the actual reuse of the memory block until it's safe.
    Debugging and Profiling: Allocators should include mechanisms for detecting memory leaks, buffer overflows, and tracking memory usage patterns to aid in debugging and performance analysis.

In summary, efficient memory management is a foundation of high-performance ML runtime systems. Techniques like arena allocation, memory pinning, careful reuse based on liveness, and potentially using unified memory, are essential tools. The optimal strategy often involves a combination of these techniques, carefully tuned based on the specific ML models, hardware platform, and performance requirements.