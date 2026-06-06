
Will Numba work for my code?

This depends on what your code looks like, if your code is numerically orientated (does a lot of math), uses NumPy a lot and/or has a lot of loops, then Numba is often a good choice. In these examples we’ll apply the most fundamental of Numba’s JIT decorators, @jit, to try and speed up some functions to demonstrate what works well and what does not.

Numba works well on code that looks like this:

from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

print(go_fast(x))

It won’t work very well, if at all, on code that looks like this:

from numba import jit
import pandas as pd

x = {'a': [1, 2, 3], 'b': [20, 30, 40]}

@jit(forceobj=True, looplift=True) # Need to use object mode, try and compile loops!
def use_pandas(a): # Function will not benefit from Numba jit
    df = pd.DataFrame.from_dict(a) # Numba doesn't know about pd.DataFrame
    df += 1                        # Numba doesn't understand what this is
    return df.cov()                # or this!

print(use_pandas(x))

Note that Pandas is not understood by Numba and as a result Numba would simply run this code via the interpreter but with the added cost of the Numba internal overheads!
What is object mode?

The Numba @jit decorator fundamentally operates in two compilation modes, nopython mode and object mode. In the go_fast example above, the @jit decorator defaults to operating in nopython mode. The behaviour of the nopython compilation mode is to essentially compile the decorated function so that it will run entirely without the involvement of the Python interpreter. This is the recommended and best-practice way to use the Numba jit decorator as it leads to the best performance.

Should the compilation in nopython mode fail, Numba can compile using object mode. This achieved through using the forceobj=True key word argument to the @jit decorator (as seen in the use_pandas example above). In this mode Numba will compile the function with the assumption that everything is a Python object and essentially run the code in the interpreter. Specifying looplift=True might gain some performance over pure object mode as Numba will try and compile loops into functions that run in machine code, and it will run the rest of the code in the interpreter. For best performance avoid using object mode mode in general!
How to measure the performance of Numba?

First, recall that Numba has to compile your function for the argument types given before it executes the machine code version of your function. This takes time. However, once the compilation has taken place Numba caches the machine code version of your function for the particular types of arguments presented. If it is called again with the same types, it can reuse the cached version instead of having to compile again.

A really common mistake when measuring performance is to not account for the above behaviour and to time code once with a simple timer that includes the time taken to compile your function in the execution time.

For example:

from numba import jit
import numpy as np
import time

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.perf_counter()
go_fast(x)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
go_fast(x)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))

This, for example prints:

Elapsed (with compilation) = 0.33030009269714355s
Elapsed (after compilation) = 6.67572021484375e-06s

A good way to measure the impact Numba JIT has on your code is to time execution using the timeit module functions; these measure multiple iterations of execution and, as a result, can be made to accommodate for the compilation time in the first execution.

As a side note, if compilation time is an issue, Numba JIT supports on-disk caching of compiled functions and also has an Ahead-Of-Time compilation mode.
How fast is it?

Assuming Numba can operate in nopython mode, or at least compile some loops, it will target compilation to your specific CPU. Speed up varies depending on application but can be one to two orders of magnitude. Numba has a performance guide that covers common options for gaining extra performance.
How does Numba work?

Numba reads the Python bytecode for a decorated function and combines this with information about the types of the input arguments to the function. It analyzes and optimizes your code, and finally uses the LLVM compiler library to generate a machine code version of your function, tailored to your CPU capabilities. This compiled version is then used every time your function is called.
Other things of interest:

Numba has quite a few decorators, we’ve seen @jit, but there’s also:

    @njit - this is an alias for @jit(nopython=True) as it is so commonly used!

    @vectorize - produces NumPy ufunc s (with all the ufunc methods supported). Docs are here.

    @guvectorize - produces NumPy generalized ufunc s. Docs are here.

    @stencil - declare a function as a kernel for a stencil like operation. Docs are here.

    @jitclass - for jit aware classes. Docs are here.

    @cfunc - declare a function for use as a native call back (to be called from C/C++ etc). Docs are here.

    @overload - register your own implementation of a function for use in nopython mode, e.g. @overload(scipy.special.j0). Docs are here.

Extra options available in some decorators:

    parallel = True - enable the automatic parallelization of the function.

    fastmath = True - enable fast-math behaviour for the function.

ctypes/cffi/cython interoperability:

    cffi - The calling of CFFI functions is supported in nopython mode.

    ctypes - The calling of ctypes wrapped functions is supported in nopython mode.

    Cython exported functions are callable.

GPU targets:

Numba can target Nvidia CUDA GPUs. You can write a kernel in pure Python and have Numba handle the computation and data movement (or do this explicitly). Click for Numba documentation on CUDA.
