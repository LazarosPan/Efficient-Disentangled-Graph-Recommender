Creating NumPy universal functions

There are two types of universal functions:

    Those which operate on scalars, these are “universal functions” or ufuncs (see @vectorize below).

    Those which operate on higher dimensional arrays and scalars, these are “generalized universal functions” or gufuncs (@guvectorize below).

The @vectorize decorator

Numba’s vectorize allows Python functions taking scalar input arguments to be used as NumPy ufuncs. Creating a traditional NumPy ufunc is not the most straightforward process and involves writing some C code. Numba makes this easy. Using the vectorize() decorator, Numba can compile a pure Python function into a ufunc that operates over NumPy arrays as fast as traditional ufuncs written in C.

Using vectorize(), you write your function as operating over input scalars, rather than arrays. Numba will generate the surrounding loop (or kernel) allowing efficient iteration over the actual inputs.

The vectorize() decorator has two modes of operation:

    Eager, or decoration-time, compilation: If you pass one or more type signatures to the decorator, you will be building a NumPy universal function (ufunc). The rest of this subsection describes building ufuncs using decoration-time compilation.

    Lazy, or call-time, compilation: When not given any signatures, the decorator will give you a Numba dynamic universal function (DUFunc) that dynamically compiles a new kernel when called with a previously unsupported input type. A later subsection, “Dynamic universal functions”, describes this mode in more depth.

As described above, if you pass a list of signatures to the vectorize() decorator, your function will be compiled into a NumPy ufunc. In the basic case, only one signature will be passed:
from test_vectorize_one_signature of numba/tests/doc_examples/test_examples.py

1from numba import vectorize, float64
2
3@vectorize([float64(float64, float64)])
4def f(x, y):
5    return x + y

If you pass several signatures, beware that you have to pass most specific signatures before least specific ones (e.g., single-precision floats before double-precision floats), otherwise type-based dispatching will not work as expected:
from test_vectorize_multiple_signatures of numba/tests/doc_examples/test_examples.py

1from numba import vectorize, int32, int64, float32, float64
2import numpy as np
3
4@vectorize([int32(int32, int32),
5            int64(int64, int64),
6            float32(float32, float32),
7            float64(float64, float64)])
8def f(x, y):
9    return x + y

The function will work as expected over the specified array types:
from test_vectorize_multiple_signatures of numba/tests/doc_examples/test_examples.py

1a = np.arange(6)
2result = f(a, a)
3# result == array([ 0,  2,  4,  6,  8, 10])

from test_vectorize_multiple_signatures of numba/tests/doc_examples/test_examples.py

1a = np.linspace(0, 1, 6)
2result = f(a, a)
3# Now, result == array([0. , 0.4, 0.8, 1.2, 1.6, 2. ])

but it will fail working on other types:

a = np.linspace(0, 1+1j, 6)
f(a, a)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: ufunc 'ufunc' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

You might ask yourself, “why would I go through this instead of compiling a simple iteration loop using the @jit decorator?”. The answer is that NumPy ufuncs automatically get other features such as reduction, accumulation or broadcasting. Using the example above:
from test_vectorize_multiple_signatures of numba/tests/doc_examples/test_examples.py

 1a = np.arange(12).reshape(3, 4)
 2# a == array([[ 0,  1,  2,  3],
 3#             [ 4,  5,  6,  7],
 4#             [ 8,  9, 10, 11]])
 5
 6result1 = f.reduce(a, axis=0)
 7# result1 == array([12, 15, 18, 21])
 8
 9result2 = f.reduce(a, axis=1)
10# result2 == array([ 6, 22, 38])
11
12result3 = f.accumulate(a)
13# result3 == array([[ 0,  1,  2,  3],
14#                   [ 4,  6,  8, 10],
15#                   [12, 15, 18, 21]])
16
17result4 = f.accumulate(a, axis=1)
18# result3 == array([[ 0,  1,  3,  6],
19#                   [ 4,  9, 15, 22],
20#                   [ 8, 17, 27, 38]])

See also

Standard features of ufuncs (NumPy documentation).

Note

Only the broadcasting and reduce features of ufuncs are supported in compiled code.

The vectorize() decorator supports multiple ufunc targets:

Target
	

Description

cpu
	

Single-threaded CPU

parallel
	

Multi-core CPU

cuda
	

CUDA GPU

Note

This creates an ufunc-like object. See documentation for CUDA ufunc for detail.

A general guideline is to choose different targets for different data sizes and algorithms. The “cpu” target works well for small data sizes (approx. less than 1KB) and low compute intensity algorithms. It has the least amount of overhead. The “parallel” target works well for medium data sizes (approx. less than 1MB). Threading adds a small delay. The “cuda” target works well for big data sizes (approx. greater than 1MB) and high compute intensity algorithms. Transferring memory to and from the GPU adds significant overhead.

Starting in Numba 0.59, the cpu target supports the following attributes and methods in compiled code:

    ufunc.nin

    ufunc.nout

    ufunc.nargs

    ufunc.identity

    ufunc.signature

    ufunc.reduce() (only the first 5 arguments - experimental feature)

The @guvectorize decorator

While vectorize() allows you to write ufuncs that work on one element at a time, the guvectorize() decorator takes the concept one step further and allows you to write ufuncs that will work on an arbitrary number of elements of input arrays, and take and return arrays of differing dimensions. The typical example is a running median or a convolution filter.

Contrary to vectorize() functions, guvectorize() functions don’t return their result value: they take it as an array argument, which must be filled in by the function. This is because the array is actually allocated by NumPy’s dispatch mechanism, which calls into the Numba-generated code.

Similar to vectorize() decorator, guvectorize() also has two modes of operation: Eager, or decoration-time compilation and lazy, or call-time compilation.

Here is a very simple example:
from test_guvectorize of numba/tests/doc_examples/test_examples.py

1from numba import guvectorize, int64
2import numpy as np
3
4@guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
5def g(x, y, res):
6    for i in range(x.shape[0]):
7        res[i] = x[i] + y

The underlying Python function simply adds a given scalar (y) to all elements of a 1-dimension array. What’s more interesting is the declaration. There are two things there:

    the declaration of input and output layouts, in symbolic form: (n),()->(n) tells NumPy that the function takes a n-element one-dimension array, a scalar (symbolically denoted by the empty tuple ()) and returns a n-element one-dimension array;

    the list of supported concrete signatures as per @vectorize; here, as in the above example, we demonstrate int64 arrays.

Note

1D array type can also receive scalar arguments (those with shape ()). In the above example, the second argument also could be declared as int64[:]. In that case, the value must be read by y[0].

We can now check what the compiled ufunc does, over a simple example:
from test_guvectorize of numba/tests/doc_examples/test_examples.py

1a = np.arange(5)
2result = g(a, 2)
3# result == array([2, 3, 4, 5, 6])

The nice thing is that NumPy will automatically dispatch over more complicated inputs, depending on their shapes:
from test_guvectorize of numba/tests/doc_examples/test_examples.py

 1a = np.arange(6).reshape(2, 3)
 2# a == array([[0, 1, 2],
 3#             [3, 4, 5]])
 4
 5result1 = g(a, 10)
 6# result1 == array([[10, 11, 12],
 7#                   [13, 14, 15]])
 8
 9result2 = g(a, np.array([10, 20]))
10g(a, np.array([10, 20]))
11# result2 == array([[10, 11, 12],
12#                   [23, 24, 25]])

Note

Both vectorize() and guvectorize() support passing nopython=True as in the @jit decorator. Use it to ensure the generated code does not fallback to object mode.
Scalar return values

Now suppose we want to return a scalar value from guvectorize(). To do this, we need to:

    in the signatures, declare the scalar return with [:] like a 1-dimensional array (eg. int64[:]),

    in the layout, declare it as (),

    in the implementation, write to the first element (e.g. res[0] = acc).

The following example function computes the sum of the 1-dimensional array (x) plus the scalar (y) and returns it as a scalar:
from test_guvectorize_scalar_return of numba/tests/doc_examples/test_examples.py

1from numba import guvectorize, int64
2import numpy as np
3
4@guvectorize([(int64[:], int64, int64[:])], '(n),()->()')
5def g(x, y, res):
6    acc = 0
7    for i in range(x.shape[0]):
8        acc += x[i] + y
9    res[0] = acc

Now if we apply the wrapped function over the array, we get a scalar value as the output:
from test_guvectorize_scalar_return of numba/tests/doc_examples/test_examples.py

1a = np.arange(5)
2result = g(a, 2)
3# At this point, result == 20.

Overwriting input values

In most cases, writing to inputs may also appear to work - however, this behaviour cannot be relied on. Consider the following example function:
from test_guvectorize_overwrite of numba/tests/doc_examples/test_examples.py

1from numba import guvectorize, float64
2import numpy as np
3
4@guvectorize([(float64[:], float64[:])], '()->()')
5def init_values(invals, outvals):
6    invals[0] = 6.5
7    outvals[0] = 4.2

Calling the init_values function with an array of float64 type results in visible changes to the input:
from test_guvectorize_overwrite of numba/tests/doc_examples/test_examples.py

1invals = np.zeros(shape=(3, 3), dtype=np.float64)
2# invals == array([[6.5, 6.5, 6.5],
3#                  [6.5, 6.5, 6.5],
4#                  [6.5, 6.5, 6.5]])
5
6outvals = init_values(invals)
7# outvals == array([[4.2, 4.2, 4.2],
8#                   [4.2, 4.2, 4.2],
9#                   [4.2, 4.2, 4.2]])

This works because NumPy can pass the input data directly into the init_values function as the data dtype matches that of the declared argument. However, it may also create and pass in a temporary array, in which case changes to the input are lost. For example, this can occur when casting is required. To demonstrate, we can use an array of float32 with the init_values function:
from test_guvectorize_overwrite of numba/tests/doc_examples/test_examples.py

 1invals = np.zeros(shape=(3, 3), dtype=np.float32)
 2# invals == array([[0., 0., 0.],
 3#                  [0., 0., 0.],
 4#                  [0., 0., 0.]], dtype=float32)
 5outvals = init_values(invals)
 6# outvals == array([[4.2, 4.2, 4.2],
 7#                   [4.2, 4.2, 4.2],
 8#                   [4.2, 4.2, 4.2]])
 9print(invals)
10# invals == array([[0., 0., 0.],
11#                  [0., 0., 0.],
12#                  [0., 0., 0.]], dtype=float32)

In this case, there is no change to the invals array because the temporary casted array was mutated instead.

To solve this problem, one needs to tell the GUFunc engine that the invals argument is writable. This can be achieved by passing writable_args=('invals',) (specifying by name), or writable_args=(0,) (specifying by position) to @guvectorize. Now, the code above works as expected:
from test_guvectorize_overwrite of numba/tests/doc_examples/test_examples.py

 1@guvectorize(
 2    [(float64[:], float64[:])],
 3    '()->()',
 4    writable_args=('invals',)
 5)
 6def init_values(invals, outvals):
 7    invals[0] = 6.5
 8    outvals[0] = 4.2
 9
10invals = np.zeros(shape=(3, 3), dtype=np.float32)
11# invals == array([[0., 0., 0.],
12#                  [0., 0., 0.],
13#                  [0., 0., 0.]], dtype=float32)
14outvals = init_values(invals)
15# outvals == array([[4.2, 4.2, 4.2],
16#                   [4.2, 4.2, 4.2],
17#                   [4.2, 4.2, 4.2]])
18print(invals)
19# invals == array([[6.5, 6.5, 6.5],
20#                  [6.5, 6.5, 6.5],
21#                  [6.5, 6.5, 6.5]], dtype=float32)

Dynamic universal functions

As described above, if you do not pass any signatures to the vectorize() decorator, your Python function will be used to build a dynamic universal function, or DUFunc. For example:
from test_vectorize_dynamic of numba/tests/doc_examples/test_examples.py

1from numba import vectorize
2
3@vectorize
4def f(x, y):
5    return x * y

The resulting f() is a DUFunc instance that starts with no supported input types. As you make calls to f(), Numba generates new kernels whenever you pass a previously unsupported input type. Given the example above, the following set of interpreter interactions illustrate how dynamic compilation works:

f
<numba._DUFunc 'f'>
f.ufunc
<ufunc 'f'>
f.ufunc.types
[]

The example above shows that DUFunc instances are not ufuncs. Rather than subclass ufunc’s, DUFunc instances work by keeping a ufunc member, and then delegating ufunc property reads and method calls to this member (also known as type aggregation). When we look at the initial types supported by the ufunc, we can verify there are none.

Let’s try to make a call to f():
from test_vectorize_dynamic of numba/tests/doc_examples/test_examples.py

1result = f(3,4)
2# result == 12
3
4print(f.types)
5# ['ll->l']

If this was a normal NumPy ufunc, we would have seen an exception complaining that the ufunc couldn’t handle the input types. When we call f() with integer arguments, not only do we receive an answer, but we can verify that Numba created a loop supporting C long integers.

We can add additional loops by calling f() with different inputs:
from test_vectorize_dynamic of numba/tests/doc_examples/test_examples.py

1result = f(1.,2.)
2# result == 2.0
3
4print(f.types)
5# ['ll->l', 'dd->d']

We can now verify that Numba added a second loop for dealing with floating-point inputs, "dd->d".

If we mix input types to f(), we can verify that NumPy ufunc casting rules are still in effect:
from test_vectorize_dynamic of numba/tests/doc_examples/test_examples.py

1result = f(1,2.)
2# result == 2.0
3
4print(f.types)
5# ['ll->l', 'dd->d']

This example demonstrates that calling f() with mixed types caused NumPy to select the floating-point loop, and cast the integer argument to a floating-point value. Thus, Numba did not create a special "dl->d" kernel.

This DUFunc behavior leads us to a point similar to the warning given above in “The @vectorize decorator” subsection, but instead of signature declaration order in the decorator, call order matters. If we had passed in floating-point arguments first, any calls with integer arguments would be cast to double-precision floating-point values. For example:
from test_vectorize_dynamic of numba/tests/doc_examples/test_examples.py

 1@vectorize
 2def g(a, b):
 3    return a / b
 4
 5print(g(2.,3.))
 6# 0.66666666666666663
 7
 8print(g(2,3))
 9# 0.66666666666666663
10
11print(g.types)
12# ['dd->d']

If you require precise support for various type signatures, you should specify them in the vectorize() decorator, and not rely on dynamic compilation.
Dynamic generalized universal functions

Similar to a dynamic universal function, if you do not specify any types to the guvectorize() decorator, your Python function will be used to build a dynamic generalized universal function, or GUFunc. For example:
from test_guvectorize_dynamic of numba/tests/doc_examples/test_examples.py

1from numba import guvectorize
2import numpy as np
3
4@guvectorize('(n),()->(n)')
5def g(x, y, res):
6    for i in range(x.shape[0]):
7        res[i] = x[i] + y

We can verify the resulting function g() is a GUFunc instance that starts with no supported input types. For instance:

g
<numba._GUFunc 'g'>
g.ufunc
<ufunc 'g'>
g.ufunc.types
[]

Similar to a DUFunc, as one make calls to g(), numba generates new kernels for previously unsupported input types. The following set of interpreter interactions will illustrate how dynamic compilation works for a GUFunc:
from test_guvectorize_dynamic of numba/tests/doc_examples/test_examples.py

1x = np.arange(5, dtype=np.int64)
2y = 10
3res = np.zeros_like(x)
4g(x, y, res)
5# res == array([10, 11, 12, 13, 14])
6print(g.types)
7# ['ll->l']

If this was a normal guvectorize() function, we would have seen an exception complaining that the ufunc could not handle the given input types. When we call g() with the input arguments, numba creates a new loop for the input types.

We can add additional loops by calling g() with new arguments:
from test_guvectorize_dynamic of numba/tests/doc_examples/test_examples.py

1x = np.arange(5, dtype=np.double)
2y = 2.2
3res = np.zeros_like(x)
4g(x, y, res)
5# res == array([2.2, 3.2, 4.2, 5.2, 6.2])

We can now verify that Numba added a second loop for dealing with floating-point inputs, "dd->d".
from test_guvectorize_dynamic of numba/tests/doc_examples/test_examples.py

1print(g.types) # shorthand for g.ufunc.types
2# ['ll->l', 'dd->d']

One can also verify that NumPy ufunc casting rules are working as expected:
from test_guvectorize_dynamic of numba/tests/doc_examples/test_examples.py

1x = np.arange(5, dtype=np.int64)
2y = 2
3res = np.zeros_like(x)
4g(x, y, res)
5print(res)
6# res == array([2, 3, 4, 5, 6])

If you need precise support for various type signatures, you should not rely on dynamic compilation and instead, specify the types them as first argument in the guvectorize() decorator.

@guvectorize functions can also be called from jitted ones. For instance:
from test_guvectorize_jit of numba/tests/doc_examples/test_examples.py

 1import numpy as np
 2
 3from numba import jit, guvectorize
 4
 5@guvectorize('(n)->(n)')
 6def copy(x, res):
 7    for i in range(x.shape[0]):
 8        res[i] = x[i]
 9
10@jit(nopython=True)
11def jit_fn(x, res):
12    copy(x, res)

Warning

Broadcasting is not supported yet. Calling a guvectorize function in a scenario where broadcasting is needed may result in incorrect behavior. Numba will attempt to detect those cases and raise an exception.
from test_guvectorize_jit of numba/tests/doc_examples/test_examples.py

 1import numpy as np
 2from numba import jit, guvectorize
 3
 4@guvectorize('(n)->(n)')
 5def copy(x, res):
 6    for i in range(x.shape[0]):
 7        res[i] = x[i]
 8
 9@jit(nopython=True)
10def jit_fn(x, res):
11    copy(x, res)
12
13x = np.ones((1, 5))
14res = np.empty((5,))
15with self.assertRaises(ValueError) as raises:
16    jit_fn(x, res)
