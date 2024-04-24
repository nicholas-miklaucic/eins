# Advanced Eins

If you're totally new to Eins or `einops` and want to see what the fuss is about, read the
[tutorial](tutorial.md). If you're interested in maximizing the power of Eins, you're in the right
place!

```py
from eins import EinsOp
```

## Beyond Tensor Shapes

Consider an operation `a b, b c -> a c`. This could be matrix multiplication, but it could also be pairwise distances or
many other things. To control the computation that's being performed beyond the shapes of the inputs and output, Eins
defines four kinds of functions that specify what's actually happening:

### Combinations
Combinations are, mathematically, functions that take two scalars and output a scalar. In Eins,
combinations should be vectorized, taking in two arrays of the same shape and returning an array of
that shape. The most common examples are `np.add` and `np.multiply`.

**Common examples**: `'add'`, `'multiply'`, `'minimum'`, `'maximum'`


!!! danger

    Eins assumes that a combination is commutative and associative, and it makes no guarantees about the
    order your arrays are combined. If you supply custom functions, that responsibility is yours.

### Reductions
Reductions are essentially functions that take in a vector of any size and return a scalar, like
`np.sum`. (These are sometimes called aggregations.) In Eins, they're functions that take an array
and an axis and return an array with that axis removed.

If you pass in a combination, Eins will essentially apply `functools.reduce` and use that
combination to reduce the axis. In general, however, there are more efficient ways of doing the same
thing: a folded `'add'` is just a slower `'sum'`, and a folded `hypot` is just a slower `l2-norm`.

**Common examples**: `'sum'`, `'prod'`, `'l2_norm'`, `'min'`, `'max'`.

Note the naming conventions, matching NumPy nomenclature. `np.max(arr, axis=0)` computes the max along axis 0,
eliminating it. `np.maximum(arr1, arr2)` is the elementwise maximum between two arrays.

!!! danger

    If you reduce more than once in a program, Eins assumes you know what you're doing and that the
    operation would be the same either way, like summing over two axes. If you supply a custom function,
    make sure there is only one potential output.



### Elementwise Operations

An elementwise operation should be thought of as a function that takes a scalar and outputs a
scalar. Eins requires that the operation is *vectorized*, so it takes in an array and outputs an
array of the same shape.

**Common examples**: `'log'`, `'exp'`, `'tanh'`, `'square'`, `'sqrt'`

### Transformations

Named after the `.transform` method in Pandas, transformations should be thought of mathematically
as functions that take in a vector of any size and produce a vector of the same size. Think of
sorting or standardization: you need multiple inputs for standardization to make sense, but at the
end you haven't changed the shape of the array.

In Eins, transformations take in a single array and `axis`, like reductions, but they don't
eliminate the axis. For example, `np.sort(arr, axis=0)` is different than `np.sort(arr, axis=1)`,
but both return an array of the same shape as `arr`.

Just like a folded combination becomes a reduction, a *scanned* or *accumulated* combination becomes
a transformation. Note that the way NumPy and other libraries notate these differs from the idea of
a scan. `cumprod`, in Eins, is really just an alias for `cummultiply`, because Eins uses the
combination rather than the reduction. If you have an array with elements `[a, b, c, d]` and an
operator like `*`, then Eins computes

```python
[a, a * b, (a * b) * c, ((a * b) * c) * d]
```

**Common examples**: `'sort'`, `'l2_normalize'`, `'min_max_normalize'`

### Composing Functions

Eins uses `combine` and `reduce` arguments that specify how to combine inputs and how to reduce axes. The point of
elementwise operations and transformations is that they can be composed with combinations and reductions.

Functions are applied right-to-left, matching existing nomenclature and function composition. For example, if
`'logaddexp'` weren't already a supported combination, you could replicate the functionality as `('log', 'add', 'exp')`.
This computes the logarithm of the sum of the exponentials of the inputs.

Similarly, if you wanted to compute root-mean-square error along an axis, you could use
`reduce=('sqrt', 'mean', 'square')`. This is common enough to get its own name: `'l2_norm'`.

### Explicit Function Objects

Eins supports a relatively sophisticated "stringly-typed" input format, as you've seen above. This
means you rarely need any imports beyond `EinsOp`, and you can easily serialize the description of
the operation, but it does also make it harder to know what functions Eins defines or use your own.

If you prefer, you can instead pass in explicit objects: `Combination`, `Reduction`,
`ElementwiseOp`, and `Transformation`. These are each base classes that you can implement yourself,
but it's easiest to use the associated object exported from the base namespace: `Combinations`,
`Reductions`, etc. These namespaces provide an autocomplete-friendly way of using these operations.

Explicit objects are the only way to specify compositions with function syntax. If you pass in a callable to `combine`
or `reduce`, Eins will assume it has the correct signature, but if you pass in `(my_func1, my_func2)` Eins has no way of
knowing what's what. Instead, you can do:

```py title="Batched Kurtosis"
from eins import EinsOp, Reductions as R, ElementwiseOps as E
from scipy.stats import kurtosis
# kurtosis has signature (x, axis=0, ...)

EinsOp('batch sample_size -> batch', reduce=(E.abs, R.from_func(kurtosis)))
```

## Standalone Operator Usage

For backend-agnostic code or simply as a wrapper for functionality Eins implements that isn't available in all libraries, there's no reason you can't just use the above functions outside of an EinsOp context:

```python
from eins import Reductions as R, Transformations as T

# 1.5-norm: somewhere between Manhattan and Euclidean distance
# akin to torch.nn.functional.normalize, but no direct numpy equivalent

data = np.random.randn(128, 64)

R.PowerNorm(1.5)(data, axis=1)

# Normalize so the 1.5-norm is 1: same shape as input
T.PowerNormalize(1.5)(data, axis=1)
```