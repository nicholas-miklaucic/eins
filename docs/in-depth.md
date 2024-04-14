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
Combinations are elementwise functions that combine two arrays into a new array of the same shape. The default,
`'multiply'`, multiplies inputs, as in matrix multiplication.

It's much easier to ensure Eins does what you want when these are commutative and associative. Instead of trying to
specify subtraction, use addition and then negate the input you want to subtract. This gives Eins freedom to optimize
your computation.

**Common examples**: `'add'`, `'multiply'`, `'minimum'`, `'maximum'`

### Reductions
Reductions take a single array and an axis and eliminate that axis. The default, `'sum'`, sums over an axis, as in
matrix multiplication.

If you pass in a combination, Eins will essentially apply `functools.reduce` and use that combination to reduce the
axis. In general, however, there are more efficient ways of doing the same thing: a folded `'add'` is just a slower
`'sum'`, and a folded `hypot` is just a slower `l2-norm`.

**Common examples**: `'sum'`, `'prod'`, `'l2_norm'`, `'min'`, `'max'`.

Note the naming conventions, matching NumPy nomenclature. `np.max(arr, axis=0)` computes the max along axis 0,
eliminating it. `np.maximum(arr1, arr2)` is the elementwise maximum between two arrays.

### Elementwise Operations
An elementwise op takes in a single array and returns an array of the same size, applying an operation individually to
each element. Eins doesn't use these explicitly, but you can combine them with combinations or reductions to
ergonomically represent more complex functions.

**Common examples**: `'log'`, `'exp'`, `'tanh'`, `'square'`, `'sqrt'`

### Transformations
Named after the `.transform` method in Pandas, transformations take in a single array and `axis`, like reductions, but
they don't eliminate the axis. For example, `np.sort(arr, axis=0)` is different than `np.sort(arr, axis=1)`, but both
return the same shape.

Just like a folded combination becomes a reduction, a *scanned* or *accumulated* combination becomes a transformation.
Note that the way NumPy and other libraries notate these differs from the idea of a scan. `cumprod`, in Eins, is really
just an alias for `cummultiply`, because Eins uses the combination rather than the reduction. If you have an array with elements `[a, b, c, d]` and an operator like `*`, then Eins computes

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

Eins supports a relatively sophisticated "stringly-typed" input format, as you've seen above. This means you rarely need
any imports beyond `EinsOp`, and plays nicely with any kind of config files or serialization, but it does also make it
harder to know what functions Eins defines or use your own.

If you prefer, you can instead pass in explicit objects: `Combination`, `Reduction`, `ElementwiseOp`, and
`Transformation`. These are each base classes that you can implement yourself, but it's easiest to use the associated
object exported from the base namespace: `Combinations`, `Reductions`, etc. These namespaces provide an autocomplete-friendly way of using these operations.

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