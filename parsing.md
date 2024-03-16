# Einop Parsing

This document explains how einop expressions are parsed by `eins`. There is a lot here that is not
in einops: it's worth reading if you're curious how expressions will be interpreted.

## Structure
An `eins` expression is a sequence of arrays, separated by `,`, then an arrow `->`, then an output array.

- Whitespace is allowed around `,` and `->`, including newlines and tabs, but not inside arrays.
- The output can only be a single array: `eins` does not implement the functionality necessary to
  intelligently share results between different computations, so if you need that you should figure
  out how you want to order the computations yourself.
- You can chain `->` to specify chains where the intermediate results are not needed, but each arrow
  after the first must go from a single input to a single output.

An array is determined by its shape and element type. `einops` has no notion of element type, so let me explain:

Most arrays are generic scalars and have no explicit element type: `a b c` means a 3D tensor of
numbers.

`eins` adds the concept of an *index array*, an array of indices into another tensor. These are
written as `a[b c]`: this is an array of shape `b c`, but its values are integers in `[0, a)`. An
example:

```py
# x would be a valid input for 'a b c'
x = np.random.randn(3, 4, 5)
# i would be a valid input for 'a[b c]'
i = np.argmax(x, axis=0)
```

## Dimensions
A shape is a space-separated list of dimensions. There are several ways to specify a dimension.

First, the ones that are familiar from `einops`:
- Constants, like `1` or `2`
- Named dimensions, like `d`, `batch`, or `chan_in`
- Parenthetical groupings of dimensions, to indicate the flattened version of the array without that
  grouping: `(a b c)` is the result from calling `.reshape(-1)` on an array of shape `a b c`
- **TODO** `*batch`: matches any number of consecutive dimensions.

`eins` adds a lot of powerful new functionality to this system:

### Constraints
To express that a dimension is equal to a different expression, you can write `dim=expr`.

```py
# Matrix multiplication, but asserting that the matrices are square.
# Will helpfully error if that is not true.
a b=a, b c=b -> a c
```

The left side of `=` indicates the *meaning* of the axis: the right side indicates the size. The two
axes named `b` are contracted, but `c` remains untouched because, semantically, the rows and columns
of a square matrix are different.

### Duplicate Axis Names
Let's say we want to flatten a square matrix, asserting that it's square. With constraints, this would be

```py
m n=m -> (m n)
```

In this case, the `=` is a little verbose. `eins` instead supports a shorthand which matches up
duplicate symbols on the left and right sides of the expression:

```py
# equivalent to above
m m -> (m m)
```

`eins` matches the `m`s on the left with their respective `m` on the right. If you want the output
to be transposed and then flattened, or dislike this ambiguity, you can always use the explicit
version with `=`.

Note that the right side has to have either *no* duplicate axis names or *exactly as many* as the
left side. `m m -> m` is too ambiguous to allow.

### Arithmetic Expressions
You can use arithmetic expressions to specify the size of a dimension. Be careful: unlike normal
arithmetic, here the order matters.

A value of `a*b` is equivalent to `(a b)` in `einops`: such a flattened axis has size `a*b`, so the
notation is appropriate.

By analogy, `dim^3` is shorthand for `(dim dim dim)`.

`eins` even supports division!

```py
# Downsample the image by reshaping
batch h w c -> batch h/3 w/4 (3 4 c)
# Equivalent to the following version with multiplication:
batch h=h1*3 w=w1*4 c -> batch h1 w1 (3 4 c)
```

Note that this enforces the standard order for manipulation, but that may not be what you want.
`eins` will error if this division produces a non-integer result.

**TODO** support combining arithmetic and assignment

## Flexible Combination and Reduction
When `einsum` sees an expression like `a b, b c -> a c`, it sees that `b` is present twice in the
input and not at all in the output. So it first *combines* the two arrays, matching along that
dimension, and then *reduces* or *contracts* that axis down to a singleton that is then removed. The
standard `einsum` uses multiplication to combine the input arrays and summation to reduce the array
down to a single value.

This is mostly what you want, but there's no reason not to allow more flexible alternatives. `eins`
does this where `einops` does not, by specifying `combine` and `reduce`. This lets you, for
instance, compute pairwise distances:

```py
vecs1 = np.random.randn(10, 3)
vecs2 = np.random.randn(20, 3)
# Pairwise Euclidean distances between vecs1 and vecs2
eins('a b, c b -> a c', combine='sum', reduce='square,sum,sqrt')(vecs1, -vecs2)
```

`combine` indicates the different input arrays are combined. `reduce` indicates how to reduce an
axis down to a scalar and eliminate it. As this shows, you can additionally sequence elementwise
operations in `reduce`, and the same works for `combine`.

`reduce` can instead be mappings that indicate different functions to use for different axes.

```py
vecs1 = np.random.randn(10, 3)
vecs2 = np.random.randn(20, 3)
# Get the distance from each vector in vecs1 to its closest vector in vecs2
eins('a b, c b -> a c -> a', combine='sum', reduce={
    'b': 'square,sum,sqrt'
    'c': 'min'
})(vecs1, -vecs2)
```

Think of this as a more powerful combination of `einops.einsum` and `einops.reduce`. Because it's a
single operation instead of a chained call, there is one thing to note: the order of reductions
matters. `eins` makes no guarantees at this time about the order it chooses to reduce if there is
ambiguity. So instead of writing `a b, c b -> a`, which could either mean the distance to the
minimum corner of the bounding box of `c` or the minimum distance to vectors in `c`, prefer the
chain.


## Semantics
What does `eins` actually do? Here's a rough sketch:

- Index arrays `c[a b]` are treated as a one-hot array of shape `a b c`: this essentially converts
  indexing into an `einsum` operation. This indexing is applied to every other input that has the
  index axis: every other input with `c` is first converted to a new array. If the inputs have `a`
  or `b`, they're lined up. If they don't, then a new axis is added.
- Arrays with flattened axes are unflattened. This includes the output, if only symbolically.
- Arrays with concatenation are decomposed as two arrays in the input: `a b c+d` is split into its
  parts, like `a b c, a b d`. **TODO** how to support addition in the output?
- Along with the hints given by the user, `eins` tries to solve for all the axis dimensions. If it
  can't, then an error is raised.
- Every input array is reshaped so they all broadcast, including with the output array. This unpacks
  every product axis as well.
- Then, the inputs are combined using the combination operation and reshaped to a strict superset of
  the axes in the output.
- Each axis that does not appear in the output is then reduced using the corresponding reduction
  operation.