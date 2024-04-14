# Tutorial

If you're new to Eins, you're in the right place!

Virtually everything Eins can do happens through a single import:

```py
from eins import EinsOp
```

Let's go through some common deep learning operations as implemented in Eins, learning about what
differentiates Eins from other einsum libraries along the way.

## Introduction

`EinsOp` represents a function on tensors without any concrete inputs. (It's like a module in
PyTorch or JAX.)

```py
matmul = EinsOp('a b, b c -> a c')
```

When you construct this operation, Eins essentially compiles a high-level program. This has two
benefits over having a single function that takes in the expression and the inputs:

- The `EinsOp` is an object you can inspect, serialize, or manipulate however you like. This opens
  the door for higher-level summaries of a program.
- It's common to apply a single operation to many inputs of the same shape. Any work Eins does in
  parsing and interpreting your expression won't have to be repeated if you define the operation
  once.

### Generalized Matrix Multiplication

When you want to apply your operation, all you need to do is call it:

```py title="Matrix multiplication"
# returns x @ y
matmul = EinsOp('a b, b c -> a c')
matmul(x, y)
```

Eins should support any implementation of the [Array
API](https://data-apis.org/array-api/latest/index.html), but it specifically promises support for
NumPy, PyTorch, and JAX. One of the best things about using Eins is that it frees you from having to
remember the subtle differences in nomenclature and behavior between libraries.

### Batched 1x1 Convolution

Eins shines when things get a little more interesting. Consider a batched 1x1 convolution that
linearly maps channels in a BHWC image array to new channels.

```py title="1×1 Convolution"
EinsOp('''
batch height width chan_in, chan_in chan_out ->
batch height width chan_out
''')(images, kernel)
```

Just like normal Python code, using descriptive identifiers helps the reader understand what's going
on. Eins allows arbitrary whitespace between commas and `->`.

## Reshaping
Let's say we have images in the BHWC format like above, and we want to stack the rows into a single axis. To notate a
flattened axis, we use `(height width)`:

```py title="Flatten inner axes"
# equivalent to ims.reshape(batch, height * width, channels)
EinsOp('''batch height width channels ->
          batch (height width) channels''')(ims)
```

You can think of the parentheses as essentially flattening.

## Fancy Reshaping

Here's something you won't find in other libraries: let's say we have a batch of square images that
we've flattened using something like the above, and we want to reverse it. Just tell Eins that the
two axes are the same size, and it'll figure out what to do. You can either do that explicitly, by
using `=`, or implicitly, by repeating an axis name within a single tensor.

```py title="Unflatten batch of square images"
EinsOp('b (h w=h) c -> b h w c')(ims)
EinsOp('b (h h) c -> b h h c')(ims)
```
Eins defaults to matching up duplicate axes in the same order they appear. If you want to transpose
after the unflatten, you need to use the explicit syntax.

Eins also understands explicit constants:

```py title="Unflattening 3D points"
EinsOp('b (n 3) -> b n 3')(pts)
```

### Strided Convolution/Patch Encoder

The patch encoder in a vision transformer is a specific kind of strided convolution, breaking the
image into p × p squares and then linearly embedding each of them. Despite the complexity, Eins can
figure everything out without any explicit shape information:

```py title="Patch encoder (ViT)"
EinsOp('''batch (num_patch p) (num_patch p) chan_in,
(p p chan_in) chan_out ->
batch (num_patch num_patch) chan_out''')(images, kernel)
```

Eins knows `chan_in` from `images`, can use that plus the knowledge that the patches are square to
deduce `p`, and then can figure out `num_patch` from there.

The constraint system Eins supports saves your fingers from having to type out redundant information
and saves your brain time debugging subtle logic errors.

## Specifying Shapes

Despite these efforts to figure out what you want, sometimes there are multiple potential values for
a symbol, and Eins needs your help to figure out which one to do. Literal integer constants are
succinct, but they lose some of the flexibility and readability of named axes. You can get the best
of both worlds by passing in explicit axis mappings, using either the `symbol_values` argument or
the `=` sign:

```py title="Unflatten non-square axes"
EinsOp('b (h w) c -> b h w c', symbol_values={'h': 4, 'w': 16})(ims)
EinsOp('b (h=4 w=16) c -> b h w c')(ims)
```

You only need one of `h` and `w` specified, because Eins can deduce the other one, but sometimes
it's nice to have an extra check that the input you give Eins is the shape you think it is.

## Advanced Shape Manipulation with Sum Axes

An axis of shape `(h w)` has h × w values, and in fact Eins will let you write that as `h*w` if you prefer. Eins also
supports `h+w`, which generalizes the notion of concatenation.

### Concatenation

```py title="Concatenate inputs along second axis"
EinsOp('b n1, b n2 -> b n1+n2')(x, y)
```

### Splitting

You can also have `+` signs in the input, which lets you slice and dice arrays in a readable way:

```py title="Split along axis"
EinsOp('5+a b -> a b')(x)
```

??? example "Advanced Example: PCA"

    PCA, more specifically [Singular value decomposition](https://www.wikiwand.com/en/Singular_value_decomposition) is
    one real-world example of when you might need this. Let's say you have three arrays U, S, V of
    shapes M × R, R, and R × N, which is what `np.linalg.svd` would return. You want to approximate the
    combined product, of shape M × N, by taking only the first 5 values along the R axis:

    ```py title="Truncated SVD"
    u, s, v = np.linalg.svd(np.random.randn(8, 7))
    truncated_rank = 2
    op = EinsOp('m t+_1, t+_2, t+_3 n -> m n', symbol_values={'t': truncated_rank})
    op(u, s, v)
    ```

    Because `t` is shared across the different inputs, Eins uses that part of the split to make the output. When in doubt
    about how to interpret something, there's no harm in breaking it up, but this showcases the deduction ability Eins has.


## Beyond Einsum

Deep learning would not be a very exciting field if the only things you did were matrix
multiplication and rearranging elements in memory. Luckily, Eins supports a lot more, and moreover
it does so without long docs pages of different single-purpose functions to learn.


!!! note
    This tutorial won't cover everything about this part of Eins: consult [Advanced Eins](in-depth.md)
    if you want a more in-depth tour of this area of the library's functionality.

To understand how Eins represents computation, let's think about how matrix multiplication works. A
matrix multiplication of two matrices M and N with shapes A × B and B × C can be broken down as
such:

$$
(MN)_{ac} = \sum_{b=1}^{B} (M_{ab} \cdot N_{bc})
$$

We first combine M and N together by a broadcasted elementwise multiplication, lining up the two
matrices along the B axis. Then, we eliminate the B axis by summing over all of its values. If you
prefer Python, here's code expressing this idea:

```py title="Matrix multiplication as broadcasted product and sum over axis"
# reshape AB to a x b x 1 and reshape BC to 1 x b x c
# then they broadcast to a x b x c
ABC = np.multiply(AB[:, :, None], BC[None, :, :])
# sum over B axis
AC = ABC.sum(axis=1)
# AC is equivalent to AB @ BC
```

We can generalize matrix multiplication by replacing `.multiply` and `.sum` with other functions that have
the same signature. Using NumPy as an example, if you go through their API you'll find a couple
functions that can be subsituted for the ones above:

| Default  | Alternatives | Signature |
|----------|--------------|-----------|
| `np.multiply`  | `np.add`, `np.minimum`, `np.maximum` | `f(Array, Array) -> Array`, with all array shapes equal |
| `np.sum` | `np.prod`, `np.min`, `np.max` | `f(Array, axis: int) -> Array`, removing the axis |

Eins calls the first kind of function a **combination** and the second kind a **reduction**. The
defaults are `combine='multiply'` and `reduce='sum'`, which is why we haven't needed them for matrix
multiplication.

### Adding a Bias Term

Let's say we just applied a linear layer to get outputs of shape `batch dim`. We can apply a
broadcasted sum with a bias parameter of shape `dim` by supplying the `combine` argument:

```py title="Bias Term"
EinsOp('batch dim, dim -> batch dim', combine='add')(linear, bias)
```

This would happen automatically by broadcasting, but when broadcasting doesn't work it's often quite
error-prone to manually line up shapes.

### Pairwise Distance

Let's say we have batches of `d`-dimensional points and we want to compute the pairwise distance between points in the same batch:

```py title="Batched Pairwise Distance"
EinsOp('batch n1 d, batch n2 d -> batch n1 n2',
       combine='add', reduce='l2_norm')(pts1, -pts2)
```

We're still using addition. Eins does not promise any particular ordering of inputs, so using
combinations that aren't commutative and associative can lead to surprising problems. Negating one
of the inputs is a more reliable way of computing the difference. (For some reason, it's also often
faster!)

That would give us the array of vectors between points, of shape `batch n1 n2 d`. We want a shape of
`batch n1 n2`, so we have to eliminate the `d` axis.

We do this by computing the Euclidean norm: the square root of the sum of the squares of the values
along the axis. This is called the $L_2$ norm, hence the name.

### Batched Custom Loss

The literals that Eins accepts are documented properly in the type system, so you should get a handy
autocomplete for a name like `l2_norm`. The time will come when one of those options isn't
appropriate, however. Eins supports various ways of supplying custom functions.

One solution is to simply pass in your own function. Combinations should have two positional
arguments and output an array of the same size, and custom reductions should take in a single
positional argument and either `axis` or `dim` as keyword arguments.

```py title="Average Huber Loss"
from torch.nn.functional import huber_loss

EinsOp('batch out_features, batch out_features -> batch',
       combine=huber_loss, reduce='mean')(y, y_hat)
```

## Composable Nonlinear Math

There's often a better way than passing in a custom function. We've only discussed operations that
work on the level of shapes: either taking two arrays and combining them, or taking a single array
and reducing an axis. There are two other kinds of operations Eins defines that don't modify the
shape of the output:

- **Elementwise functions** are basically just functions from real numbers to real numbers that you
  can batch arbitrarily. Examples are `np.sin`, `np.abs`, and `np.exp`. They have the signature
  `f(Array) -> Array`.
- **Transformations** use an axis, but don't eliminate it. Examples are `np.sort`, `np.flip`,
  `np.roll`, and normalization. They have the signature `f(Array, axis: int) -> Array`.

Eins implements a library of these functions to go along with combinations and reductions. Combining
them lets you make new functions that are easy to reason about and framework-agnostic. Passing in a
sequence of operations applies them right-to-left, like functions. For example, although Eins
already has `'logaddexp'`, you could instead pass in `combine=('log', 'add', 'exp')`.

For combinations, you can pass in any number of elementwise operations, sandwiching a single
combination operation. For reductions, we do know an axis, so transformations are allowed on top of
elementwise operations and reductions. Once you reduce the axis, it's gone, so a transformation or
reduction can't come after the first reduction.

### Approximate Numerical Equality

It's hard to give an example for "a common function Eins doesn't support", because the hope is that
Eins supports what you're likely to use! So, even if a little contrived, let's say you're interested
in finding whether two arrays are all equal within an epsilon. We could (and probably should) supply
a custom function, but we can instead implement it using elementwise functions: rounding answers so
only values above 1 show up.

```py title="Batched Approximate Numerical Equality"
eps = 1e-6
eps_apart = EinsOp('batch dim, batch dim -> batch',
                    combine=('round', 'abs', 'add'),
                    reduce='max')
eps_apart(x / eps, y / eps) == 0
```

### Softmax with Labeled Axes

Elementwise functions aren't something you generally want to use outside of composition with other
functions in Eins. If you know that you're using a JAX array, `jnp.sin` is going to be just as clean
and readable as anything Eins would provide. Transformations distinguish an axis, however, so the
machinery Eins has for working with shapes makes those transformations easier to understand. Eins
lets you write out transformations in a clearer way than frameworks that use indexed axes.

Consider the transformer attention operation. The [Annotated
Transformer](https://nlp.seas.harvard.edu/annotated-transformer/#encoder-and-decoder-stacks) gives
this equation and this code for the softmax in attention:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} \right) V $$

```py
def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, value), p_attn
```

It's legitimately not clear what axis `scores.softmax(dim=-1)` means in this context, and it's not
indicated in the equation either. The Eins version clearly indicates what dimension we're applying
softmax to:

```py title="Softmax with Labeled Axes"
p_attn = EinsOp('batch q_seq k_seq heads', transform={'k_seq': 'softmax'})(scores)
```

In many applications of attention, `q_seq` and `k_seq` are the same size: there would be no
indication that you were performing softmax incorrectly until things stopped working well.

When you give `EinsOp` an expression without an arrow, it considers your input as a single array,
and leaves the shape unchanged. You can pass in a `transform` mapping to indicate how you want to
transform an axis. Eins does not guarantee an order of transformations, so if you do use multiple
transformations make sure they're commutative.

### Typed Functions

Passing in strings is quick and doesn't require polluting the namespace, but it's not always easy to
know what Eins allows you to write. For the price of a few more imports, you can use something
closer to strong typing than string typing:

```py
from eins import ElementwiseOps as E
from eins import Reductions as R
from eins import Transformations as T
from eins import Combinations as C
```

Feel free to use other names, but the analogy to `import torch.nn.functional as F` is a good one:
these are all simple namespaces with autocomplete-friendly interfaces for accessing functions. The
different functions are all dataclasses that can be serialized and inspected easily.

These namespaces support some customizations that can't be done through strings alone. For example,
we can use softmax with temperature as `T.Softmax(temperature=2)`. The `l2_norm` we saw above can
actually be any power, so we could do Manhattan distance through `l1_norm` or even `2.4_norm`. At
that point, however, you're probably better off writing `R.PowerNorm(power=2.4)`.

In addition, because these namespaces are typed, Eins knows what you want to do with them, and so
these let you provide custom functions in more situations. Eins will only accept raw functions if
they're the only input, because otherwise it's unclear what the signature is.

## Next Steps

That's everything you need to get good use out of Eins! Feel free to check out [Advanced
Eins](in-depth.md) if you want to learn more.