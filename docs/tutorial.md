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

```py
# returns x @ y
matmul = EinsOp('a b, b c -> a c')
matmul(x, y)
```

### Batched 1x1 Convolution

Eins shines when things get a little more interesting. Consider a batched 1x1 convolution that
linearly maps channels in a BHWC image array to new channels.

```py
# 1x1 convolution
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

```py
# flatten height and width together
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

```py
# unflatten batch of square images
EinsOp('b (h w=h) c -> b h w c')(ims)
EinsOp('b (h h) c -> b h h c')(ims)
```
Eins defaults to matching up duplicate axes in the same order they appear. If you want to transpose
after the unflatten, you need to use the explicit syntax.

Eins also understands explicit constants:

```py
# unflatten list of 3D points
EinsOp('b (n 3) -> b n 3')(pts)
```

The constraint system Eins supports saves you keyboard strokes and saves your future self time
debugging subtle logic errors.

## Specifying Shapes

Literal integer constants are succinct, but they lose some of the flexibility and readability of
named axes. You can get the best of both worlds by passing in explicit axis mappings, using either
the `symbol_values` argument or the `=` sign:

```py
EinsOp('b (h w) c -> b h w c', symbol_values={'h': 4, 'w': 16})(ims)
EinsOp('b (h=4 w=16) c -> b h w c')(ims)
```

You only need one of `h` and `w` specified, because Eins can deduce the other one, but sometimes
it's nice to have an extra check that the input you give Eins is the shape you think it is.

## Concatenation

An axis of shape `(h w)` has h × w values, and in fact Eins will let you write that as `h*w` if you prefer. Eins also
supports `h+w`, which generalizes the notion of concatenation.

```py
# concatenate inputs along second axis
EinsOp('b n1, b n2 -> b n1+n2')(x, y)
```

## Splitting

You can also have `+` signs in the input, which lets you slice and dice arrays in a readable way.
[Singular value decomposition](https://www.wikiwand.com/en/Singular_value_decomposition) decomposes
an M × N matrix into a product of three matrices: M × M, M × N, and N × N. Computing PCA requires
selecting only the first R rows/columns, instead using M × R, R × R, and R × N.

```py
# Truncated SVD:
rank = 5
EinsOp(f'm=r1+u1 r1, r1 r2=r1, r2=n+u2 n', symbol_values={'rank': rank})
```

## Strided Convolution/Patch Encoder

The patch encoder in a vision transformer is a specific kind of strided convolution, breaking the image into p × p
squares and then linearly embedding each of them. There's no need to break up the image before embedding the patches,
because Eins can figure it out:

```py
# Patch encoder
EinsOp('''batch (num_patch p) (num_patch p) chan_in, (p p chan_in) chan_out ->
batch (num_patch num_patch) chan_out''')(images, kernel)
```

Eins figures out `chan_in` from the last axis of the images, uses that to deduce `p` from the first axis of the kernel
(because we're using square patches, there's only one valid value of `p`), and then knows how to break up the height and
width into patches.

## Pairwise Distance

Let's say we have batches of `d`-dimensional points and we want to compute the pairwise distance between points in the same batch:

```py
# Batched pairwise distance
EinsOp('batch n1 d, batch n2 d -> batch n1 n2',
        combine='add', reduce='l2_norm')(pts1, -pts2)
```

`combine='add'` means we're adding together the elements: when we pass in `pts1` and `-pts2`, this computes a
broadcasted version of `pts1 - pts2`. We then eliminate the `d` axis by computing the Euclidean norm, given by
`reduce='l2_norm'`.

