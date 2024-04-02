# Tutorial

Essentially everything Eins can do happens through a single import:

```py
from eins import EinsOp
```

Let's see what it can do!

## Generalized Matrix Multiplication (Einstein Summation)

`EinsOp` represents the operation itself: you first create the operation you want, and then execute it with specific inputs.

```py
# returns x @ y
EinsOp('a b, b c -> a c')(x, y)
```

This takes in two matrices of size A × B and B × C and returns their product with shape A × C.

Eins shines in more complex situations. Let's see what a batched 1x1 convolution looks like:

```py
# 1x1 convolution
EinsOp('''batch height width chan_in, chan_in chan_out ->
batch height width chan_out''')(images, kernel)
```

Adding informative names makes it much easier to catch errors and understand what's happening.

## Reshaping
Let's say we have images in the BHWC format like above, and we want to stack the rows into a single axis. To notate a
flattened axis, we use `(height width)`:

```py
# flatten height and width together
# equivalent to ims.reshape(batch, height * width, channels)
EinsOp('''batch height width channels ->
          batch (height width) channels''')(ims)
```

## Fancy Reshaping

Here's something you won't find in other libraries: let's say we have a batch of square images that we've flattened
using something like the above, and we want to reverse it. Just tell Eins that the two axes are the same size, and it'll
figure out how to reshape it. You can either do that explicitly, by using `=`, or implicitly, by repeating an axis name
within a single tensor.

```py
# unflatten batch of square images
EinsOp('b (h w=h) c -> b h w c')(ims)
EinsOp('b (h h) c -> b h h c')(ims)
```
Eins defaults to matching up duplicate axes in the same order they appear. If you want to transpose after the
unflatten, you need to use the explicit syntax.

Eins implements a general system for inferring what you want from your inputs and constraints. This also lets Eins
handle constant sizes properly.

```py
# unflatten list of 3D points
EinsOp('b (n 3) -> b n 3')(ims)
```

## Specifying Shapes

Sometimes, you will need to tell Eins a specific axis. (Often, you can resolve this by doing
whatever you're going to do next and letting Eins infer using that shape.) There are two ways to do
that: through the `symbol_values` argument, or by using `=` with constant values:

```py
EinsOp('b (h w) c -> b h w c', symbol_values={'h': 4, 'w': 16})(ims)
EinsOp('b (h=4 w) c -> b h w c')(ims)
EinsOp('b (h w=16) c -> b h w c')(ims)
```

## Concatenation

An axis of shape `(h w)` has h × w values, and in fact Eins will let you write that as `h*w` if you prefer. Eins also
supports `h+w`, which generalizes the notion of concatenation and stacking.

```py
# concatenate inputs along second axis
EinsOp('b n1, b n2 -> b n1+n2')(x, y)
```

Just like multiplication, you can put sums anywhere in the inputs or outputs and Eins will deal with it correctly.

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

