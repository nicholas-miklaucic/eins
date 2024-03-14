# Array Types Thinking

```py
species = top_k(density.mean(axis=1), 5)[1]
top_density = permute_dims(
    take_along_axis(
        permute_dims(density, [0, 2, 1]), data['species'][..., None], axis=1),
    [0, 2, 1])
```

```py
# matrix multiply
a b, b c -> a c, combine = 'mul', reduce = 'sum'

# square matrix multiply
a b=a, b c=a -> a c

# Pairwise combine: two bs in input, one in output
m b, n b -> m n b, combine='sub'

# Pairwise distance: instead of 'sub', probably best to input a negative and do sums
m b, n b -> m n, combine = 'sub', reduce = 'l2'

# Take along axis
a b c, c[a k] -> a b k

# Unpack 3D grid: no need for explicit axis sizes because there is <= 1 solutions for any input (d h w)
b (d h=d w=d) c -> b d h w c

# Equivalent, with assumption that same variables match up in order
b (d d d) c -> b d d d c

# d^3 equivalent to (d d d)
b d^3 c -> b d d d c

# Error, although explicit version is fine: unclear which axis gets preserved
!! b (d d d) c -> b d c

# Upsample 2D image: s^2*c = (s s c). (s dim) would mean the different channels are far apart in the output
batch dim dim s^2*c -> batch (dim s) (dim s) c

# Downsample 2D image
batch (dim s) (dim s) c -> batch dim dim s^2*c

# Version with constant s is brittle, because all 2s are the same, but can work
# maybe allow 4 because 4 = 2*2? probably not?
b (dim*2) (dim*2) c -> batch dim dim 2^2*c

# Equivalent to above, because dim can be done as (dim/s * s)
# Default is this order
batch dim dim c -> batch dim/s dim/s s^2*c

# Flipped, usually incorrect, version
b (s d) (s d) c -> b d d s^2*c

# Version that will work, but why would you do this
# d becomes (d/e)*e
b d d c -> b e e (d/e d/e c)
```

- Reduction: `[a] -> a`
- Combination: `[[a]] -> [a]`
- If there's an index array, that can only combine via indexing other tensors.
- All other tensors with that axis are indexed, then the index tensor is no longer used: the axis is completely replaced
  - `a b c, c[a k], c e f -> a b e f k => a b k, k e f -> a b e f k`
  - Any further use of c is disallowed
- `=` implies constraint without semantic equality: `m b, n=m b -> m n b` means
  pairwise combination, but it's not equivalent to `n m b`: only the shape is
- `()` means reshaping in the normal numpy fashion: `(a b)` means point 1 is a=0, b=1


- some kind of simple wrapper for functions, but people should be pushed towards layers: precompilation, constraint checking early


MLP/layer with weights/bias like EinMix:

```py
# batched, maps 2x2xc_in blocks to c_out
batch h*a=2 w*b=2 c_in -> b h w c_out, weights = (a b c_in) c_out, bias = c_out

batch h*a=2 w*b=2 c_in, (a b c_in) c_out, c_out -> batch h w c_out, combine = 'mul', reduce = 'sum'
```


```py
batch h*a=2 w*b=2 c_in, (a b c_in) c_out -> batch h w c_out (combine = 'mul', reduce = 'sum')
,c_out -> batch h w c_out, (combine = 'sum')
```

- `batch`, `c_out`: appear once in input, once in output, `vmap`
- `h`: appears once in input as inner dim, once in output, `vmap` after initial unpacking
- `a`, `b`, `c_in`: appear twice in input, none in output, so combine => reduce
- we have to solve `(h a) = in1` and `(h b) = in2`: this is only possible if two out of three are known
- because a, b are given, we can deduce output shape, once we have input shape: this is lazy-in


Some way of implementing this to work with argmax? A different kind of reduction?

```py
vecs1 = np.random.randn(10, 3)
vecs2 = np.random.randn(20, 3)
# Which vector in vecs1 is maximally far from its closest vector in vecs2?
eins('a b, c b -> a c -> a -> a[1]', combine='sum', reduce={
    'a': 'argmax',
    'b': 'square,sum,sqrt'
    'c': 'min'
})(vecs1, -vecs2)
```