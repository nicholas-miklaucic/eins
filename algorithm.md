# Algorithm

Input:

```python
b (d=(n p) d) c, b p*p*c h, h[k] -> b n n k
```

## Step 1: Shorthand Expansion

```python
b ((n p n p)) c, b (p p c) h, Index[h][h k] -> b n n k
```

- Index array unpacking
- Substituting bindings for their values
- Unpack nested flattens

## Step 2: Input/Output Shape Processing

```python
b ((n p n p)) c, b (p p c) h, Index[k][h k] -> b n n k

n * p = d

i00 = b
i01 = n * p * n * p
i02 = c
i10 = b
i11 = p * p * c
i12 = h
i20 = k
```

- Input shapes will be known at JIT time
- Possibility for meaningful errors at this level: `i01` has to be a perfect square
- Perhaps basic checks can be performed before actual inputs are given?


## Step 3: Solve for Shapes

Consider inputs of shapes `64 100 3, 64 75 8, 15`

Fill in known values until everything is done:

```python
i00 = 64 => b = 64
i02 = 3  => c = 3
i12 = 8  => h = 8
i20 = 15 => h = 15
##############
i11 = 75, c = 3  => p = 5
i01 = 100, p = 5 => n = 2
```

Basically a graph: you can fill in a node if its dependencies are known.

Here, both `i00` and `i10` must match: when a variable has two paths to a solution, compute both and verify

## Step 4: Apply Indexing

```python
b ((n p n p)) c, b (p p c) k -> b n n k
```

Error if index axis appears in output.

## Step 5: Disambiguate Axes

```python
b (n1 p1 n2 p2) c, b (p1 p2 c) k -> b n1 n2 k
```

Start counting again after each array.


## Step 6: Determine Steps

- `b k n1 n2 h` appear in input and output once: batch axes
- `p1 p2 c` appear in input twice, output none: reduce

Both inputs have all reduction axes: otherwise, it's unclear what order we reduce in

But, because the order of combining inputs is known and our reductions commute, we can proceed

```python
Combine(i0, i1, 'product') -> b n1 p1 n2 p2 c k (i2)
Reduce(i2, {p1, p2, c}, 'sum')
```

Reductions should happen in the order that gets rid of the most each time, unless an explicit intermediate is given:

`b (n p n p) c, b (p p c) k -> b n p n p k -> b n n k`