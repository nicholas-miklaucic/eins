# Cookbook

This is a collection of different things people often do with tensors, implemented in Eins.

```python
from eins import EinsOp
```

## Matrix Multiplication

```python title="Matrix multiplication"
EinsOp("a b, b c -> a c")(mat1, mat2)
```

```python title="Batched matrix multiplication"
EinsOp("batch a b, batch b c -> batch a c")(mat1, mat2)
```

## Dense Feedforward With Bias

```python title="Dense layer: Wx + b"
matmul = EinsOp("batch d_in, d_in d_out -> batch d_out")
add_bias = EinsOp("batch d_out, d_out -> batch d_out", combine="add")

add_bias(matmul(x, weights), bias)
```