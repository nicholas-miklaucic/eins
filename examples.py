# type: ignore

from typing import cast

from eins import EinsOp
from eins import Reductions as R  # noqa: N817
from eins import Transformations as T
from eins.common_types import Array
from eins.namespaces import ElementwiseOps

# Set this to 'jax', 'numpy', or 'torch'
BACKEND = 'jax'

if BACKEND == 'jax':
    import jax.numpy as jnp

    xp = jnp
    arr = jnp.array
elif BACKEND == 'numpy':
    import numpy as np

    xp = np
    arr = np.array
elif BACKEND == 'torch':
    import torch

    xp = torch
    arr = torch.tensor
else:
    raise ValueError


def randn(*shape) -> Array:
    if BACKEND == 'jax':
        import jax.numpy as jnp
        import numpy as np

        # TODO debug why bfloat16 doesn't work here
        return jnp.asarray(np.random.randn(*shape))
    elif BACKEND == 'numpy':
        import numpy as np

        return np.array(np.random.randn(*shape))
    elif BACKEND == 'torch':
        import torch

        return cast(Array, torch.randn(*shape))

    raise ValueError


EPSILON = 1e-3


def test_close(a: Array, b: Array):
    if a.shape != b.shape:
        msg = f'{a.shape} != {b.shape}'
        raise ValueError(msg)
    diffs = xp.mean(xp.abs(a - b))  # type: ignore
    assert diffs.max() < EPSILON, f'{a.shape} != {b.shape}, {R.mean(a)}, {R.mean(b)}, {diffs.max()}'  # noqa: S101


# Softmax
x = randn(5, 4)

op = EinsOp('a b', transform={'b': ('softmax', ElementwiseOps.from_func(lambda x: x + 2))})
y = op(x)
# print(op)
y2 = T.Softmax(temperature=1)(x, axis=1)
test_close(y, y2)

# Splitting
x, y = randn(3, 4), randn(5, 4)
z2 = xp.concatenate((x, y), axis=0)

x1 = EinsOp('α+β γ -> α γ', symbol_values={'α': x.shape[0]})(z2)
test_close(x1, x)

# Concatenation

# z1 = EinsOp('a c, b c -> a+b c')(x, y)
# test_close(z1, z2)

# Simple matrix multiplication
x = randn(32, 64)
y = randn(64, 16)
op = EinsOp('a b, b c -> a c')
z = op(x, y)
# print(op)
test_close(z, x @ y)

# Patch embedding from Vision Transformer. Take batches of (I * p) x (I * p) images and embed with a
# kernel of shape (p
# * p * C, D) and bias D.
kernel = randn(5 * 5 * 3, 12)
bias = randn(12)
images = randn(100, 55, 55, 3)

linear = EinsOp("""batch (I patch) (I patch) channels, (patch patch channels) embed_dim
                 -> batch (I I) embed_dim""")
# print(linear.program)
patches = linear(images, kernel)

affine = EinsOp('batch (I I) embed_dim, embed_dim -> batch (I I) embed_dim', combine='add')
patches = affine(patches, bias)


# Specify multiple reductions
x = randn(4, 4, 5, 6)
y1 = EinsOp('a a b c -> b', reduce={'a': 'mean', 'c': 'sum'})(x)
y2 = xp.sum(xp.mean(xp.mean(x, axis=0), axis=0), axis=-1)
test_close(y1, y2)

# Expansion
x = randn(8, 64, 3)

y1 = EinsOp('a b c -> a (b c) 4')(x)
y2 = xp.tile(x.reshape(x.shape[0], -1)[..., None], (1, 1, 4))
test_close(y1, y2)

op = EinsOp('a d 4 -> d 4', reduce=('sum', 'softmax'))
z1 = op(y1)
test_close(z1, xp.ones_like(y1[0, ...]))


# Batched pairwise Euclidean distance.
x = randn(8, 6, 32)
y = randn(8, 6, 32)
op = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce='l2_norm')
z1 = op(x, -y)
z2 = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce=('sqrt', 'sum', 'square'))(x, -y)
z3 = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce='hypot')(x, -y)
z4 = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce=R.l2_norm)(x, -y)

# Version without eins. Note how easy it would be to write x[:, None, ...] - y[:, :, None, ...],
# which would lead to the transposed version of the pairwise distances you want.
z5 = xp.sqrt(xp.sum(xp.square(x[:, :, None, ...] - y[:, None, ...]), axis=-1))  # type: ignore
test_close(z1, z2)
test_close(z2, z3)
test_close(z3, z4)
test_close(z4, z5)

# Only computing pairwise distance for the first three points in each batch of x. x = randn(8, 6,
# 32) y = randn(8, 6, 32) i = arr([0, 1, 2]) z1 = EinsOp('b n1 d, b n2 d, n1[3] -> b 3 n2',
# combine='add', reduce='l2-norm')(x, -y, i)

# Reshaping with needed hint
x = randn(8, 64, 3)
y1 = EinsOp('b (h w) c -> b h w c', symbol_values={'h': 4, 'w': 16})(x)
y2 = EinsOp('b (h=4 w) c -> b h w c')(x)
y3 = EinsOp('b (h w=16) c -> b h w c')(x)
test_close(y1, y2)
test_close(y2, y3)


# Truncated SVD
u = randn(8, 8)
s = randn(8)
v = randn(7, 7)

rank = 5
op = EinsOp('m r+_1, r+_2, r+_3 n -> m n', symbol_values={'r': rank})
usv1 = op(u, s, v)
usv2 = (u[:, :rank] * s[:rank]) @ v[:rank, :]
# print(op)
# print(usv1.shape)
# print(usv2.shape)
test_close(usv1, usv2)
