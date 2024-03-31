from string import ascii_lowercase

from eins import EinsOp, einsop

# Set this to 'jax', 'numpy', or 'torch'
BACKEND = 'torch'

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


def randn(*shape):
    if BACKEND == 'jax':
        import jax.numpy as jnp
        import numpy as np

        # TODO debug why bfloat16 doesn't work here
        return jnp.asarray(np.random.randn(*shape))
    elif BACKEND == 'numpy':
        import numpy as np

        return jnp.array(np.random.randn(*shape))
    elif BACKEND == 'torch':
        import torch

        return torch.randn(*shape)
    else:
        raise ValueError


EPSILON = 1e-3


def test_close(a, b):
    diffs = xp.mean(xp.abs(a - b))
    assert diffs.max() < EPSILON, f'{a.shape} != {b.shape}, {a.mean()}, {b.mean()}, {diffs.max()}'


# Simple matrix multiplication
x = randn(32, 64)
y = randn(64, 16)
z = EinsOp('a b, b c -> a c')(x, y)
test_close(z, x @ y)

# Patch embedding from Vision Transformer. Take batches of (I * p) x (I * p) images and embed with a kernel of shape (p
# * p * C, D) and bias D.
kernel = randn(5 * 5 * 3, 12)
bias = randn(12)
images = randn(100, 55, 55, 3)

linear = EinsOp('batch (I patch) (I patch) channels, (patch patch channels) embed_dim -> batch (I I) embed_dim')
patches = linear(images, kernel)

affine = EinsOp('batch (I I) embed_dim, embed_dim -> batch (I I) embed_dim', combine='add')
patches = affine(patches, bias)


# Batched pairwise Euclidean distance.
x = randn(8, 6, 32)
y = randn(8, 6, 32)
z1 = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce='l2-norm')(x, -y)
z2 = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce=('sqrt', 'sum', 'square'))(x, -y)
z3 = EinsOp('b n1 d, b n2 d -> b n1 n2', combine='add', reduce='hypot')(x, -y)

# Version without eins. Note how easy it would be to write x[:, None, ...] - y[:, :, None, ...], which would lead to the
# transposed version of the pairwise distances you want.
z4 = xp.sqrt(xp.sum(xp.square(x[:, :, None, ...] - y[:, None, ...]), axis=-1))
test_close(z1, z2)
test_close(z2, z3)
test_close(z3, z4)
