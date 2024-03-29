from string import ascii_lowercase

from eins import einsop

# Set this to 'jax', 'numpy', or 'torch'
BACKEND = 'torch'


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
    shape = ' '.join(ascii_lowercase[: a.ndim])
    diffs = einsop(f'{shape}, {shape} -> {shape}', a, -b, combine=('add', 'abs'))
    assert diffs.max() < EPSILON, f'{a.shape} != {b.shape}, {a.mean()}, {b.mean()}, {diffs.max()}'


# Simple matrix multiplication
x = randn(32, 64)
y = randn(64, 16)
z = einsop('a b, b c -> a c', x, y)
test_close(z, x @ y)

# Patch embedding from Vision Transformer. Take batches of (I * p) x (I * p) images and embed with a kernel of shape (p
# * p * C, D) and bias D.
kernel = randn(5 * 5 * 3, 12)
bias = randn(12)
images = randn(5, 55, 55, 3)

patches = einsop(
    'batch (I patch) (I patch) channels, (patch patch channels) embed_dim -> batch (I I) embed_dim', images, kernel
)
patches = einsop('batch (I I) embed_dim, embed_dim -> batch (I I) embed_dim', patches, bias, combine='add')
