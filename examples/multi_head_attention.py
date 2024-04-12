"""
Implementation of Multi-Headed Attention (MHA) in Flax using Eins.

https://nn.labml.ai/transformers/mha.html gives a PyTorch equivalent.
"""

from functools import partial

import flax.linen as nn
from jaxtyping import Array, Float

from eins import EinsOp
from eins import ElementwiseOps as E
from eins import Transformations as T


class MultiHeadedSelfAttention(nn.Module):
    """Multi-headed self-attention."""

    num_heads: int

    @nn.compact
    def __call__(self, qkv: Float[Array, 'batch seq dim']) -> Float[Array, 'batch seq dim']:
        """
        Performs multi-headed self-attention.
        """

        batch, seq, dim = qkv.shape
        axes = {'heads': self.num_heads}

        make_linear = partial(nn.DenseGeneral, features=dim, axis=-1)

        query = make_linear(name='query')
        key = make_linear(name='key')
        value = make_linear(name='value')

        q, k, v = query(qkv), key(qkv), value(qkv)

        dim_normalize = T.from_func(lambda x, axis: x / (x.shape[axis] ** 0.5))

        # We do Q / sqrt(d_k) and then multiply K^T

        # this could be a custom reduction f(x) = sum(x)/sqrt(len(x)), but wherever possible it's
        # nice to have einsums be matrix multiplications for performance optimizations
        q_norm = EinsOp(
            'batch seq (heads d_k)', transform={'d_k': dim_normalize}, symbol_values=axes
        )(q)

        # QK^T
        qkt = EinsOp(
            """
batch q_seq (heads d_k),
batch k_seq=q_seq (heads d_k) ->
batch q_seq k_seq heads
            """,
            symbol_values=axes,
        )(q_norm, k)

        attn_scores = EinsOp('batch q_seq k_seq heads', transform={'k_seq': 'softmax'})(qkt)

        attn_values = EinsOp(
            'batch q_seq kv_seq heads, batch kv_seq (heads d_k) -> batch q_seq heads d_k',
        )(attn_scores, v)

        # project back to original dim
        out = nn.DenseGeneral(name='out', features=dim, axis=(-2, -1))(attn_values)

        return out


if __name__ == '__main__':
    import jax
    import jax.numpy as jnp
    import numpy as np

    x = jnp.array(np.random.randn(64, 20, 128))

    rng = jax.random.key(123)
    mhsa = MultiHeadedSelfAttention(num_heads=4)
    attn_out, mhsa_params = mhsa.init_with_output(rng, x)

    attn_out2, mhsa2_params = nn.MultiHeadAttention(mhsa.num_heads).init_with_output(rng, x)

    print(jnp.mean(jnp.abs(attn_out - attn_out2)))
