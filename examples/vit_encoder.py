"""
Implementation of Vision Transformer (ViT) encoding (patch and position embeddings) in Flax using
Eins.

https://nn.labml.ai/transformers/vit/index.html gives a PyTorch equivalent.
"""

import flax.linen as nn
from jaxtyping import Array, Float

from eins.einsop import EinsOp


class PatchEmbeddings(nn.Module):
    """Patch embedding."""

    embed_dim: int
    patch_size: int

    @nn.compact
    def __call__(self, ims: Float[Array, 'b c h w']) -> Float[Array, '(h*w/p^2) b embed']:
        """
        Embeds an image into patches.
        """

        b, c, h, w = ims.shape

        kernel = self.param(
            'kernel',
            nn.initializers.truncated_normal(0.02),
            (self.patch_size, self.patch_size, c, self.embed_dim),
        )

        bias = self.param('bias', nn.initializers.truncated_normal(0.02), (self.embed_dim,))

        linear = EinsOp(
            """batch chan (num_hpatch patch) (num_wpatch patch),
            patch patch chan embed -> (num_hpatch num_wpatch) batch embed"""
        )
        affine = EinsOp('num_patches batch embed, embed -> num_patches batch embed', combine='add')

        out = linear(ims, kernel)
        out = affine(out, bias)
        return out


class LearnedPositionalEmbeddings(nn.Module):
    """Positional embeddings."""

    embed_dim: int
    max_len: int

    @nn.compact
    def __call__(
        self, x: Float[Array, 'patches batch embed']
    ) -> Float[Array, 'patches batch embed']:
        """
        Learned positional embeddings.
        """

        embeddings = self.param(
            'pos_embed', nn.initializers.truncated_normal(0.02), (self.max_len, self.embed_dim)
        )

        op = EinsOp(
            'patches batch embed, patches+unused embed -> patches batch embed', combine='add'
        )

        return op(x, embeddings)


if __name__ == '__main__':
    import jax
    import jax.numpy as jnp
    import numpy as np

    x = jnp.array(np.random.randn(64, 3, 50, 100))

    rng = jax.random.key(123)
    patch_emb = PatchEmbeddings(embed_dim=384, patch_size=5)
    pos_emb = LearnedPositionalEmbeddings(embed_dim=patch_emb.embed_dim, max_len=1000)
    patchified, patch_params = patch_emb.init_with_output(rng, x)
    patch_with_pos_emb, pos_params = pos_emb.init_with_output(rng, patchified)
    print(patch_with_pos_emb.shape)
