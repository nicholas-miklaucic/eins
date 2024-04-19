"""General utilities."""

import array_api_compat

from eins.common_types import Arr, Array, ArrayBackend


def array_backend(x: Arr) -> ArrayBackend[Arr]:
    """Gets the Array API backend for x, returning None if no such backend exists."""
    if array_api_compat.is_jax_array(x):
        import jax
        import jax.numpy as jnp

        _jnp_arr: Array = jnp.ones(1)
        xp_jnp: ArrayBackend[jax.Array] = jnp

        # I think the type error here is from not knowing x's type: that's fine, but we want to
        # catch if jax.numpy isn't valid.
        return xp_jnp  # type: ignore
    elif array_api_compat.is_numpy_array(x):
        from array_api_compat import numpy as arr_numpy

        _arr: Array = arr_numpy.ones(1)
        xp_numpy: ArrayBackend[arr_numpy.ndarray] = arr_numpy

        return xp_numpy  # type.ignore
    elif array_api_compat.is_torch_array(x):
        from array_api_compat import torch as arr_torch

        _tens: Array = arr_torch.ones(123)
        # TODO figure out why this doesn't apply
        xp_torch: ArrayBackend[arr_torch.Tensor] = arr_torch  # type: ignore

        return xp_torch  # type: ignore
    elif array_api_compat.is_array_api_obj(x):
        return array_api_compat.array_namespace(x)
    else:
        msg = f'Cannot get Array API backend for {x} of type {type(x)}'
        raise ValueError(msg)
