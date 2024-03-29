"""General utilities."""

import array_api_compat


def array_backend(x):
    """Gets the Array API backend for x, returning None if no such backend exists."""
    if array_api_compat.is_array_api_obj(x):
        return array_api_compat.array_namespace(x)
    else:
        return None
