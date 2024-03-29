"""Reduction operations."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import array_api_compat

from eins.common_types import Array


class Reduction(metaclass=ABCMeta):
    """A function that takes in an arbitrary number of arrays and reduces them to a single array
    along an axis. Common examples: sum, product, mean, norm."""

    @classmethod
    @abstractmethod
    def parse(cls, name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'mean' instead of the named Mean op."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, arr: Array, axis: int = 0) -> Array:
        """Applies the combination to an array on an axis, eliminating it.
        a1 *rest -> *rest."""
        raise NotImplementedError


# https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
ARRAY_REDUCE_OPS = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']


@dataclass(frozen=True, unsafe_hash=True)
class ArrayReduction(Reduction):
    """One of the methods in the Array API. Falls back to a method of the object."""

    func_name: str

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name.lower().strip() in ARRAY_REDUCE_OPS:
            return cls(name.lower().strip())
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0):
        try:
            xp = array_api_compat.get_namespace(arr)
            if hasattr(xp, self.func_name):
                func = getattr(xp, self.func_name)
                return func(arr, axis=axis)
        except TypeError:
            if hasattr(arr, self.func_name):
                func = getattr(arr, self.func_name)(axis=axis)
            else:
                msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
                raise ValueError(msg) from None


# todo:
# p-norm
# p-deviance
# logsumexp
# softmax
# median
# quantile
