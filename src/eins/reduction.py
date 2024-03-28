"""Reduction operations."""

from abc import ABCMeta, abstractmethod

from numpy import array_api as aa
from numpy.array_api import _array_object as ao

Array = ao.Array


class Reduction(metaclass=ABCMeta):
    """A function that takes in an arbitrary number of arrays and reduces them to a single array
    along an axis. Common examples: sum, product, mean, norm."""

    @classmethod
    @abstractmethod
    def parse(cls, name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'mean' instead of the named Mean op."""
        pass

    @abstractmethod
    def reduce(self, arr: Array, axis: int = 0) -> Array:
        """Applies the combination to an array on an axis, eliminating it.
        a1 *rest -> *rest."""
        pass


class ArrayMethod(Reduction):
    """One of the methods in the Array API."""

    def __init__(self, func_name: str):
        if hasattr(aa, func_name):
            self.func_name = func_name
            self.func = getattr(aa, func_name)
        else:
            msg = f'Name {func_name} not a valid Array API function'
            raise ValueError(msg)

    def __repr__(self):
        return f'ArrayMethod[{self.func_name}]'

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        try:
            return cls(name)
        except ValueError as _e:
            return None

    def reduce(self, arr: Array, axis: int = 0):
        return self.func(arr, axis=axis, keepdims=False)


Min = ArrayMethod('min')
Max = ArrayMethod('max')
Mean = ArrayMethod('mean')
Prod = ArrayMethod('prod')
Std = ArrayMethod('std')
Var = ArrayMethod('var')
Sum = ArrayMethod('sum')
All = ArrayMethod('all')
Any = ArrayMethod('any')


# todo:
# p-norm
# p-deviance
# logsumexp
# softmax
# median
# quantile
