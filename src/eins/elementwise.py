"""Elementwise operations, for use in combination with reduction operations."""

from abc import ABCMeta, abstractmethod

from numpy import array_api as aa
from numpy.array_api import _array_object as ao

Array = ao.Array


class ElementwiseOp(metaclass=ABCMeta):
    """Elementwise operation on scalars that can map to arrays of any shape.
    A wrapped function of signature float -> float."""

    @classmethod
    @abstractmethod
    def parse(cls, name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'sqrt' instead of the named Sqrt op."""
        pass

    @abstractmethod
    def reduce(self, arr: Array) -> Array:
        """Applies the combination to an array on its first axis, eliminating it.
        *shape -> *shape."""
        pass


class ArrayElementwiseOp(ElementwiseOp):
    """Elementwise operation defined in Array API."""

    def __init__(self, func_name: str):
        if hasattr(aa, func_name):
            self.func_name = func_name
            self.func = getattr(aa, func_name)
        else:
            msg = f'Name {func_name} not a valid Array API function'
            raise ValueError(msg)

    def __repr__(self):
        return f'ArrayElementwiseOp[{self.func_name}]'

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        try:
            return cls(name)
        except ValueError as _e:
            return None

    def reduce(self, arr: Array):
        return self.func(arr)


# does it make sense to have every potential operation? acos, signbit, etc?

Sqrt = ArrayElementwiseOp('sqrt')
Exp = ArrayElementwiseOp('exp')
Log = ArrayElementwiseOp('log')
Square = ArrayElementwiseOp('square')
Abs = ArrayElementwiseOp('abs')
