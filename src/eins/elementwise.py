"""Elementwise operations, for use in combination with reduction operations."""

import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

from eins.common_types import Array
from eins.utils import array_backend


class ElementwiseOp(metaclass=ABCMeta):
    """Elementwise operation on scalars that can map to arrays of any shape.
    Signature *arr -> *arr."""

    @classmethod
    def parse(cls, _name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'sqrt' instead of the named Sqrt op."""
        return None

    @abstractmethod
    def __call__(self, arr: Array) -> Array:
        """Applies the function elementwise."""
        raise NotImplementedError


# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
# Every method without a second argument.

ArrayElementwiseLiteral = Literal[
    'abs',
    'acos',
    'acosh',
    'asin',
    'asinh',
    'atan',
    'atanh',
    'bitwise_invert',
    'ceil',
    'conj',
    'cos',
    'cosh',
    'exp',
    'expm1',
    'floor',
    'imag',
    'log',
    'log1p',
    'log2',
    'log10',
    'negative',
    'positive',
    'real',
    'round',
    'sign',
    'sin',
    'sinh',
    'square',
    'sqrt',
    'tan',
    'tanh',
    'trunc',
]

ARRAY_ELEMWISE_OPS = typing.get_args(ArrayElementwiseLiteral)


@dataclass(frozen=True, unsafe_hash=True)
class ArrayElementwiseOp(ElementwiseOp):
    """Elementwise operation defined in Array API."""

    func_name: str

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name in ARRAY_ELEMWISE_OPS:
            return cls(name)
        else:
            return None

    def __call__(self, arr: Array) -> Array:
        xp = array_backend(arr)
        if hasattr(xp, self.func_name):
            func = getattr(xp, self.func_name)
            return func(arr)
        elif hasattr(arr, self.func_name):
            func = getattr(arr, self.func_name)()
        else:
            msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
            raise ValueError(msg) from None


def parse_elementwise(name: str) -> ElementwiseOp:
    arr_parse = ArrayElementwiseOp.parse(name)
    if arr_parse is not None:
        return arr_parse
    else:
        return None


ElementwiseLiteral = ArrayElementwiseLiteral
