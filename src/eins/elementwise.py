"""Elementwise operations, for use in combination with reduction operations."""

import math
import typing
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Union

import array_api_compat

from eins.common_types import Array, ElementwiseFunc, ElementwiseOp
from eins.utils import array_backend

# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html Every method
# without a second argument.

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

        msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
        raise ValueError(msg) from None


@dataclass(frozen=True, unsafe_hash=True)
class CustomElementwiseOp(ElementwiseOp):
    """Elementwise operation defined by user.

    func must take in an array and return an array of the same size."""

    func: ElementwiseFunc

    def __call__(self, arr: Array) -> Array:
        return self.func(arr)


@dataclass(frozen=True, unsafe_hash=True)
class Affine(ElementwiseOp):
    """
    Affine transformation.

    Applies f(x) = x * scale + shift.
    """

    scale: float = 1.0
    shift: float = 0.0

    def __call__(self, arr: Array) -> Array:
        return arr * self.scale + self.shift


@dataclass(frozen=True, unsafe_hash=True)
class DelegatedElementwiseOp(ElementwiseOp):
    """
    Operation that is delegated to different backend function calls.

    A generic Array-API-compatible version is also required, so this isn't for backend-specific
    logic: that should be a custom callable. This is for, e.g., activation functions: there are
    specific performance-optimized versions we'd like to use, but we can implement it ourselves if
    we need to (e.g., in numpy).

    Strings are members of a submodule of the overarching module for each array API, which helps
    avoid requiring any of the individual backends.
    """

    generic: Callable
    numpy: Union[Callable, str, None]
    jax: Union[Callable, str, None]
    torch: Union[Callable, str, None]

    @staticmethod
    def get_method(func_or_name: Union[Callable, str], module: str):
        if isinstance(func_or_name, str):
            if '.' in func_or_name:
                # has submodule, import that
                mod_name, func_name = func_or_name.rsplit('.', maxsplit=1)
            else:
                mod_name = ''
                func_name = func_or_name
            mod = __import__('.'.join((module, mod_name)), fromlist=[''])
            return getattr(mod, func_name)
        else:
            return func_or_name

    def __call__(self, arr: Array) -> Array:
        if array_api_compat.is_torch_array(arr):
            module = 'torch'
        elif array_api_compat.is_jax_array(arr):
            module = 'jax'
        elif array_api_compat.is_numpy_array(arr):
            module = 'numpy'
        else:
            return self.generic(arr, axis=axis)

        func = getattr(self, module)
        if func is None:
            func = self.generic

        try:
            return self.get_method(func, module)(arr)
        except TypeError:
            # try using dim= instead of axis=
            # works for torch
            return self.get_method(func, module)(arr)


class StandardDelegatedElementwiseOp(DelegatedElementwiseOp):
    """
    Delegated elementwise operation with standard backend implementations, without the need for
    special documentation.
    """


def parse_elementwise(name: str) -> Optional[ElementwiseOp]:
    arr_parse = ArrayElementwiseOp.parse(name)
    if arr_parse is not None:
        return arr_parse
    else:
        return None


ElementwiseLiteral = ArrayElementwiseLiteral

ops = {str(op): parse_elementwise(op) for op in typing.get_args(ElementwiseLiteral)}
ops = {k: v for k, v in ops.items() if v is not None}


# others to add: logit expit/sigmoid activation functions
