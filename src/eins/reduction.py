"""Reduction operations."""

import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Literal, Sequence, Union

from eins.combination import Combination, parse_combination
from eins.common_types import Array
from eins.elementwise import ElementwiseOp
from eins.utils import array_backend


class Reduction(metaclass=ABCMeta):
    """A function that takes in an arbitrary number of arrays and reduces them to a single array
    along an axis. Common examples: sum, product, mean, norm."""

    @classmethod
    def parse(cls, _name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'mean' instead of the named Mean op."""
        return None

    @abstractmethod
    def __call__(self, arr: Array, axis: int = 0) -> Array:
        """Applies the combination to an array on an axis, eliminating it.
        a1 *rest -> *rest."""
        raise NotImplementedError


# https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
ARRAY_REDUCE_OPS = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

ReductionLiteral = Literal['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']


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
        xp = array_backend(arr)
        if xp is not None and hasattr(xp, self.func_name):
            func = getattr(xp, self.func_name)
            return func(arr, axis=axis)
        elif hasattr(arr, self.func_name):
            func = getattr(arr, self.func_name)(axis=axis)
        else:
            msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
            raise ValueError(msg) from None


REDUNDANT_FOLDS = {
    'add': 'sum',
    'multiply': 'prod',
    'maximum': 'max',
    'minimum': 'min',
}


@dataclass(frozen=True, unsafe_hash=True)
class Fold(Reduction):
    """Reduction using a combination operation to fold an array.

    Many of the simple choices here already have reductions. These will give warnings for now, because you probably want
    sum instead of add or max instead of maximum.
    """

    combination: Combination

    @classmethod
    def parse(cls, name: str):
        if name in REDUNDANT_FOLDS:
            msg = f'For reduction, prefer {REDUNDANT_FOLDS[name]} instead of {name}.'
            warnings.warn(msg, stacklevel=2)

        combo = parse_combination(name)
        if combo is not None:
            return cls(combo)
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0):
        xp = array_backend(arr)
        if xp is not None and hasattr(xp, 'unstack'):
            slices = xp.unstack(arr, axis=axis)
            return reduce(self.combination, slices)
        else:
            msg = f'Cannot fold over non-Array {arr} of type {type(arr)}'
            raise ValueError(msg) from None


@dataclass(frozen=True, unsafe_hash=True)
class CompositeReduction(Reduction):
    """Reduction using elementwise operations in addition to a reduction."""

    ops: Sequence[Union[ElementwiseOp, Reduction]]

    @classmethod
    def parse(cls, _name: str):
        # perhaps a syntax like 'square |> sum |> sqrt' could be added in the future, but for now I'll only support
        # explicit tuples.
        return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        out = arr
        # must reduce exactly once
        reductions = 0
        for op in self.ops:
            if isinstance(op, Reduction):
                reductions += 1
                if reductions > 1:
                    msg = f'{self} has more than one reduction operation'
                    raise ValueError(msg)
                out = op(out, axis=axis)
            else:
                out = op(out)

        if reductions == 0:
            msg = f'{self} must have one reduction operation'
            raise ValueError(msg)
        return out


@dataclass
class UserReduction(Reduction):
    """Reduction using a user-defined function."""

    func: Callable

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        return self.func(arr, axis=axis)


def parse_reduction(name: str) -> Reduction:
    arr_parse = ArrayReduction.parse(name)
    if arr_parse is not None:
        return arr_parse

    fold_parse = Fold.parse(name)
    if fold_parse is not None:
        return fold_parse

    return None


# todo:
# p-norm
# p-deviance
# logsumexp
# softmax
# median
# quantile
