"""Reduction operations."""

import typing
import warnings
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from math import isnan
from typing import Literal, Optional, Sequence, Union

from eins.combination import Combination, parse_combination
from eins.common_types import Array, ElementwiseOp, Reduction, ReductionFunc, Transformation
from eins.utils import array_backend

# https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
ArrayReductionLiteral = Literal['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

ARRAY_REDUCE_OPS = [str(x) for x in typing.get_args(ArrayReductionLiteral)]


@dataclass(frozen=True, unsafe_hash=True)
class ArrayReduction(Reduction):
    """One of the methods in the Array API. Falls back to a method of the object."""

    func_name: str

    def __repr__(self):
        return repr(self.func_name)

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name in ARRAY_REDUCE_OPS:
            return cls(name)
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None and hasattr(xp, self.func_name):
            func = getattr(xp, self.func_name)
            return func(arr, axis=axis)
        elif hasattr(arr, self.func_name):
            func = getattr(arr, self.func_name)(axis=axis)

        msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
        raise ValueError(msg)


REDUNDANT_FOLDS = {
    'add': 'sum',
    'multiply': 'prod',
    'maximum': 'max',
    'minimum': 'min',
}


@dataclass(frozen=True, unsafe_hash=True)
class Fold(Reduction):
    """Reduction using a combination operation to fold an array.

    Many of the simple choices here already have reductions. These will give warnings for now,
    because you probably want sum instead of add or max instead of maximum.
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

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            # unstack isn't implemented in Torch yet, even though it's part of the API
            slc: list[Union[slice, int]] = [slice(None)] * arr.ndim
            slices = []
            for i in range(arr.shape[axis]):
                slc[axis] = i
                slices.append(arr[tuple(slc)])
            return reduce(self.combination, slices)
        else:
            msg = f'Cannot fold over non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


@dataclass(frozen=True, unsafe_hash=True)
class PowerNorm(Reduction):
    """p-norm: defaults to L2 norm.

    Parses a string of the form `{p}_norm`, such as `2_norm` for the Euclidean norm. Also supports a
    prefix of `L` or `l`, so you can write `l1_norm` instead of `1_norm` if you prefer. `norm` is
    special-cased to the Euclidean norm.

    Supports the
    [np.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) vector
    norm options: `inf_norm` is `max(abs(x))`, `-inf_norm` is `min(abs(x))`, and `0_norm` is `sum(x
    != 0)`."""

    power: float = 2

    @classmethod
    def parse(cls, name: str):
        if name == 'norm':
            return cls()

        if name and name[0] in ('L', 'l'):
            name = name[1:]
        if name.endswith('_norm'):
            power = name[: -len('_norm')]
            try:
                power = float(power)
                if not isnan(power):
                    return cls(float(power))
            except ValueError:
                return None
        return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            if self.power == 0:
                return xp.sum(xp.abs(arr) != 0, axis=axis)
            elif self.power == float('inf'):
                return xp.max(xp.abs(arr), axis=axis)
            elif self.power == float('-inf'):
                return xp.min(xp.abs(arr), axis=axis)
            # Faster special-cases:
            elif self.power == 1:
                return xp.sum(xp.abs(arr), axis=axis)
            elif self.power == 2:
                return xp.sqrt(xp.sum(xp.square(arr), axis=axis))
            else:
                return xp.sum(xp.abs(arr) ** self.power, axis=axis) ** (1 / self.power)
        else:
            msg = f'Cannot compute reduction for non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


# we can't enumerate every valid norm string, but we can special-case the common ones
NormLiteral = Literal['norm', 'l2_norm', 'l1_norm', 'inf_norm']


class Range(Reduction):
    """Computes the range of an array along an axis. Equivalent to np.ptp."""

    @classmethod
    def parse(cls, name: str):
        if name in ('range', 'ptp'):
            return cls()
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            return xp.max(arr, axis=axis) - xp.min(arr, axis=axis)
        else:
            msg = f'Cannot compute reduction for non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


RangeLiteral = Literal['range', 'ptp']


@dataclass(frozen=True, unsafe_hash=True)
class CompositeReduction(Reduction):
    """Reduction using elementwise operations and transformations in addition to a reduction.

    Functions are applied from right-to-left. For example, `('log', 'sum', 'exp')` computes
    `log(sum(exp(x)))`."""

    ops: Sequence[Union[ElementwiseOp, Transformation, Reduction]]

    @classmethod
    def parse(cls, _name: str):
        # perhaps a syntax like 'square |> sum |> sqrt' could be added in the future, but for now
        # I'll only support explicit tuples.
        return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        out = arr
        # must reduce exactly once
        reductions = 0
        for op in self.ops[::-1]:
            if isinstance(op, Reduction):
                reductions += 1
                if reductions > 1:
                    msg = f'{self} has more than one reduction operation'
                    raise ValueError(msg)
                out = op(out, axis=axis)
            elif isinstance(op, Transformation):
                if reductions == 1:
                    msg = f'{self} is invalid. Cannot transform after reducing.'
                    msg += '\nPerhaps you meant the reverse? Sequences are applied right-to-left.'
                out = op(out, axis=axis)
            else:
                out = op(out)

        if reductions == 0:
            msg = f'{self} must have one reduction operation'
            raise ValueError(msg)
        return out


@dataclass
class CustomReduction(Reduction):
    """Reduction using a user-defined function.

    func must take in an array and an axis argument, returning an array with that axis removed."""

    func: ReductionFunc

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        return self.func(arr, axis=axis)


def parse_reduction(name: str) -> Optional[Reduction]:
    for cls in (ArrayReduction, Fold, PowerNorm, Range):
        parse = cls.parse(name)
        if parse is not None:
            return parse

    return None


ReductionLiteral = Union[ArrayReductionLiteral, NormLiteral, RangeLiteral]

ops = {
    str(op): parse_reduction(op)
    for op in chain.from_iterable(map(typing.get_args, typing.get_args(ReductionLiteral)))
}
ops = {k: v for k, v in ops.items() if v is not None}
