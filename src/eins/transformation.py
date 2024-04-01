"""
Transformations, inspired by the Pandas usage, are operations that maintain the array shape but
perform an operation along an axis. The motivating example is normalize: it must be done on an axis,
but it doesn't eliminate it.
"""

# normalize scanned combine ops rank quantile (min-max normalize) sort partition

import typing
import warnings
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Callable, Literal, Optional, Union, cast

from eins.combination import Combination, parse_combination
from eins.common_types import Array, Transformation
from eins.reduction import PowerNorm
from eins.utils import array_backend

# cumsum is the numpy version, so we support that name in addition to the Array API cumulative sum
ArrayTransformationLiteral = Literal['sort', 'cumulative_sum', 'cumsum']


@dataclass(frozen=True, unsafe_hash=True)
class ArrayTransformation(Transformation):
    """One of the methods in the Array API. Falls back to a method of the object."""

    func_name: str

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name in ('cumulative_sum', 'cumsum'):
            return cls('cumulative_sum')
        elif name in typing.get_args(ArrayTransformationLiteral):
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


REDUNDANT_SCANS = {'add': 'cumsum'}

# the convention is to use the name of the reduction, not the combination, even though this makes a
# lot more sense. So, instead of forcing people to use 'cummul', they can use 'cumprod'.
ScanLiteral = Literal['cumprod', 'cummax', 'cummin']

SCAN_ALIASES = {'cumprod': 'cummultiply', 'cummax': 'cummaximum', 'cummin': 'cumminimum'}


@dataclass(frozen=True, unsafe_hash=True)
class Scan(Transformation):
    """
    Transformation using a combination operation to scan over an array: for example, computing the
    cumulative product, or running maximum.

    Parses strings like "cumprod" or "cum-sum".
    """

    combination: Combination

    @classmethod
    def parse(cls, name: str):
        if name in SCAN_ALIASES:
            return cls.parse(SCAN_ALIASES[name])

        op_name = name
        for pref in ('cum-', 'cum'):
            if op_name.startswith(pref):
                op_name = op_name[len(pref) :]
                break
        if op_name in REDUNDANT_SCANS:
            msg = f'For transformation, prefer {REDUNDANT_SCANS[op_name]} instead of {name}.'
            warnings.warn(msg, stacklevel=2)

        combo = parse_combination(op_name)
        if combo is not None:
            return cls(combo)
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            slices = xp.unstack(arr, axis=axis)
            return cast(Array, accumulate(slices, self.combination))
        else:
            msg = f'Cannot scan over non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


DEFAULT_NORM = PowerNorm()
NormalizeLiteral = Literal['normalize', 'l2_normalize', 'l1_normalize', 'inf_normalize']


@dataclass(frozen=True, unsafe_hash=True)
class PowerNormalize(Transformation):
    """
    Normalizes the inputs to have Lp norm equal to 1. See torch.nn.functional.normalize for details.

    Parses like norm, but instead of norm it's normalize.
    """

    norm: PowerNorm = DEFAULT_NORM
    eps: float = 1e-12

    @classmethod
    def parse(cls, name: str):
        if name.endswith('normalize'):
            norm = PowerNorm.parse(name[: -len('alize')])
            if norm is not None:
                return cls(norm=norm)

        return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            norm = xp.expand_dims(self.norm(arr, axis=axis), axis=axis)
            return xp / xp.maximum(norm, self.eps)
        else:
            msg = f'Cannot normalize non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


QuantileLiteral = Literal['quantile', 'min_max_normalize']


class Quantile(Transformation):
    """
    Transforms data so 0 is the minimum and 1 is the maximum. Can be thought of as min-max
    normalization.
    """

    eps: float = 1e-12

    @classmethod
    def parse(cls, name: str):
        if name in ('quantile', 'min-max-normalize'):
            return cls()
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            lo = xp.expand_dims(xp.min(arr, axis=axis), axis=axis)
            hi = xp.expand_dims(xp.max(arr, axis=axis), axis=axis)

            return (xp - lo) / xp.maximum(hi - lo, self.eps)
        else:
            msg = f'Cannot min-max normalize non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


@dataclass
class CustomTransformation(Transformation):
    """
    Reduction using a user-defined function.

    func must take in an array and an axis argument, returning an array of the same shape.
    """

    func: Callable

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        return self.func(arr, axis=axis)


def parse_transformation(name: str) -> Optional[Transformation]:
    for cls in (ArrayTransformation, Scan, PowerNormalize, Quantile):
        parse = cls.parse(name)
        if parse is not None:
            return parse

    return None


TransformationLiteral = Union[
    ArrayTransformationLiteral, ScanLiteral, NormalizeLiteral, QuantileLiteral
]

ops = {
    str(op): parse_transformation(op)
    for op in chain.from_iterable(map(typing.get_args, typing.get_args(TransformationLiteral)))
}