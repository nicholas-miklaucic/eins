"""Transformations, inspired by the Pandas usage, are operations that maintain the array shape but perform an operation
along an axis. The motivating example is softmax: softmax must be done on an axis, but it doesn't eliminate it."""

# normalize
# scanned combine ops
# rank
# quantile (min-max normalize)
# sort
# partition

import typing
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal, Union

from eins.combination import Combination, parse_combination
from eins.common_types import Array
from eins.utils import array_backend


class Transformation(metaclass=ABCMeta):
    """A function that takes in an array and axis and returns an array of the same shape.
    Examples are softmax, normalization, cumulative sum, etc."""

    @classmethod
    def parse(cls, _name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None."""
        return None

    @abstractmethod
    def __call__(self, arr: Array, axis: int = 0) -> Array:
        """Applies the transformation to an array on an axis."""
        raise NotImplementedError


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

    def __call__(self, arr: Array, axis: int = 0):
        xp = array_backend(arr)
        if xp is not None and hasattr(xp, self.func_name):
            func = getattr(xp, self.func_name)
            return func(arr, axis=axis)
        elif hasattr(arr, self.func_name):
            func = getattr(arr, self.func_name)(axis=axis)
        else:
            msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
            raise ValueError(msg)


REDUNDANT_SCANS = {'add': 'cumsum'}

# the convention is to use the name of the reduction, not the combination, even though this makes a lot more sense. So,
# instead of forcing people to use 'cummul', they can use 'cumprod'.
ScanLiteral = Literal['cumprod', 'cummax', 'cummin']

SCAN_ALIASES = {'cumprod': 'cummultiply', 'cummax': 'cummaximum', 'cummin': 'cumminimum'}


@dataclass(frozen=True, unsafe_hash=True)
class Scan(Transformation):
    """
    Transformation using a combination operation to scan over an array: for example, computing the cumulative product,
    or running maximum.

    Parses strings like "cumprod" or "cum-sum".
    """

    combination: Combination

    @classmethod
    def parse(cls, name: str):
        if name in SCAN_ALIASES:
            return cls.parse(SCAN_ALIASES[name])

        op_name = name.removeprefix('cum').removeprefix('-')
        if op_name in REDUNDANT_SCANS:
            msg = f'For transformation, prefer {REDUNDANT_SCANS[op_name]} instead of {name}.'
            warnings.warn(msg, stacklevel=2)

        combo = parse_combination(op_name)
        if combo is not None:
            return cls(combo)
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0):
        xp = array_backend(arr)
        if xp is not None:
            slices = xp.unstack(arr, axis=axis)
            return accumulate(slices, self.combination)
        else:
            msg = f'Cannot scan over non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


def parse_transformation(name: str) -> Transformation:
    for cls in (ArrayTransformation, Scan):
        parse = cls.parse(name)
        if parse is not None:
            return parse

    return None


TransformationLiteral = Union[ArrayTransformationLiteral, ScanLiteral]
