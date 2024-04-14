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
from typing import Callable, Literal, Optional, Sequence, Union, cast

import array_api_compat

from eins.combination import Combination, parse_combination
from eins.common_types import Array, ElementwiseOp, Transformation
from eins.reduction import PowerNorm
from eins.utils import array_backend

# cumsum is the numpy version, so we support that name in addition to the Array API cumulative sum
ArrayTransformationLiteral = Literal['sort', 'cumulative_sum', 'cumsum']


ARRAY_TRANSFORM_OPS = [str(x) for x in typing.get_args(ArrayTransformationLiteral)]


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
            return arr / xp.maximum(norm, self.eps)
        else:
            msg = f'Cannot normalize non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


MinMaxNormalizeLiteral = Literal['min_max_normalize']


@dataclass(frozen=True, unsafe_hash=True)
class MinMaxNormalize(Transformation):
    """
    Transforms data so 0 is the minimum and 1 is the maximum. Can be thought of as min-max
    normalization.
    """

    eps: float = 1e-12

    @classmethod
    def parse(cls, name: str):
        if name in ('min_max_normalize',):
            return cls()
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        xp = array_backend(arr)
        if xp is not None:
            lo = xp.expand_dims(xp.min(arr, axis=axis), axis=axis)
            hi = xp.expand_dims(xp.max(arr, axis=axis), axis=axis)

            return (arr - lo) / xp.maximum(hi - lo, self.eps)
        else:
            msg = f'Cannot min-max normalize non-Array {arr} of type {type(arr)}'
            raise ValueError(msg)


@dataclass(frozen=True, unsafe_hash=True)
class DelegatedTransformation(Transformation):
    """
    Operation that is delegated to different backend function calls.

    A generic Array-API-compatible version is also required, so this isn't for backend-specific
    logic: that should be a custom callable. This is for, e.g., softmax: there are specific
    performance-optimized versions we'd like to use, but we can implement it ourselves if we need to
    (e.g., in numpy, which doesn't have softmax).

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

    def __call__(self, arr: Array, axis: int = 0) -> Array:
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
            return self.get_method(func, module)(arr, axis=axis)
        except TypeError:
            # try using dim= instead of axis=
            # works for torch
            return self.get_method(func, module)(arr, dim=axis)


def _generic_softmax(arr: Array, axis: int = 0) -> Array:
    """
    Generic default softmax implementation.
    """
    # https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/special/_logsumexp.py#L231

    # log-sum-exp trick is important
    xp = array_backend(arr)
    amax = xp.expand_dims(xp.max(arr, axis=axis), axis=axis)
    exp_shifted = xp.exp(arr - amax)

    return PowerNormalize(PowerNorm(1.0))(exp_shifted)


_SoftmaxDelegate = DelegatedTransformation(
    generic=_generic_softmax, numpy=None, jax='nn.softmax', torch='nn.functional.softmax'
)

SoftmaxLiteral = Literal['softmax']


@dataclass(frozen=True, unsafe_hash=True)
class Softmax(Transformation):
    """
    Softmax normalization. Delegates to array backends, so should have the same performance/behavior
    as their code.

    Has a user-specified temperature, defaulting to the standard softmax with 1. Inputs are divided
    by the temperature before softmax is applied: higher values mean the distribution is more
    uniform. This can't be parsed for now, so the only option is the standard 'softmax'.
    """

    temperature: float = 1

    @classmethod
    def parse(cls, _name: str) -> Optional[Transformation]:
        if _name == 'softmax':
            return cls()
        else:
            return None

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        return _SoftmaxDelegate(arr / self.temperature, axis=axis)


@dataclass(frozen=True, unsafe_hash=True)
class CustomTransformation(Transformation):
    """
    Reduction using a user-defined function.

    func must take in an array and an axis argument, returning an array of the same shape.
    """

    func: Callable

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        return self.func(arr, axis=axis)


def parse_transformation(name: str) -> Optional[Transformation]:
    for cls in (ArrayTransformation, Scan, PowerNormalize, MinMaxNormalize, Softmax):
        parse = cls.parse(name)
        if parse is not None:
            return parse

    return None


@dataclass(frozen=True, unsafe_hash=True)
class CompositeTransformation(Transformation):
    """
    Applies multiple transformations in sequence.
    """

    transformations: Sequence[Union[ElementwiseOp, Transformation]]

    def __call__(self, arr: Array, axis: int = 0) -> Array:
        for t in self.transformations[::-1]:
            if isinstance(t, ElementwiseOp):
                arr = t(arr)
            else:
                arr = t(arr, axis=axis)
        return arr


TransformationLiteral = Union[
    ArrayTransformationLiteral,
    ScanLiteral,
    NormalizeLiteral,
    MinMaxNormalizeLiteral,
    SoftmaxLiteral,
]

ops = {
    str(op): parse_transformation(op)
    for op in chain.from_iterable(map(typing.get_args, typing.get_args(TransformationLiteral)))
}
ops = {k: v for k, v in ops.items() if v is not None}
