"""Operations that combine arrays."""

import typing
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

from eins.common_types import Array, Combination, CombinationFunc
from eins.elementwise import ElementwiseOp
from eins.utils import array_backend

# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html Must be
# commutative, associative, R x R -> R

ArrayCombineLiteral = Literal[
    'add',
    'hypot',
    'logaddexp',
    'maximum',
    'minimum',
    'multiply',
    'bitwise_xor',
    'bitwise_and',
    'bitwise_or',
]

ARRAY_COMBINE_OPS = [str(x) for x in typing.get_args(ArrayCombineLiteral)]


@dataclass(frozen=True, unsafe_hash=True)
class ArrayCombination(Combination):
    """Combination operation defined in Array API."""

    func_name: str

    def __repr__(self):
        return repr(self.func_name)

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name in ARRAY_COMBINE_OPS:
            return cls(name)
        else:
            return None

    def __call__(self, arr1: Array, arr2: Array) -> Array:
        arrs = (arr1, arr2)
        if len(arrs) == 0:
            msg = 'Cannot combine empty list of arrays'
            raise ValueError(msg)

        arr = arrs[0]
        xp = array_backend(arr)
        if xp is not None and hasattr(xp, self.func_name):
            func = getattr(xp, self.func_name)
        elif hasattr(arr, self.func_name):
            func = lambda x, y: getattr(x, self.func_name)(y)
        else:
            msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
            raise ValueError(msg) from None

        out = arrs[0]
        for other in arrs[1:]:
            out = func(out, other)
        return out


@dataclass
class CustomCombination(Combination):
    """Combination operation using a user-defined function."""

    func: CombinationFunc

    def __call__(self, arr1: Array, arr2: Array) -> Array:
        return self.func(arr1, arr2)


@dataclass
class CompositeCombination(Combination):
    """Combination operation with one or more elementwise ops."""

    ops: Sequence[Union[ElementwiseOp, Combination]]

    def __call__(self, arr1: Array, arr2: Array) -> Array:
        arrs = (arr1, arr2)
        out = arrs
        combines = 0
        for op in self.ops[::-1]:
            if isinstance(op, Combination):
                combines += 1
                if combines > 1:
                    msg = f'{self} has more than one combination operation'
                    raise ValueError(msg)
                out = [op(*arrs)]
            else:
                out = [op(o) for o in out]
        return out[0]


CombineLiteral = ArrayCombineLiteral


def parse_combination(name: str) -> Optional[Combination]:
    arr_parse = ArrayCombination.parse(name)
    if arr_parse is not None:
        return arr_parse
    else:
        return None


ops = {str(op): parse_combination(op) for op in typing.get_args(CombineLiteral)}
ops = {k: v for k, v in ops.items() if v is not None}
