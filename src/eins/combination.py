"""Operations that combine arrays."""

import typing
from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Union

from eins.common_types import Array, Combination
from eins.elementwise import ElementwiseOp
from eins.utils import array_backend

# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
# Must be commutative, associative, R x R -> R

ArrayCombineLiteral = Literal[
    'add', 'hypot', 'logaddexp', 'maximum', 'minimum', 'multiply', 'bitwise_xor', 'bitwise_and', 'bitwise_or'
]

ARRAY_COMBINE_OPS = [str(x) for x in typing.get_args(ArrayCombineLiteral)]


@dataclass(frozen=True, unsafe_hash=True)
class ArrayCombination(Combination):
    """Combination operation defined in Array API."""

    func_name: str

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name in ARRAY_COMBINE_OPS:
            return cls(name)
        else:
            return None

    def __call__(self, *arrs: Array) -> Array:
        if len(arrs) == 0:
            msg = 'Cannot combine empty list of arrays'
            raise ValueError(msg)

        arr = arrs[0]
        xp = array_backend(arr)
        if xp is not None and hasattr(xp, self.func_name):
            func = getattr(xp, self.func_name)
        elif hasattr(arr, self.func_name):
            func = lambda x, y: getattr(x, self.func_name)(y)  # noqa: E731
        else:
            msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
            raise ValueError(msg) from None

        out = arrs[0]
        for other in arrs[1:]:
            out = func(out, other)
        return out


class UserCombination(Combination):
    """Combination operation using a user-defined function."""

    func: Callable

    def __call__(self, *arrs: Array) -> Array:
        return self.func(*arrs)


@dataclass
class CompositeCombination(Combination):
    """Combination operation with one or more elementwise ops."""

    ops: Sequence[Union[ElementwiseOp, Combination]]

    def __call__(self, *arrs: Array) -> Array:
        out = arrs
        combines = 0
        for op in self.ops:
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


def parse_combination(name: str) -> Combination:
    arr_parse = ArrayCombination.parse(name)
    if arr_parse is not None:
        return arr_parse
    else:
        return None
