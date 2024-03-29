"""Operations that combine arrays."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import array_api_compat

from eins.common_types import Array


class Combination(metaclass=ABCMeta):
    """Operation to combine array values together. Commutative, associative function of signature R x R -> R."""

    @classmethod
    @abstractmethod
    def parse(cls, name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'sqrt' instead of the named Sqrt op."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *arrs: Array) -> Array:
        """Applies the function to the inputs, returning a single output."""
        raise NotImplementedError


# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html

# Must be commutative, associative, R x R -> R
ARRAY_COMBINE_OPS = [
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


@dataclass(frozen=True, unsafe_hash=True)
class ArrayCombination(Combination):
    """Combination operation defined in Array API."""

    func_name: str

    def __str__(self):
        return self.func_name

    @classmethod
    def parse(cls, name: str):
        if name.lower().strip() in ARRAY_COMBINE_OPS:
            return cls(name.lower().strip())
        else:
            return None

    def __call__(self, *arrs: Array) -> Array:
        if len(arrs) == 0:
            msg = 'Cannot combine empty list of arrays'
            raise ValueError(msg)

        try:
            arr = arrs[0]
            xp = array_api_compat.get_namespace(arr)
            if hasattr(xp, self.func_name):
                func = getattr(xp, self.func_name)
        except TypeError:
            if hasattr(arr, self.func_name):
                func = lambda x, y: getattr(x, self.func_name)(y)  # noqa: E731
            else:
                msg = f'Name {self.func_name} not a valid function for array of type {type(arr)}'
                raise ValueError(msg) from None

        out = arrs[0]
        for other in arrs[1:]:
            out = func(out, other)
        return out
