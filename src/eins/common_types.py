"""Type definitions"""

from abc import ABCMeta, abstractmethod
from typing import TypeVar

Array = TypeVar('Array')


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


class Combination(metaclass=ABCMeta):
    """Operation to combine array values together. Commutative, associative function of signature R x R -> R."""

    @classmethod
    def parse(cls, _name: str):
        """Attempts to construct an instance of the operation from the string name.
        If unsuccessful, returns None. Used for shorthand syntax, like passing in
        'sqrt' instead of the named Sqrt op."""
        return None

    @abstractmethod
    def __call__(self, *arrs: Array) -> Array:
        """Applies the function to the inputs, returning a single output."""
        raise NotImplementedError
