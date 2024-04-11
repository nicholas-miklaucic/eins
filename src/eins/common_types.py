"""Type definitions"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Protocol, Sequence, Tuple, TypeVar, Union

Array = TypeVar('Array')


class Reduction(metaclass=ABCMeta):
    """
    A function with signature f(Array, axis: int) → Array that removes the axis. Common examples:
    sum, product, mean, norm.
    """

    @classmethod
    def parse(cls, _name: str) -> Optional['Reduction']:
        """
        Attempts to construct an instance of the operation from the string name. If unsuccessful,
        returns None. Used for shorthand syntax, like passing in 'mean' instead of the named Mean
        op.
        """
        return None

    @abstractmethod
    def __call__(self, arr: Array, axis: int = 0) -> Array:
        """
        Applies the combination to an array on an axis, eliminating it. a1 *rest -> *rest.
        """
        raise NotImplementedError


class Transformation(metaclass=ABCMeta):
    """
    A function with signature f(Array, axis: int) → Array. Examples are softmax, normalization,
    cumulative sum.
    """

    @classmethod
    def parse(cls, _name: str) -> Optional['Transformation']:
        """
        Attempts to construct an instance of the operation from the string name. If unsuccessful,
        returns None.
        """
        return None

    @abstractmethod
    def __call__(self, arr: Array, axis: int = 0) -> Array:
        """Applies the transformation to an array on an axis."""
        raise NotImplementedError


class ElementwiseOp(metaclass=ABCMeta):
    """
    Elementwise operation on scalars that can map to arrays of any shape. Signature f(Array) →
    Array.
    """

    @classmethod
    def parse(cls, _name: str) -> Optional['ElementwiseOp']:
        """
        Attempts to construct an instance of the operation from the string name. If unsuccessful,
        returns None. Used for shorthand syntax, like passing in 'sqrt' instead of the named Sqrt
        op.
        """
        return None

    @abstractmethod
    def __call__(self, arr: Array) -> Array:
        """Applies the function elementwise."""
        raise NotImplementedError


class Combination(metaclass=ABCMeta):
    """
    Operation to combine array values together. Commutative, associative function of signature
    f(Array, Array) → Array.
    """

    @classmethod
    def parse(cls, _name: str) -> Optional['Combination']:
        """
        Attempts to construct an instance of the operation from the string name. If unsuccessful,
        returns None. Used for shorthand syntax, like passing in 'sqrt' instead of the named Sqrt
        op.
        """
        return None

    @abstractmethod
    def __call__(self, arr1: Array, arr2: Array) -> Array:
        """Applies the function to the two inputs, returning a single output."""
        raise NotImplementedError


CombinationFunc = Callable[[Array, Array], Array]
ReductionFunc = Callable[..., Array]
TransformationFunc = Callable[..., Array]
ElementwiseFunc = Callable[[Array], Array]
