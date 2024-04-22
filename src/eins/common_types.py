"""Type definitions"""

from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union

from typing_extensions import Protocol, Self, TypeVar


class Array(Protocol):
    """
    An array protocol: any class implementing these operations can be used in Eins.
    """

    def __add__(self, __other: Union[Self, int, float]) -> Self: ...
    def __sub__(self, __other: Union[Self, int, float]) -> Self: ...
    def __mul__(self, __other: Union[Self, int, float]) -> Self: ...
    def __truediv__(self, __other: Union[Self, int, float]) -> Self: ...
    def __floordiv__(self, __other: Union[Self, int, float]) -> Self: ...
    def __pow__(self, __other: Union[Self, int, float]) -> Self: ...
    def __mod__(self, __other: Union[Self, int, float]) -> Self: ...

    def __eq__(self, __other: Union[Self, int, float]) -> Self: ...  # type: ignore
    def __ne__(self, __other: Union[Self, int, float]) -> Self: ...  # type: ignore
    def __lt__(self, __other: Union[Self, int, float]) -> Self: ...
    def __le__(self, __other: Union[Self, int, float]) -> Self: ...
    def __ge__(self, __other: Union[Self, int, float]) -> Self: ...
    def __gt__(self, __other: Union[Self, int, float]) -> Self: ...

    def __neg__(self) -> Self: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def __getitem__(self, __key) -> Self: ...


Arr = TypeVar('Arr', bound=Array)
Axis = Union[None, int]


class ArrayBackend(Protocol[Arr]):
    """
    Generic array backend over arrays of type A. Allows structural subtyping for the many machine
    learning libraries that don't explicitly implement a common backend.
    """

    def sum(self, __arr: Arr, /, *, axis: Axis) -> Arr: ...
    def prod(self, __arr: Arr, axis: Axis) -> Arr: ...
    def max(self, __arr: Arr, axis: Axis) -> Arr: ...
    def min(self, __arr: Arr, axis: Axis) -> Arr: ...
    def mean(self, __arr: Arr, axis: Axis) -> Arr: ...
    def std(self, __arr: Arr, axis: Axis) -> Arr: ...
    def var(self, __arr: Arr, axis: Axis) -> Arr: ...

    def sort(self, __arr: Arr, axis: Axis) -> Arr: ...

    def expand_dims(self, __arr: Arr, axis: Union[int, Sequence[int]]) -> Arr: ...

    # Combinations
    def minimum(self, __arr1: Arr, __arr2: Union[int, float, Arr]) -> Arr: ...
    def maximum(self, __arr1: Arr, __arr2: Union[int, float, Arr]) -> Arr: ...

    # Elementwise ops
    def abs(self, __arr: Arr) -> Arr: ...
    def sqrt(self, __arr: Arr) -> Arr: ...
    def square(self, __arr: Arr) -> Arr: ...
    def exp(self, __arr: Arr) -> Arr: ...
    def log(self, __arr: Arr) -> Arr: ...


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


class Reduction(metaclass=ABCMeta):
    """
    A function with signature f(Array, axis: int) → Array that removes the axis. Common examples:
    sum, product, mean, norm.
    """

    def fold_of(self) -> Optional[Combination]:
        """
        If this reduction can be thought of mathematically as a fold—for instance, sum can be
        thought of as a folded add, and logsumexp can be thought of as a folded logaddexp, returns
        that operation. If not (e.g., median), then returns None.

        Used for determining how Eins can order operations when compiling expressions.
        """
        # None won't ever lead to problems: it's safe, if slow
        return None

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

    def commutes_with(self, op: Combination) -> bool:  # noqa: ARG002
        """Returns whether op(f(x), f(y)) == f(op(x, y))."""
        # by default, assume it doesn't: this is safe.
        return False

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


T = TypeVar('T')


class CombinationFunc(Protocol[T]):
    def __call__(self, __arr: T, __arr2: T) -> T: ...


class ElementwiseFunc(Protocol[T]):
    def __call__(self, __arr: T) -> T: ...


class TransformationFuncAxis(Protocol[T]):
    def __call__(self, __arr: T, axis: int) -> T: ...


class ReductionFuncAxis(Protocol[T]):
    def __call__(self, __arr: T, axis: int) -> T: ...


class TransformationFuncDim(Protocol[T]):
    def __call__(self, __arr: T, dim: int) -> T: ...


class ReductionFuncDim(Protocol[T]):
    def __call__(self, __arr: T, dim: int) -> T: ...


TransformationFunc = Union[TransformationFuncAxis, TransformationFuncDim]
ReductionFunc = Union[ReductionFuncAxis, ReductionFuncDim]


def call_axis_func(f: Union[TransformationFunc, ReductionFunc], arr: Arr, axis: int = 0) -> Arr:
    try:
        return f(arr, axis=axis)  # type: ignore
    except AttributeError:
        return f(arr, dim=axis)  # type: ignore
