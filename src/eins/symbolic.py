"""Symbolic representation of tensor operations and manipulations."""

import pprint
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import MutableSequence, Sequence, Union

from eins.combination import Combination
from eins.common_types import Transformation
from eins.parsing import Constant, Expr, Node, Symbol
from eins.reduction import Reduction


class Tensor:
    """Node in a tensor network."""

    def __init__(self, expr: Node):
        self.parents = []
        self.children = []
        self.idx_axis = None
        self.axes: MutableSequence[Node] = []
        if isinstance(expr, (Constant, Symbol)):
            self.axes.append(expr)
            return

        if expr.op == ' ':
            self.axes = expr.children
        elif expr.op == '@':
            self.axes = Tensor(expr.children[0]).axes
            self.idx_axis = expr.children[1]
        else:
            self.axes.append(expr)

    def deepcopy(self) -> 'Tensor':
        t = Tensor(Expr(' ', deepcopy(self.axes)))
        t.idx_axis = self.idx_axis
        return t

    def axes_list(self) -> 'list[str]':
        if any(not isinstance(ax, Symbol) for ax in self.axes):
            msg = f'All axes must be symbols, but got {pprint.pformat(self.axes)}'
            raise ValueError(msg)
        return [ax.value for ax in self.axes if isinstance(ax, Symbol)]

    def add_child_op(self, children: 'Sequence[Tensor]', op: 'ShapeOp'):
        for child in children:
            child.parents.append(self)
        self.children.append((op, children))

    def is_same_shape(self, other: 'Tensor') -> bool:
        return self.axes == other.axes and self.idx_axis == other.idx_axis

    def __repr__(self):
        idx_expr = '' if self.idx_axis is None else f' @ {self.idx_axis}'
        ax_expr = ' '.join(map(str, self.axes))
        return ax_expr + idx_expr


class ShapeOp(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        raise NotImplementedError

    def __call__(self, tensors: Union[Tensor, Sequence[Tensor]]) -> Sequence[Tensor]:
        if isinstance(tensors, Tensor):
            tensors = [tensors]

        if self.is_identity_for(tensors):
            return tensors

        children = self.apply(tensors)

        for t in tensors:
            t.add_child_op(children, self)
        return children

    def is_identity_for(self, _tensors: Sequence[Tensor]) -> bool:
        return False


@dataclass(unsafe_hash=True)
class Reshape(ShapeOp):
    new_shape: Sequence[Node]

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:  # noqa: ARG002
        return [Tensor(Expr(' ', list(self.new_shape)))]


@dataclass(unsafe_hash=True)
class Transpose(ShapeOp):
    perm: Sequence[int]

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [Tensor(Expr(' ', [tensors[0].axes[i] for i in self.perm]))]

    def is_identity_for(self, _tensors: Sequence[Tensor]) -> bool:
        return tuple(self.perm) == tuple(range(len(self.perm)))


@dataclass(unsafe_hash=True)
class Split(ShapeOp):
    axis_num: int

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        children = []
        split_ax = tensors[0].axes[self.axis_num]
        if not isinstance(split_ax, Expr) or split_ax.op != '+':
            msg = f'Tried to split on {split_ax} in {tensors}, which is not a sum.'
            raise ValueError(msg)

        for addend in split_ax.children:
            child = tensors[0].deepcopy()
            child.axes[self.axis_num] = addend
            children.append(child)
        return children


@dataclass(unsafe_hash=True)
class Concat(ShapeOp):
    axis_num: int

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        add_axes = [tensor.axes[self.axis_num] for tensor in tensors]
        concatenated = tensors[0].deepcopy()
        concatenated.axes[self.axis_num] = Expr('+', add_axes)
        return [concatenated]


@dataclass(unsafe_hash=True)
class OneHot(ShapeOp):
    idx_axis: Node

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        child = tensors[0].deepcopy()
        if child.idx_axis is None:
            msg = f'Cannot one-hot on {child} because it does not have an index axis'
            raise ValueError(msg)
        child.axes.append(child.idx_axis)
        child.idx_axis = None
        return [child]


@dataclass(unsafe_hash=True)
class ExpandTo(ShapeOp):
    """Adds new axes with 1 to broadcast with the current shape. Output should be
    a supersequence of the current shape."""

    new_shape: Sequence[Node]

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:  # noqa: ARG002
        return [Tensor(Expr(' ', list(self.new_shape)))]


@dataclass(unsafe_hash=True)
class Tile(ShapeOp):
    """Repeats values along axes to go from an array broadcastable with a shape to an array that
    exactly matches a shape."""

    new_shape: Sequence[Node]

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:  # noqa: ARG002
        return [Tensor(Expr(' ', list(self.new_shape)))]


@dataclass(unsafe_hash=True)
class Combine(ShapeOp):
    method: Combination

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [tensors[0].deepcopy()]

    def is_identity_for(self, tensors: Sequence[Tensor]) -> bool:
        return len(tensors) == 1


@dataclass(unsafe_hash=True)
class Transform(ShapeOp):
    method: Transformation
    axis: Node

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [tensors[0].deepcopy()]

    def is_identity_for(self, tensors: Sequence[Tensor]) -> bool:
        return self.axis not in tensors[0].axes


@dataclass(unsafe_hash=True)
class Reduce(ShapeOp):
    method: Reduction
    axis: Node

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        out = tensors[0].deepcopy()
        out.axes = [ax for ax in out.axes if ax != self.axis]
        return [out]


# env = Program.parse('b ((n p) (n p)) c d=c, b p*p*d*c h, h[g+i k] -> b (n^2 g+i) k') env =
# Program.parse('a b, b c -> a+c b') print(env)


# ops to reverse graph from sink to targets just get it fing done
