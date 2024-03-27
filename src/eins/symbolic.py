"""Symbolic representation of tensor operations and manipulations."""

import functools as ft
import pprint
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Sequence, Set, Tuple, Union

from eins.parsing import Constant, Constraints, Expr, Node, Symbol, make_expr, postprocess_ast, unpack_shorthands
from eins.parsing import expr as expr_parser
from eins.reduction import Array, Prod, Reduction, Sum


class Tensor:
    """Node in a tensor network."""

    def __init__(self, expr: Node):
        self.parents = []
        self.children = []
        self.idx_axis = None
        if expr.op == ' ':
            self.axes = expr.children
        elif expr.op == '@':
            self.axes = Tensor(expr.children[0]).axes
            self.idx_axis = expr.children[1]
        else:
            self.axes = [expr]

    def deepcopy(self) -> 'Tensor':
        t = Tensor(Expr(' ', deepcopy(self.axes)))
        t.idx_axis = self.idx_axis
        return t

    def axes_list(self) -> Sequence[str]:
        return [ax.value if isinstance(ax, Symbol) else print('uh', ax, type(ax)) for ax in self.axes]

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
    new_shape: tuple[Node]

    def apply(self, _tensor: Sequence[Tensor]) -> Sequence[Tensor]:
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
        for addend in tensors[0].axes[self.axis_num].children:
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
        child.axes.append(child.idx_axis)
        child.idx_axis = None
        return [child]


@dataclass(unsafe_hash=True)
class ExpandTo(ShapeOp):
    """Adds new axes with 1 to broadcast with the current shape. Output should be
    a supersequence of the current shape."""

    new_shape: tuple[Node]

    def apply(self, _tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [Tensor(Expr(' ', list(self.new_shape)))]


@dataclass(unsafe_hash=True)
class Combine(ShapeOp):
    method: Reduction

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [tensors[0].deepcopy()]

    def is_identity_for(self, tensors: Sequence[Tensor]) -> bool:
        return len(tensors) == 1


@dataclass(unsafe_hash=True)
class Reduce(ShapeOp):
    method: Reduction
    axis: Node

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        out = tensors[0].deepcopy()
        out.axes = [ax for ax in out.axes if ax != self.axis]
        return [out]


def expanded_shape(node: Node) -> Sequence[Node]:
    if isinstance(node, (Constant, Symbol)):
        return [node]
    else:
        match node.op:
            case '*':
                return list(chain.from_iterable(map(expanded_shape, node.children)))
            case '^':
                return list(
                    chain.from_iterable([expanded_shape(node.children[0]) for _ in range(node.children[1].value)])
                )
            case '+':
                return [node]


def normalize_step(tensor: Tensor) -> Sequence[Tensor]:
    # tensor with axes
    if tensor.idx_axis is None:
        for i, ax in enumerate(tensor.axes):
            if isinstance(ax, Expr) and ax.op in ('*', '^'):
                # simple flatten
                new_shape = deepcopy(tensor.axes[:i]) + expanded_shape(ax) + deepcopy(tensor.axes[i + 1 :])
                return Reshape(new_shape)(tensor)
            elif isinstance(ax, Expr) and ax.op == '+':
                # split
                return Split(i)(tensor)
    else:
        return OneHot(tensor.idx_axis)(tensor)

    # if we get here, no change: in normal form
    return []


def normalize_until_done(tensor: Tensor) -> Sequence[Tensor]:
    outs = [tensor]
    done = False
    while not done:
        new_outs = []
        done = True
        for out in outs:
            normalized = normalize_step(out)
            if normalized:
                # did something
                new_outs.extend(normalized)
                done = False
            else:
                new_outs.append(out)

        outs = new_outs
    return outs


def reverse_graph(root: Tensor):
    assert len(root.children) <= 1

    if not root.children:
        return root

    op, children = root.children[0]
    if isinstance(op, Reshape):
        return Reshape(root.axes)(reverse_graph(children[0]))
    elif isinstance(op, Split):
        return Concat(op.axis_num)([reverse_graph(child) for child in children])
    else:
        raise ValueError


class Program:
    def __init__(
        self,
        expr: Expr,
        constr: Constraints,
        combine: Reduction = Prod,
        reduce: Reduction | Mapping[str, Reduction] = Sum,
    ):
        assert expr.op == '->' and len(expr.children) == 2
        lhs, rhs = expr.children

        if lhs.op == ',':
            self.sources = [Tensor(c) for c in lhs.children]
        else:
            self.sources = [Tensor(lhs)]

        self.current = []
        for source in self.sources:
            self.current.extend(normalize_until_done(source))

        self.orig_sink = Tensor(rhs)
        self.sinks = normalize_until_done(self.orig_sink)
        self.constr = constr
        self.combine = combine

        self.reduce = defaultdict(lambda: reduce)
        if not isinstance(reduce, Reduction):
            for k, v in reduce.items():
                self.reduce[k] = v

        self.outputs = []
        if len(self.sinks) > 1:
            # multiple tensors in output
            # this is the one time the rules are relaxed about using all inputs
            # tensors that only have an axis in a different split aren't used
            # e.g., a b, a c -> a b+c just does a b -> a b, a c -> a c
            axes_sets = [set(sink.axes_list()) for sink in self.sinks]

            all_splits = set()
            diffs = []
            for i, ax_set in enumerate(axes_sets):
                others = set()
                for j, ax_set2 in enumerate(axes_sets):
                    if i == j:
                        continue
                    others |= ax_set2
                diff = ax_set - others
                diffs.append(diff)
                all_splits |= diff

            per_sink_inputs = []
            for diff in diffs:
                per_sink_input = []
                for curr in self.current:
                    if diff <= set(curr.axes_list()) or not (all_splits <= set(curr.axes_list())):
                        per_sink_input.append(curr)
                per_sink_inputs.append(per_sink_input)

            for sink_input, sink in zip(per_sink_inputs, self.sinks):
                self.outputs.append(self.connect(sink_input, sink))

            reverse_graph(self.orig_sink)
        else:
            self.outputs.append(self.connect(self.current, self.sinks[0]))

    @ft.lru_cache
    @staticmethod
    def apply_op(op: ShapeOp, tensors: Union[Tensor, Sequence[Tensor]]):
        return op(tensors)

    @classmethod
    def parse(cls, op: str):
        expr = make_expr(expr_parser.parse_string(unpack_shorthands(op)).as_list())
        constraints = postprocess_ast(expr)
        return cls(expr, constraints)

    def connect(self, start: Sequence[Tensor], goal: Tensor):
        goal_axes = goal.axes_list()
        start_axes = start[0].axes_list()
        for other in start[1:]:
            for ax in other.axes_list():
                if ax not in start_axes:
                    start_axes.append(ax)

        reduce_axes = set(start_axes) - set(goal_axes)

        transposed = []
        for s in start:
            axs = s.axes_list()
            perm = tuple(sorted(range(len(axs)), key=lambda x: start_axes.index(axs[x])))
            transposed.extend(Program.apply_op(Transpose(perm), s))

        expanded = []
        for t in transposed:
            expanded.extend(Program.apply_op(ExpandTo(tuple(map(Symbol, start_axes))), t))

        combined = Program.apply_op(Combine(self.combine), tuple(expanded))[0]

        reduced = combined
        for ax in reduce_axes:
            reduced = Program.apply_op(Reduce(self.reduce[ax], Symbol(ax)), reduced)[0]

        r_axs = reduced.axes_list()
        assert set(r_axs) == set(goal_axes), f'{r_axs} != {goal_axes}'

        perm = tuple(sorted(range(len(r_axs)), key=lambda x: goal_axes.index(r_axs[x])))

        out = Program.apply_op(Transpose(perm), reduced)[0]

        assert out.axes_list() == goal_axes, f'{out.axes_list()} != {goal_axes}'

        return out

    def __repr__(self):
        strings = []
        for i, inp in enumerate(self.sources):
            strings.append(f'Input {i+1}:\n{pprint.pformat(inp)}')
        strings.append('-' * 50)
        for i, inp in enumerate(self.current):
            strings.append(f'Current {i+1}:\n{pprint.pformat(inp)}')
        strings.append('-' * 50)
        for i, sink in enumerate(self.sinks):
            strings.append(f'Output {i+1}:\n{pprint.pformat(sink)}')
        strings.append('-' * 50)
        strings.append(f'Final Output:\n{pprint.pformat(self.orig_sink)}')
        strings.append('-' * 50)
        strings.append('Equations:')
        strings.append(str(self.constr))
        return '\n'.join(strings)


env = Program.parse('b ((n p) (n p)) c d=c, b p*p*d*c h, h[g+i k] -> b (n^2 g+i) k')
# env = Program.parse('a b, b c -> a c')
# print(env)


# ops to reverse graph from sink to targets
# just get it fing done
