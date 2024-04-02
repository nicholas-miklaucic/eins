"""Symbolic representation of tensor operations and manipulations."""

import functools as ft
import pprint
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Mapping, MutableSequence, Sequence, Union, cast

from eins.combination import ArrayCombination, Combination
from eins.constraint import Constraints, postprocess_ast
from eins.parsing import Constant, Expr, Node, Symbol, make_expr, unpack_shorthands
from eins.parsing import expr as expr_parser
from eins.reduction import ArrayReduction, Reduction


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
class Combine(ShapeOp):
    method: Combination

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


# Ruff *really* does not like magic numbers!
EXP_ARITY = 2


def expanded_shape(node: Node) -> Sequence[Node]:
    if isinstance(node, (Constant, Symbol)):
        return [node]
    elif node.op == '*':
        return list(chain.from_iterable(map(expanded_shape, node.children)))
    elif node.op == '^':
        if not (len(node.children) == EXP_ARITY and isinstance(node.children[1], Constant)):
            msg = f'Unexpected exponent {node.children[1]}'
            raise ValueError(msg)

        return list(
            chain.from_iterable(
                [expanded_shape(node.children[0]) for _ in range(node.children[1].value)]
            )
        )
    elif node.op == '+':
        return [node]

    msg = f'Unexpected node {node}, {node.op!r}'
    raise ValueError(msg)


def normalize_step(tensor: Tensor) -> Sequence[Tensor]:
    # tensor with axes
    if tensor.idx_axis is None:
        for i, ax in enumerate(tensor.axes):
            if isinstance(ax, Expr) and ax.op in ('*', '^'):
                # simple flatten
                new_shape = (
                    list(deepcopy(tensor.axes[:i]))
                    + list(expanded_shape(ax))
                    + list(deepcopy(tensor.axes[i + 1 :]))
                )
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
    if len(root.children) > 1:
        msg = f'Cannot reverse {root} because it has more than one child'
        raise ValueError(msg)

    if not root.children:
        return root

    op, children = root.children[0]
    if isinstance(op, Reshape):
        return Reshape(root.axes)([reverse_graph(children[0])])[0]
    elif isinstance(op, Split):
        return Concat(op.axis_num)([reverse_graph(child) for child in children])[0]
    else:
        raise ValueError


DEFAULT_COMBINE = ArrayCombination('multiply')
DEFAULT_REDUCE = ArrayReduction('sum')
# *rolls eyes*
MAX_ARROW_OPERANDS = 2


class Program:
    def __init__(
        self,
        expr: Expr,
        constr: Constraints,
        combine: Combination = DEFAULT_COMBINE,
        reduce: Union[Reduction, Mapping[str, Reduction]] = DEFAULT_REDUCE,
        reduce_early: bool = True,  # noqa: FBT001, FBT002
    ):
        self.reduce_early = reduce_early
        self.graph = {}
        if expr.op != '->':
            msg = f'Expected -> as operator, but got {expr.op}'
            raise ValueError(msg)

        if len(expr.op) != MAX_ARROW_OPERANDS:
            msg = f'For now, Eins only supports one -> per expression, but got {len(expr.op)}'
            raise ValueError(msg)

        lhs, rhs = expr.children

        if not isinstance(lhs, Expr):
            lhs = Expr(' ', cast(MutableSequence[Node], [lhs]))

        if not isinstance(rhs, Expr):
            rhs = Expr(' ', cast(MutableSequence[Node], [rhs]))

        if lhs.op == ',':
            self.sources = [Tensor(c) for c in lhs.children]
        else:
            self.sources = [Tensor(lhs)]

        self.current = []
        for source in self.sources:
            self.current.extend(normalize_until_done(source))

        self.orig_sink = Tensor(rhs)
        self.sinks = list(normalize_until_done(self.orig_sink))
        self.constr = constr

        for t in self.current + self.sinks:
            self.constr.add_variables(t.axes_list())

        self.combine = combine

        if isinstance(reduce, Reduction):
            self.reduce = defaultdict(lambda: reduce)
        else:
            self.reduce = dict(reduce)
        if not isinstance(reduce, Reduction):
            for k, v in reduce.items():
                self.reduce[k] = v

        self.outputs = []
        if len(self.sinks) > 1:
            # multiple tensors in output this is the one time the rules are relaxed about using all
            # inputs tensors that only have an axis in a different split aren't used e.g., a b, a c
            # -> a b+c just does a b -> a b, a c -> a c
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
                    # TODO needs to be more robust for multiple splits? it can't be on the "wrong
                    # side" of a single split a c, a d, b c, b d -> a+b c+d
                    curr_ax = set(curr.axes_list())
                    if curr_ax.isdisjoint(all_splits - diff):
                        per_sink_input.append(curr)
                per_sink_inputs.append(per_sink_input)

            # print(per_sink_inputs) print(self.sinks)
            for sink_input, sink in zip(per_sink_inputs, self.sinks):
                self.outputs.append(self.connect(sink_input, sink))

        else:
            self.outputs.append(self.connect(self.current, self.sinks[0]))

        reverse_graph(self.orig_sink)

        for out, sink in zip(self.outputs, self.sinks):
            if not out.is_same_shape(sink):
                msg = f'Output {out} is not the same shape as sink {sink}'
                raise ValueError(msg)

            out.children = sink.children
            for _op, children in out.children:
                for child in children:
                    for i, parent in enumerate(child.parents):
                        if id(parent) == id(sink):
                            child.parents[i] = out

    def apply_op(self, op: ShapeOp, tensors: Union[Tensor, Sequence[Tensor]]):
        if isinstance(tensors, Tensor):
            tensors = [tensors]

        key = (op, tuple(map(id, tensors)))
        if key in self.graph:
            return self.graph[key]
        else:
            res = op(tensors)
            self.graph[key] = res
            return res

    def combine_mismatched(self, combine: Combine, tensors: Sequence[Tensor]) -> Tensor:
        """Combines tensors together, broadcasting and transposing as needed. Assumes tensors are in
        normal form."""
        out = tensors[0]
        for t in tensors[1:]:
            new_axes = out.axes_list() + [ax for ax in t.axes_list() if ax not in out.axes_list()]
            axs = t.axes_list()
            perm = tuple(sorted(range(len(axs)), key=lambda x: new_axes.index(axs[x])))
            transposed_t = Transpose(perm)(t)[0]
            new_ax_op = ExpandTo(tuple(map(Symbol, new_axes)))
            expanded_t = self.apply_op(new_ax_op, transposed_t)[0]
            expanded_out = self.apply_op(new_ax_op, out)[0]
            out = self.apply_op(combine, (expanded_out, expanded_t))[0]
        return out

    @classmethod
    def parse(cls, op: str, **kwargs):
        expr = make_expr(expr_parser.parse_string(unpack_shorthands(op)).as_list())
        if not isinstance(expr, Expr):
            msg = f'Expected expression, but got {expr}'
            raise ValueError(msg)
        constraints = postprocess_ast(expr)
        return cls(expr, constraints, **kwargs)

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
            transposed.extend(self.apply_op(Transpose(perm), s))

        if self.reduce_early:
            exp_axes = [set(exp.axes_list()) for exp in transposed]
            combined = transposed[0]
            for i, exp in enumerate(transposed):
                if i == 0:
                    continue
                else:
                    combined = self.combine_mismatched(Combine(self.combine), (combined, exp))

                in_rest = set()
                for remaining_axes in exp_axes[i + 1 :]:
                    in_rest |= remaining_axes

                # axes in the input, to reduce, that aren't in future tensors
                to_reduce = (reduce_axes - in_rest) & set(combined.axes_list())
                for r_ax in to_reduce:
                    combined = self.apply_op(Reduce(self.reduce[r_ax], Symbol(r_ax)), combined)[0]
            reduced = combined
        else:
            expanded = []
            for t in transposed:
                expanded.extend(self.apply_op(ExpandTo(tuple(map(Symbol, start_axes))), t))
            combined = self.apply_op(Combine(self.combine), tuple(expanded))[0]

            reduced = combined
            for ax in reduce_axes:
                reduced = self.apply_op(Reduce(self.reduce[ax], Symbol(ax)), reduced)[0]

        r_axs = reduced.axes_list()
        if set(r_axs) != set(goal_axes):
            msg = f'{r_axs} != {goal_axes}'
            raise ValueError(msg)

        perm = tuple(sorted(range(len(r_axs)), key=lambda x: goal_axes.index(r_axs[x])))

        out = self.apply_op(Transpose(perm), reduced)[0]

        if out.axes_list() != goal_axes:
            msg = f'{out.axes_list()} != {goal_axes}'
            raise ValueError(msg)

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


# env = Program.parse('b ((n p) (n p)) c d=c, b p*p*d*c h, h[g+i k] -> b (n^2 g+i) k') env =
# Program.parse('a b, b c -> a+c b') print(env)


# ops to reverse graph from sink to targets just get it fing done
