# Ruff *really* does not like magic numbers!
from collections import defaultdict
from copy import deepcopy
from itertools import chain
import pprint
from typing import Mapping, MutableSequence, Sequence, Union, cast

from eins.combination import ArrayCombination
from eins.common_types import Combination, Reduction
from eins.constraint import Constraints, postprocess_ast
from eins.parsing import Constant, Expr, Symbol, Node, make_expr, unpack_shorthands
from eins.reduction import ArrayReduction
from eins.symbolic import (
    Combine,
    Concat,
    ExpandTo,
    OneHot,
    Reduce,
    Reshape,
    ShapeOp,
    Split,
    Tensor,
    Transpose,
    expr_parser,
)


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

    def make_path(self, strat):
        self.outputs = []
        if len(self.sinks) > 1:
            # multiple tensors in output this is the one time the rules are relaxed about using all
            # inputs tensors that only have an axis in a different split aren't used e.g., a b, a c
            # -> a b+c just does a b -> a b and a c -> a c
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
                self.outputs.append(strat.connect(sink_input, sink))

        else:
            self.outputs.append(strat.connect(self.current, self.sinks[0]))

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

    @classmethod
    def parse(cls, op: str, **kwargs):
        expr = make_expr(expr_parser.parse_string(unpack_shorthands(op)).as_list())
        if not isinstance(expr, Expr):
            msg = f'Expected expression, but got {expr}'
            raise ValueError(msg)
        constraints = postprocess_ast(expr)
        return cls(expr, constraints, **kwargs)

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
