# Ruff *really* does not like magic numbers!
import pprint
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Mapping, MutableSequence, Sequence, Tuple, Union, cast

from eins.combination import ArrayCombination
from eins.common_types import Combination, Reduction, Transformation
from eins.constraint import Constraints, postprocess_ast
from eins.parsing import Constant, Expr, Node, Symbol, make_expr, unpack_shorthands
from eins.parsing import expr as expr_parser
from eins.reduction import ArrayReduction
from eins.symbolic import (
    Concat,
    OneHot,
    Reshape,
    ShapeOp,
    Split,
    Tensor,
    Transform,
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
    ):
        self.graph = {}
        self.expr = expr
        if expr.op != '->':
            msg = 'Eins expressions must have -> unless they are a transformation.'
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
        self.source_curr_map = {}
        for i, source in enumerate(self.sources):
            currents = normalize_until_done(source)
            self.current.extend(currents)
            self.source_curr_map[i] = currents

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
            # we may have already renamed axes, but the user is passing in the old names
            # e.g., a b b, c b b -> a, reduce={'b': 'mean', 'c': 'sum'}
            # b now is b-1 and b-2
            for lhs, rhs in self.constr.equations:
                if isinstance(rhs, Symbol) and isinstance(lhs, Symbol):
                    if rhs.value in self.reduce:
                        self.reduce[lhs.value] = self.reduce[rhs.value]
                    elif lhs.value in self.reduce:
                        self.reduce[rhs.value] = self.reduce[lhs.value]

        if not isinstance(reduce, Reduction):
            for k, v in reduce.items():
                self.reduce[k] = v

    def make_path(self, strat):
        self.outputs = []
        # Normally, every input is used for every output. The exception is when there are splits in
        # the input. For example, a b+c -> a b doesn't use a c. The rule is that a tensor with a
        # split input is unused for an output only if it was split from a sum, of which at least one
        # axis is in the output, but the specific split we're dealing with isn't.

        # a c, b c -> a+b c: all used
        # a+b c -> a c: only a c used, not b c, because a in {a, c} but b not in {a c}
        # a+b c, a+b d -> a c+d: a c used for a c, a d used for a d

        # TODO very challenging case: a+b c, a+b d, e a -> e c+d

        # only necessary if we split at some point, so either multiple outputs or multiple inputs
        # from a single source
        if len(self.sinks) > 1 or len(self.current) != len(self.sources):
            sink_axes = [set(sink.axes_list()) for sink in self.sinks]
            common_sink_axes = set.intersection(*sink_axes)
            split_sink_axes = [ax - common_sink_axes for ax in sink_axes]
            missing_sink_axes = [common_sink_axes - ax for ax in sink_axes]

            for sink_i, (axs, split, missing) in enumerate(
                zip(sink_axes, split_sink_axes, missing_sink_axes)
            ):
                sink_inputs = []
                for source_i, currs in self.source_curr_map.items():
                    curr_axes = [set(curr.axes_list()) for curr in currs]
                    common_curr_axes = set.union(*curr_axes) - set.intersection(*curr_axes)
                    split_curr_axes = [ax & common_curr_axes for ax in curr_axes]
                    missing_curr_axes = [common_curr_axes - ax for ax in curr_axes]
                    for curr, curr_axs, curr_split, curr_missing in zip(
                        currs, curr_axes, split_curr_axes, missing_curr_axes
                    ):
                        # print(currs, curr_axes, curr_split, curr_missing, axs)
                        add_to_input = False
                        if curr_split <= axs and curr_missing.isdisjoint(axs):
                            # split axes line up with output
                            add_to_input = True
                        elif curr_missing <= axs and curr_split.isdisjoint(axs):
                            # other axes line up with output, skip
                            add_to_input = False
                        else:
                            # e.g., a+b, c a, d b, d e -> c e

                            # Neither a nor b appear in output. Probably not solvable in general,
                            # but we can at least try a basic heuristic: if the split axis appears
                            # in another input, include it
                            other_axes = []
                            for source_j, other_currs in self.source_curr_map.items():
                                if source_i != source_j:
                                    other_axes.extend(
                                        [set(curr.axes_list()) for curr in other_currs]
                                    )
                            other_axes = set.union(*other_axes)
                            # print(other_axes)
                            if curr_missing.isdisjoint(other_axes) and curr_split <= other_axes:
                                add_to_input = True
                            elif curr_missing <= other_axes and curr_split.isdisjoint(other_axes):
                                add_to_input = False
                            else:
                                msg = (
                                    f'Unclear split: {axs}, {split}, {missing} '
                                    f'{curr_axs}, {curr_split}, {curr_missing}'
                                )
                                raise ValueError(msg)

                        if add_to_input:
                            sink_inputs.append(curr)

                # print(sink_inputs, self.sinks[sink_i])
                self.outputs.append(strat.connect(sink_inputs, self.sinks[sink_i]))

        else:
            self.outputs.append(strat.connect(self.current, self.sinks[0]))

        self.link_outs_to_sink()

    def link_outs_to_sink(self):
        """
        Connects the outputs (from the sources) to the sink.
        """
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


class TransformProgram(Program):
    """
    Program that applies transformations to the input tensor without modifying the shape.
    """

    def __init__(self, expr: Node, constr: Constraints, transform: Mapping[str, Transformation]):
        if not isinstance(expr, Expr):
            expr = Expr(' ', cast(MutableSequence[Node], [expr]))

        if expr.op in (',', '->'):
            msg = 'Eins transformations only take in a single input tensor.'
            raise ValueError(msg)

        self.constr = constr
        self.graph = {}
        self.expr = expr

        self.sources = [Tensor(expr)]
        self.current = list(normalize_until_done(self.sources[0]))

        for t in self.current:
            self.constr.add_variables(t.axes_list())

        self.transform = transform

        self.orig_sink = Tensor(expr)
        self.sinks = list(normalize_until_done(self.orig_sink))

        for lhs, rhs in self.constr.equations:
            if isinstance(rhs, Symbol) and isinstance(lhs, Symbol):
                if lhs.value in self.transform or rhs.value in self.transform:
                    val = lhs.value if lhs.value in self.transform else rhs.value
                    msg = (
                        f"It's not clear which {val} axis you want to apply the transformation to."
                        " Specify by giving the duplicates different names, e.g., 'a b=a' instead"
                        " of 'a a'."
                    )
                    raise ValueError(msg)

    def make_path(self, strat=None):  # noqa: ARG002
        self.transform_map = {}
        used_axes = set()

        for i, curr in enumerate(self.current):
            for ax in curr.axes_list():
                if ax in self.transform:
                    if ax in used_axes:
                        msg = (
                            f'Axis {ax} appears on both sides of a split axis. '
                            'It is ambiguous whether transformation should happen'
                            ' before or after the split.'
                        )
                        raise ValueError(msg)

                    if curr in self.transform_map:
                        msg = (
                            f'Multiple transforms could be applied to tensor {curr}.'
                            ' Split the operation into multiple to ensure consistent order.'
                        )

                    self.transform_map[i] = Transform(self.transform[ax], Symbol(ax))
                    used_axes.add(ax)

        self.outputs = []
        for i, curr in enumerate(self.current):
            if i in self.transform_map:
                out = self.apply_op(self.transform_map[i], curr)[0]
            else:
                out = curr
            self.outputs.append(out)

        self.link_outs_to_sink()
