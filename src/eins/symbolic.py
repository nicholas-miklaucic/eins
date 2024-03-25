"""Symbolic representation of tensor operations and manipulations."""

import pprint
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Sequence, Union

from eins.parsing import Constant, Constraints, Expr, Node, Symbol, make_expr, postprocess_ast, unpack_shorthands
from eins.parsing import expr as expr_parser
from eins.reduction import Prod, Reduction, Sum


class ShapeOp(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, tensor: Expr) -> Expr:
        raise NotImplementedError


@dataclass
class ReshapeOp(ShapeOp):
    new_shape: tuple[Node]

    def __call__(self, _tensor: Expr) -> Expr:
        return Expr(' ', list(self.new_shape))


@dataclass
class TransposeOp(ShapeOp):
    perm: tuple[int]

    def __call__(self, tensor: Expr) -> Expr:
        return Expr(' ', [tensor.children[i] for i in self.perm])


@dataclass
class SplitOp(ShapeOp):
    axis_num: int

    def __call__(self, tensor: Expr) -> Expr:
        if tensor.op == '+':
            assert self.axis_num == 0
            return Expr(',', tensor.children)
        else:
            assert tensor.children[self.axis_num].op == '+'
            splits = []
            for child in tensor.children[self.axis_num].children:
                split = Expr(' ', tensor.children)
                split.children[self.axis_num] = child
                splits.append(split)
            return Expr(',', splits)


@dataclass
class OneHotOp(ShapeOp):
    def __call__(self, tensor: Expr) -> Expr:
        assert tensor.op == '@'
        assert len(tensor.children) == 2
        shape, idx_axis = tensor.children
        if isinstance(shape, Expr) and shape.op == ' ':
            new_shape = deepcopy(shape)
            new_shape.children.append(idx_axis)
            return new_shape
        else:
            return Expr(' ', [shape, idx_axis])


def apply_shape_ops(node: Node, ops: Sequence[ShapeOp]) -> Node:
    for op in ops:
        node = op(node)
    return node


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


def normalize_recipe(node: Node) -> Sequence[ShapeOp]:
    msg = 'Not a tensor'
    if isinstance(node, (Constant, Symbol)):
        return []
    match node.op:
        case '->':
            raise ValueError(msg)
        case ',':
            raise ValueError(msg)
        case ' ':
            # tensor with axes
            new_shape = []
            for child in node.children:
                if isinstance(child, (Constant, Symbol)):
                    # nothing to unpack
                    new_shape.append(child)
                elif isinstance(child, Expr):
                    new_shape.extend(expanded_shape(child))
            if new_shape != node.children:
                return [ReshapeOp(new_shape), *normalize_recipe(ReshapeOp(new_shape)(node))]
            else:
                return []
        case '@':
            return [OneHotOp(), *normalize_recipe(OneHotOp()(node))]
        case _:
            msg = f'Invalid expression {node}: invalid op {node.op}'
            raise ValueError(msg)


class Program:
    def __init__(self, expr: Expr, constr: Constraints):
        assert expr.op == '->' and len(expr.children) == 2
        lhs, rhs = expr.children

        if lhs.op == ',':
            self.inputs = lhs.children
        else:
            self.inputs = [lhs]

        self.output = rhs

        self.axes = {}
        self.equations = constr.equations

    @classmethod
    def parse(cls, op: str):
        expr = make_expr(expr_parser.parse_string(unpack_shorthands(op)).as_list())
        constraints = postprocess_ast(expr)
        return cls(expr, constraints)

    def __repr__(self):
        strings = []
        for i, inp in enumerate(self.inputs):
            strings.append(f'Input {i+1}:\n{pprint.pformat(inp)}')
        strings.append('-' * 50)
        strings.append(f'Output:\n{pprint.pformat(self.output)}')
        strings.append('-' * 50)
        strings.append('Equations:')
        for eqn in self.equations:
            strings.append(f'{eqn[0]}\t=\t{eqn[1]}')
        return '\n'.join(strings)


env = Program.parse('b ( d=(n p ) d) c, b p*p*c h, h[g k], h[i k] -> b (n^2 g+i) k')
for child in [*env.inputs, env.output]:
    ops = normalize_recipe(child)
    print(ops)
    print(apply_shape_ops(child, ops))


print(env)
