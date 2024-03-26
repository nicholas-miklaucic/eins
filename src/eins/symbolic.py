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


class Tensor:
    """Node in a tensor network."""
    def __init__(self, expr: Node):
        self.parents = []
        self.children = []
        self.child_ops = []
        self.idx_axis = None
        if expr.op == ' ':
            self.axes = expr.children
        elif expr.op == '@':
            self.axes = Tensor(expr.children[0]).axes
            self.idx_axis = expr.children[1]
        else:
            self.axes = [expr]

    def add_child(self, child: 'Tensor', op: 'ShapeOp'):
        child.parents.append(self)
        self.children.append(child)
        self.child_ops.append(op)

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
        children = self.apply(tensors)

        for child in children:
            for t in tensors:
                t.add_child(child, self)
        return children
    


@dataclass
class Reshape(ShapeOp):
    new_shape: tuple[Node]

    def apply(self, _tensor: Sequence[Tensor]) -> Sequence[Tensor]:
        return [Tensor(Expr(' ', list(self.new_shape)))]


@dataclass
class Transpose(ShapeOp):
    perm: tuple[int]

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [Tensor(Expr(' ', [tensors[0][i] for i in self.perm]))]


@dataclass
class Split(ShapeOp):
    axis_num: int

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        children = []
        for addend in tensors[0].axes[self.axis_num].children:
            child = deepcopy(tensors[0])
            child.axes[self.axis_num] = addend
            children.append(child)
        return children


@dataclass
class OneHot(ShapeOp):
    idx_axis: Node

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        child = deepcopy(tensors[0])
        child.axes.append(child.idx_axis)
        child.idx_axis = None
        return [child]
    
@dataclass
class ExpandTo(ShapeOp):
    """Adds new axes with 1 to broadcast with the current shape. Output should be 
    a supersequence of the current shape."""
    new_shape: tuple[Node]
    
    def apply(self, _tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [Tensor(Expr(' ', list(self.new_shape)))]
    
    
@dataclass
class Combine(ShapeOp):
    method: Reduction

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        return [deepcopy(tensors[0])]
    

@dataclass
class Reduce(ShapeOp):
    method: Reduction
    axis: Node

    def apply(self, tensors: Sequence[Tensor]) -> Sequence[Tensor]:
        out = deepcopy(tensors[0])
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
                new_shape = (deepcopy(tensor.axes[:i]) + expanded_shape(ax) + deepcopy(tensor.axes[i+1:]))
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


class Program:
    def __init__(self, expr: Expr, constr: Constraints):
        assert expr.op == '->' and len(expr.children) == 2
        lhs, rhs = expr.children

        if lhs.op == ',':
            self.sources = [Tensor(c) for c in lhs.children]
        else:
            self.sources = [Tensor(lhs)]

        self.current = []
        for source in self.sources:
            self.current.extend(normalize_until_done(source))

        self.sink = Tensor(rhs)
    
        self.equations = constr.equations

    @classmethod
    def parse(cls, op: str):
        expr = make_expr(expr_parser.parse_string(unpack_shorthands(op)).as_list())
        constraints = postprocess_ast(expr)
        return cls(expr, constraints)

    def __repr__(self):
        strings = []
        for i, inp in enumerate(self.sources):
            strings.append(f'Input {i+1}:\n{pprint.pformat(inp)}')
        strings.append('-' * 50)
        for i, inp in enumerate(self.current):
            strings.append(f'Current {i+1}:\n{pprint.pformat(inp)}')
        strings.append('-' * 50)
        strings.append(f'Output:\n{pprint.pformat(self.sink)}')
        strings.append('-' * 50)
        strings.append('Equations:')
        for eqn in self.equations:
            strings.append(f'{eqn[0]}\t=\t{eqn[1]}')
        return '\n'.join(strings)


env = Program.parse('b ((n p) (n p)) c d=c, b p*p*d*c h, h[g+i k] -> b (n^2 g+i) k')
print(env)


# sugar: doubled axes can be replaced with =, using the narrow "different axis + constraint"
# interpretation of =

# add cost analysis to exprs in future?