"""Symbolic representation of tensor operations and manipulations."""

from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from multiprocessing import Value
from re import I

from eins.parsing import Expr
from eins.reduction import Prod, Reduction, Sum


@dataclass(frozen=True, unsafe_hash=True)
class Axis:
    """A semantic representation of an axis."""

    name: str

    def __str__(self):
        return self.name


@dataclass
class Tensor:
    """Symbolic representation of a tensor, as a set of axes."""

    dims: set[Axis]

    def __str__(self):
        return ' '.join(map(str, self.dims))


class Op(metaclass=ABCMeta):
    @abstractmethod
    def name(self) -> str:
        """The name of the operation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def inputs(self) -> list[Tensor]:
        """The inputs to the operation."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def output(self) -> Tensor:
        """The output of the operation."""
        raise NotImplementedError()

    def __str__(self):
        lhs = ', '.join(map(str, self.inputs))
        rhs = str(self.output)
        return f'{lhs:>20} -> {rhs}'


class Combine(Op):
    """Combines two tensors along an axis that matches."""

    def __init__(self, left: Tensor, right: Tensor, axis: Axis, func: Reduction) -> None:
        super().__init__()

        if left.dims.intersection(right.dims) != {axis}:
            msg = f'Cannot combine along {axis}.\nLeft:{left}\nRight:{right}'
            raise ValueError(msg)

        self.left = left
        self.right = right
        self.axis = axis
        self.func = func

    def name(self):
        return f'Combine({self.left}, {self.right}, {self.axis}, {self.func})'

    @property
    def inputs(self):
        return [self.left, self.right]

    @property
    def output(self):
        return Tensor(self.left.dims.union(self.right.dims))


class Reduce(Op):
    """Reduction, eliminating an axis."""

    def __init__(self, tensor: Tensor, axis: Axis, func: Reduction) -> None:
        super().__init__()
        if axis not in tensor.dims:
            msg = f'Cannot reduce along {axis}, not in tensor.\nTensor:{tensor}'
            raise ValueError(msg)

        self.tensor = tensor
        self.axis = axis
        self.func = func

    def name(self):
        return f'Reduce({self.tensor}, {self.axis}, {self.func})'

    @property
    def inputs(self):
        return [self.tensor]

    @property
    def output(self):
        return Tensor(self.tensor.dims - {self.axis})


class Program:
    def __init__(self, expr: Expr, equations: list):
        assert expr.op == '->' and len(expr.children) == 2
        lhs, rhs = expr.children

        if lhs.op == ',':
            self.inputs = [lhs.children]
        else:
            self.inputs = [lhs]

        self.output = rhs

        self.axes = {}
        self.equations = equations

    def ax(self, name: str) -> Axis:
        if name in self.axes:
            return self.axes[name]
        else:
            axis = Axis(name)
            self.axes[name] = axis
            return axis

    def simple_tensor(self, name: str, shape: str) -> Tensor:
        tens = Tensor({self.ax(name) for name in shape.split(' ')})
        self.tensors[name] = tens
        return tens

    def apply_op(self, op: Op, out_name: str) -> Tensor:
        print(op)
        self.tensors[out_name] = op.output
        return self.tensors[out_name]


env = Program()
P = env.simple_tensor('P', 'a b')
Q = env.simple_tensor('P', 'c b')
R = env.simple_tensor('R', 'd c')
S = env.apply_op(Combine(P, Q, env.ax('b'), Prod), 'S')
Sr = env.apply_op(Reduce(S, env.ax('b'), Sum), 'Sr')
T = env.apply_op(Combine(Sr, R, env.ax('c'), Prod), 'T')
Tr = env.apply_op(Reduce(T, env.ax('c'), Sum), 'Tr')
