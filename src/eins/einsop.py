"""User-facing API."""

from collections import defaultdict
from typing import Callable, Mapping, Sequence, Union

from eins.combination import (
    ARRAY_COMBINE_OPS,
    Combination,
    CombineLiteral,
    CompositeCombination,
    UserCombination,
    parse_combination,
)
from eins.common_types import Array
from eins.concrete import ArrayBackend
from eins.elementwise import ARRAY_ELEMWISE_OPS, ElementwiseLiteral, ElementwiseOp, parse_elementwise
from eins.parsing import Constant
from eins.reduction import (
    ARRAY_REDUCE_OPS,
    CompositeReduction,
    Reduction,
    ReductionLiteral,
    UserReduction,
    parse_reduction,
)
from eins.symbolic import Program, Tensor

ElementwiseKind = Union[ElementwiseLiteral, Callable, ElementwiseOp]
ReductionKind = Union[ReductionLiteral, str, Callable, Reduction]
CombinationKind = Union[CombineLiteral, Callable, Combination]

GeneralReductionKind = Union[ReductionKind, Sequence[Union[ElementwiseKind, ReductionKind]]]
ReduceArg = Union[GeneralReductionKind, Mapping[str, GeneralReductionKind]]

CombineArg = Union[CombinationKind, Sequence[Union[ElementwiseKind, CombinationKind]]]


def parse_reduce_arg(reduce: GeneralReductionKind) -> Reduction:
    if isinstance(reduce, Reduction):
        return reduce
    elif isinstance(reduce, Callable):
        return UserReduction(reduce)
    elif isinstance(reduce, str):
        reduce_parse = parse_reduction(reduce)
        if reduce_parse is not None:
            return reduce_parse

        msg = f'Cannot parse reduction {reduce}. Valid literals are: {", ".join(ARRAY_REDUCE_OPS + ARRAY_COMBINE_OPS)}'
        raise ValueError(msg)
    else:
        ops = []
        for op in reduce:
            if isinstance(op, (ElementwiseOp, Reduction)):
                ops.append(op)
                continue

            op_parse = parse_reduction(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            op_parse = parse_elementwise(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            if isinstance(op, Callable):
                # callables are ambiguous here
                msg = (
                    f'User-supplied function in reduce={reduce} is ambiguous: either write a custom lambda'
                    ' combining these operations or explicitly pass in UserElementwiseOp or UserReduction.'
                )
                raise ValueError(msg)

            msg = f'Cannot parse operation {op} in {ops}. Valid literals are: ' + ', '.join(
                ARRAY_REDUCE_OPS + ARRAY_ELEMWISE_OPS + ARRAY_COMBINE_OPS
            )
            raise ValueError(msg)

        return CompositeReduction(ops)


def parse_combine_arg(combine: CombineArg) -> Combination:
    if isinstance(combine, Reduction):
        return combine
    elif isinstance(combine, Callable):
        return UserCombination(combine)
    elif isinstance(combine, str):
        combo_parse = parse_combination(combine)
        if combo_parse is not None:
            return combo_parse

        msg = f'Cannot parse reduction {combine}. Valid literals are: {", ".join(ARRAY_COMBINE_OPS)}'
        raise ValueError(msg)
    else:
        ops = []
        for op in combine:
            if isinstance(op, (ElementwiseOp, Combination)):
                ops.append(op)
                continue

            op_parse = parse_combination(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            op_parse = parse_elementwise(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            if isinstance(op, Callable):
                # callables are ambiguous here
                msg = (
                    f'User-supplied function in reduce={combine} is ambiguous: either write a custom lambda'
                    ' combining these operations or explicitly pass in UserElementwiseOp or UserReduction.'
                )
                raise ValueError(msg)

            msg = f'Cannot parse operation {op} in {ops}. Valid literals are: ' + ', '.join(
                ARRAY_REDUCE_OPS + ARRAY_ELEMWISE_OPS + ARRAY_COMBINE_OPS
            )
            raise ValueError(msg)

        return CompositeCombination(ops)


class EinsOp:
    """A computation on tensors, defined abstractly without specific inputs."""

    def __init__(self, op: str, /, *, reduce: ReduceArg = 'sum', combine: CombineArg = 'multiply'):
        """
        A tensor operation that takes in tensors of the given shapes and outputs a tensor of the given shape. Has a
        generic, powerful shape description language that supports the majority of tensor operations.

        Parameters
        ----------
        op:
            The description of the operation. For example, `'a b, b c -> a c'` performs matrix multiplication, and
            `'batch (size size) channels -> batch size size channels'` unpacks a batch of square images.

        reduce: str, Reduction, function with signature func(Array, axis: int) -> Array
            Describes how axes that appear in the input but not the output are eliminated. The default is `'sum'`, like
            in `einsum`. Common alternatives are `'mean'`, `'std'`, `'max'`, and `'min'`. This can also be a combine
            operation, which is reduced in the functional programming sense. For example, `'add'` would perform the same
            reduction as `'sum'` but in a less efficient loop. You can also pass in a function: it should be callable as
            `func(arr, axis=0)` and return an array with that axis eliminated. `eins` makes no guarantees about the
            order reductions are performed.

            For even more flexibility, this can be any of the previous inputs "sandwiched" by elementwise operations.
            For example, `('log', 'sum', 'exp')` would be equivalent to `logsumexp`. Note that passing in custom
            callables here is not allowed, because `eins` doesn't know whether a user-supplied function is an
            elementwise operation or not. Use a lambda instead.

            The final option is to pass in a mapping from axes to any of the previous reduction operation
            specifications. `eins` makes no guarantees about the order reductions are performed unless explicitly
            indicated, so be careful. `'a b c -> a', reduce={'b': 'max', 'c': 'min'}` has two meanings, depending on
            which happens first. Instead, you can pass `'a b c -> a b -> a'`, which forces a specific order.

        combine:
            Describes how the elements of different input tensors are combined. The default is `'multiply'`, which is
            what `einsum` does. This can be a list of elementwise operations and a single combination operation, like
            reduce: `('log', 'add', 'exp')` would be a less efficient equivalent operation to `'logaddexp'`.

            A custom callable should be callable as `func(arr1, arr2)` and return an array of the same shape as the two
            inputs. `eins` makes no guarantees about the order combinations are performed, so this function should be
            commutative and associative.
        """
        if '->' not in op:
            msg = f'Einsop "{op}" has no "->", which is required'
            raise ValueError(msg)

        self.op_str = op

        if isinstance(reduce, Mapping):
            self.reduce = {k: parse_reduce_arg(v) for k, v in reduce.items()}
        else:
            default_reduce = parse_reduce_arg(reduce)
            self.reduce = defaultdict(lambda: default_reduce)

        self.combine = parse_combine_arg(combine)

        self.program = Program.parse(self.op_str)
        self.program.reduce = self.reduce
        self.program.combine = self.combine

    def __repr__(self) -> str:
        return f'EinsOp({self.op_str}, reduce={self.reduce}, combine={self.combine})'

    def __call__(self, *tensors: Array) -> Array:
        """
        Apply the operation to the given tensors, returning the output tensor.

        Parameters
        ----------
        tensors: any number of Array objects
            The tensors to apply the operation to. Should be all the same type: numpy, torch, jax, cupy, and dask are
            all supported. The order matches the order of the input arguments.

        Returns
        -------
        The result of the operation.

        Raises
        ------
        ValueError:
            if the wrong number of tensors is passed in.
        """
        if len(tensors) != len(self.program.sources):
            msg = f'Expected {len(self.program.sources)} tensors, got {len(tensors)}'
            raise ValueError(msg)

        for concrete, shape in zip(tensors, self.program.sources):
            for lhs, rhs in zip(concrete.shape, shape.axes):
                self.program.constr.add_constraint(Constant(lhs), rhs)

        self.program.constr.solve()
        if self.program.constr.free_vars:
            # print(prog.constr)
            msg = f'Could not solve: free variables {self.program.constr.free_vars} remain'
            raise ValueError(msg)

        backend = ArrayBackend(self.program.constr)

        abstract = {}
        concrete = {}
        frontier = defaultdict(list)

        def fill(src: Tensor, arr, frontier=frontier):
            concrete[id(src)] = arr
            abstract[id(src)] = src

            # print(src, src.children)
            if len(src.children) == 0:
                return True

            for op, children in src.children:
                # either one-to-many or many-to-one or one-to-one
                if len(children) == 1:
                    child = children[0]
                    key = tuple(map(id, child.parents))
                    if key not in frontier:
                        frontier[key].append((op, (child,)))
                else:
                    frontier[(id(src),)].append((op, tuple(children)))
            return False

        for src, arr in zip(self.program.sources, tensors):
            fill(src, arr, frontier)

        # alg:
        # fill in value
        # if any children are now ready, add them

        changed = True
        while frontier and changed:
            # print('\n'.join([', '.join(str(abstract.get(k, '_unknown')) for k in ids) for ids in frontier.keys()]))
            # print('-' * 20)
            changed = False
            for input_ids in list(frontier.keys()):
                if all(input_id in concrete for input_id in input_ids):
                    for op, outputs in frontier[input_ids]:
                        x = [concrete[input_id] for input_id in input_ids]
                        i = [abstract[input_id] for input_id in input_ids]
                        o = list(outputs)
                        concrete_outputs = backend.do(x, op, i, o)
                        # print('co', concrete_outputs)
                        for concrete_out, out in zip(concrete_outputs, outputs):
                            if fill(out, concrete_out, frontier):
                                return concrete_out

                        changed = True
                    del frontier[input_ids]
                else:
                    pass
                    # print(input_ids, concrete.keys())

        if frontier:
            msg = 'Could not finish computation'
            print(frontier)
            print({k: v.shape for k, v in concrete.items()})
            print(abstract)
            print(list(map(id, self.program.sinks)))
            for k, outs in frontier.items():
                for op, outputs in outs:
                    print('---')
                    missing = list(outputs)
                    printed = set()
                    while missing:
                        m = missing.pop()
                        if id(m) in printed:
                            continue
                        print(f'Missing {m} ({id(m)})')
                        printed.add(id(m))
                        missing.extend([p for p in m.parents if id(p) not in concrete])

            raise ValueError(msg)
        return None


def einsop(op_str: str, *tensors: Array, reduce: ReduceArg = 'sum', combine: CombineArg = 'multiply') -> Array:
    """
    A functional version of [EinsOp] that does not allow for inspection or caching.

    This exists mainly as a bridge between that interface and the familiar one used by einops. Use [EinsOp] instead for
    serious development. That is also where the arguments are documented in more detail.

    Parameters
    ----------
    op_str:
        The einops operation string.
    tensors:
        The tensors to apply the operation to. Should be all the same type, and support the Array API.
    reduce:
        The reduction operation to apply to the outputs of the operation. Defaults to `'sum'`.
    combine:
        The combination operation to apply to the outputs of the operation. Defaults to `'multiply'`.

    Returns
    -------
    The result of the EinsOp.
    """
    op = EinsOp(op_str, reduce=reduce, combine=combine)
    return op(*tensors)
