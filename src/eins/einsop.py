"""User-facing API."""

import typing
from itertools import chain
from typing import AnyStr, Callable, Mapping, MutableMapping, Sequence, Union

from eins.combination import (
    ARRAY_COMBINE_OPS,
    Combination,
    CombineLiteral,
    CompositeCombination,
    CustomCombination,
    parse_combination,
)
from eins.common_types import Array
from eins.concrete import ArrayBackend
from eins.elementwise import ElementwiseLiteral, ElementwiseOp, parse_elementwise
from eins.parsing import Constant
from eins.reduction import (
    ARRAY_REDUCE_OPS,
    CompositeReduction,
    CustomReduction,
    Reduction,
    ReductionLiteral,
    parse_reduction,
)
from eins.symbolic import Program, Tensor
from eins.transformation import Transformation, TransformationLiteral, parse_transformation

ElementwiseKind = Union[ElementwiseLiteral, Callable, ElementwiseOp]
# use AnyStr to ensure autocomplete works
# https://stackoverflow.com/questions/77012761/literal-string-union-autocomplete-for-python
ReductionKind = Union[ReductionLiteral, AnyStr, Callable, Reduction]
CombinationKind = Union[CombineLiteral, Callable, Combination]
TransformationKind = Union[TransformationLiteral, Transformation]

GeneralReductionKind = Union[
    ReductionKind, Sequence[Union[ElementwiseKind, TransformationKind, ReductionKind]]
]
ReduceArg = Union[GeneralReductionKind, Mapping[str, GeneralReductionKind]]

CombineArg = Union[CombinationKind, Sequence[Union[ElementwiseKind, CombinationKind]]]


def _parse_reduce_arg(reduce: GeneralReductionKind) -> Reduction:
    if isinstance(reduce, Reduction):
        return reduce
    elif isinstance(reduce, Callable):
        return CustomReduction(reduce)
    elif isinstance(reduce, str):
        reduce_parse = parse_reduction(reduce)
        if reduce_parse is not None:
            return reduce_parse

        msg = f'Cannot parse reduction {reduce}. Valid literals are: {", ".join(
            ARRAY_REDUCE_OPS + ARRAY_COMBINE_OPS)}'
        raise ValueError(msg)
    else:
        ops = []
        for op in reduce:
            if isinstance(op, (ElementwiseOp, Reduction, Transformation)):
                ops.append(op)
                continue

            if isinstance(op, Callable):
                # callables are ambiguous here
                msg = f"""
User-supplied function in reduce={reduce} is ambiguous: either write a custom lambda combining these
operations or explicitly create objects using e.g., eins.ElementwiseOps.from_func().
                      """
                raise ValueError(msg)

            did_parse = False
            for parser in (parse_reduction, parse_elementwise, parse_transformation):
                op_parse = parser(op)
                if op_parse is not None:
                    ops.append(op_parse)
                    did_parse = True
                    break

            if did_parse:
                continue

            msg = f'Cannot parse operation {op} in {ops}. Valid literals are: ' + (
                '\n'.join(
                    [
                        ', '.join(
                            map(
                                str, chain.from_iterable(map(typing.get_args, typing.get_args(ops)))
                            )
                        )
                        for ops in (
                            ReductionLiteral,
                            ElementwiseLiteral,
                            TransformationLiteral,
                            CombineLiteral,
                        )
                    ]
                )
            )
            raise ValueError(msg)

        return CompositeReduction(tuple(ops))


def _parse_combine_arg(combine: CombineArg) -> Combination:
    if isinstance(combine, Combination):
        return combine
    elif isinstance(combine, Callable):
        return CustomCombination(combine)
    elif isinstance(combine, str):
        combo_parse = parse_combination(combine)
        if combo_parse is not None:
            return combo_parse

        msg = (
            f'Cannot parse reduction {combine}. Valid literals are: {", ".join(ARRAY_COMBINE_OPS)}'
        )
        raise ValueError(msg)
    else:
        ops = []
        for op in combine:
            if isinstance(op, (ElementwiseOp, Combination)):
                ops.append(op)
                continue

            if isinstance(op, Callable):
                # callables are ambiguous here
                msg = f"""
User-supplied function in combine={combine} is ambiguous: either write a custom lambda combining
these operations or explicitly create objects using e.g., eins.ElementwiseOps.from_func().
                      """
                raise ValueError(msg)

            op_parse = parse_combination(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            op_parse = parse_elementwise(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            msg = f'Cannot parse operation {op} in {ops}. Valid literals are: ' + (
                '\n'.join(
                    [
                        ', '.join(typing.get_args(ops))
                        for ops in (CombineLiteral, ElementwiseLiteral)
                    ]
                )
            )
            raise ValueError(msg)

        return CompositeCombination(tuple(ops))


class EinsOp:
    """A computation on tensors, defined abstractly without specific inputs."""

    def __init__(self, op: str, /, *, reduce: ReduceArg = 'sum', combine: CombineArg = 'multiply'):
        """
        A tensor operation that takes in tensors of the given shapes and outputs a tensor of the
        given shape. Has a generic, powerful shape description language that supports the majority
        of tensor operations.

        Parameters
        ----------
        op:
            The description of the operation. For example, `'a b, b c -> a c'` performs matrix
            multiplication, and `'batch (size size) channels -> batch size size channels'` unpacks a
            batch of square images.

        reduce: function f(Array, axis: int) → Array, Reduction, str, or mapping from axes to
        previous
            Describes how axes that appear in the input but not the output are eliminated: use
            [eins.Reductions] to get an autocomplete-friendly list of options. The default is
            `'sum'`, like in `einsum`. Common alternatives are `'mean'`, `'std'`, `'max'`, and
            `'min'`. This can also be a combine operation, which is reduced in the functional
            programming sense. For example, `'add'` would perform the same reduction as `'sum'` but
            in a less efficient loop. You can also pass in a function: it should be callable as
            `func(arr, axis=0)` and return an array with that axis eliminated. `eins` makes no
            guarantees about the order reductions are performed.

            For even more flexibility, this can be any of the previous inputs "sandwiched" by
            elementwise operations. For example, `('log', 'sum', 'exp')` would be equivalent to
            `logsumexp`. Note that passing in custom callables here is not allowed, because `eins`
            doesn't know whether a user-supplied function is an elementwise operation or not. Use a
            lambda instead.

            The final option is to pass in a mapping from axes to any of the previous reduction
            operation specifications. `eins` makes no guarantees about the order reductions are
            performed unless explicitly indicated, so be careful. `'a b c -> a', reduce={'b': 'max',
            'c': 'min'}` has two meanings, depending on which happens first. Instead, you can pass
            `'a b c -> a b -> a'`, which forces a specific order.

        combine:
            Describes how the elements of different input tensors are combined: use
            [eins.Combinations] to get an autocomplete-friendly list of options. The default is
            `'multiply'`, which is what `einsum` does. This can be a list of elementwise operations
            and a single combination operation, like reduce: `('log', 'add', 'exp')` would be a less
            efficient equivalent operation to `'logaddexp'`.

            A custom callable should be callable as `func(arr1, arr2)` and return an array of the
            same shape as the two inputs. `eins` makes no guarantees about the order combinations
            are performed, so this function should be commutative and associative.
        """
        if '->' not in op:
            msg = f'Einsop "{op}" has no "->", which is required'
            raise ValueError(msg)

        self.op_str = op

        if isinstance(reduce, Mapping):
            self.reduce = {k: _parse_reduce_arg(v) for k, v in reduce.items()}
        else:
            self.reduce = _parse_reduce_arg(reduce)

        self.combine = _parse_combine_arg(combine)

        self.program = Program.parse(self.op_str, combine=self.combine, reduce=self.reduce)

    def __repr__(self) -> str:
        return f'EinsOp({self.op_str}, reduce={self.reduce}, combine={self.combine})'

    def __call__(self, *tensors: Array) -> Array:
        """
        Apply the operation to the given tensors, returning the output tensor.

        Parameters
        ----------
        tensors: any number of Array objects
            The tensors to apply the operation to. Should be all the same type: numpy, torch, jax,
            cupy, and dask are all supported. The order matches the order of the input arguments.

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

        for concrete_arr, shape in zip(tensors, self.program.sources):
            for lhs, rhs in zip(concrete_arr.shape, shape.axes):
                self.program.constr.add_constraint(Constant(lhs), rhs)

        self.program.constr.solve()
        if self.program.constr.free_vars:
            # print(prog.constr)
            msg = f'Could not solve: free variables {self.program.constr.free_vars} remain'
            raise ValueError(msg)

        backend = ArrayBackend(self.program.constr)

        abstract: MutableMapping[int, Tensor] = {}
        concrete: MutableMapping[int, Array] = {}

        def fill_from(src: Tensor, arr: Array):
            """Fills forward from the known tensor. Returns the answer, if found."""
            concrete[id(src)] = arr
            abstract[id(src)] = src

            if len(src.children) == 0:
                # we're done
                return arr

            for op, children in src.children:
                if len(children) == 1:
                    # could be a many-to-one op
                    child = children[0]
                    concrete_parents = [concrete[id(parent)] for parent in child.parents]
                    if all(p is not None for p in concrete_parents):
                        # we can fill in this operation
                        result = backend.do(concrete_parents, op, child.parents, [child])[0]
                        res = fill_from(child, result)
                        if res is not None:
                            return res
                else:
                    # no many-to-many ops: we can fill in all of these
                    child_results = backend.do([arr], op, [src], children)
                    for child, result in zip(children, child_results):
                        res = fill_from(child, result)
                        if res is not None:
                            return res

            return None

        for src, arr in zip(self.program.sources, tensors):
            ans = fill_from(src, arr)
            if ans is not None:
                return ans

        msg = f'Could not solve tensor graph: {self.program}'
        raise ValueError(msg)


def einsop(
    *tensors_and_pattern: Union[Array, str],
    reduce: ReduceArg = 'sum',
    combine: CombineArg = 'multiply',
) -> Array:
    """
    A functional version of [EinsOp] that does not allow for inspection or caching.

    This exists mainly as a bridge between that interface and the familiar one used by einops. Use
    [EinsOp] instead, unless you really want a roughly compatible einops-like function.

    Parameters
    ----------
    tensors_and_pattern:
        The tensors to apply the operation to, followed by the op string. Should be all the same
        type, and support the Array API.
    reduce:
        The reduction operation to apply to the outputs of the operation. Defaults to `'sum'`.
    combine:
        The combination operation to apply to the outputs of the operation. Defaults to
        `'multiply'`.

    Returns
    -------
    The result of the EinsOp.
    """
    tensors = []
    pattern = ''
    for t in tensors_and_pattern:
        if isinstance(t, str):
            if pattern:
                msg = f"""
Two strings passed in: {pattern} and {t}. Perhaps you mean reduce= or combine=?"""
                raise ValueError(msg)
            else:
                pattern = t
        else:
            tensors.append(t)

    op = EinsOp(pattern, reduce=reduce, combine=combine)
    return op(*tensors)
