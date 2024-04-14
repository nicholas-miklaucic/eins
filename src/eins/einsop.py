"""User-facing API."""

import typing
from itertools import chain
from string import ascii_uppercase
from typing import AnyStr, Callable, Mapping, MutableMapping, Optional, Sequence, Union

from eins.combination import (
    Combination,
    CombineLiteral,
    CompositeCombination,
    CustomCombination,
    parse_combination,
)
from eins.combination import (
    ops as _combination_ops,
)
from eins.common_types import Array
from eins.concrete import ArrayBackend
from eins.elementwise import ElementwiseLiteral, ElementwiseOp, parse_elementwise
from eins.elementwise import ops as _elementwise_ops
from eins.parsing import Constant, Symbol
from eins.program import Program, TransformProgram
from eins.reduction import (
    CompositeReduction,
    CustomReduction,
    Reduction,
    ReductionLiteral,
    parse_reduction,
)
from eins.reduction import (
    ops as _reduction_ops,
)
from eins.strategy import BaseStrategy
from eins.symbolic import Tensor
from eins.transformation import (
    CompositeTransformation,
    CustomTransformation,
    Transformation,
    TransformationLiteral,
    parse_transformation,
)
from eins.transformation import ops as _transformation_ops

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
TransformArg = Union[TransformationKind, Sequence[Union[ElementwiseKind, TransformationKind]]]


def _parse_reduce_arg(reduce: GeneralReductionKind) -> Reduction:
    if isinstance(reduce, Reduction):
        return reduce
    elif isinstance(reduce, Callable):
        return CustomReduction(reduce)
    elif isinstance(reduce, str):
        reduce_parse = parse_reduction(reduce)
        if reduce_parse is not None:
            return reduce_parse

        msg = f'Cannot parse reduction {reduce}. Valid literals are: ' + ', '.join(
            list(_reduction_ops) + list(_combination_ops)
        )
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


def _parse_transform_arg(transform: TransformArg) -> Transformation:
    if isinstance(transform, Transformation):
        return transform
    elif isinstance(transform, Callable):
        return CustomTransformation(transform)
    elif isinstance(transform, str):
        combo_parse = parse_transformation(transform)
        if combo_parse is not None:
            return combo_parse

        msg = f'Cannot parse transformation {transform}. Valid literals: {", ".join(_transformation_ops)}'
        raise ValueError(msg)
    else:
        ops = []
        for op in transform:
            if isinstance(op, (ElementwiseOp, Transformation)):
                ops.append(op)
                continue

            if isinstance(op, Callable):
                # callables are ambiguous here
                msg = f"""
User-supplied function in transform={transform} is ambiguous: either write a custom lambda combining
these operations or explicitly create objects using e.g., eins.ElementwiseOps.from_func().
                      """
                raise ValueError(msg)

            op_parse = parse_transformation(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            op_parse = parse_elementwise(op)
            if op_parse is not None:
                ops.append(op_parse)
                continue

            msg = f'Cannot parse operation {op} in {ops}. Valid literals: ' + (
                '\n'.join([', '.join(list(_transformation_ops) + list(_elementwise_ops))])
            )
            raise ValueError(msg)

        return CompositeTransformation(tuple(ops))


def _parse_combine_arg(combine: CombineArg) -> Combination:
    if isinstance(combine, Combination):
        return combine
    elif isinstance(combine, Callable):
        return CustomCombination(combine)
    elif isinstance(combine, str):
        combo_parse = parse_combination(combine)
        if combo_parse is not None:
            return combo_parse

        msg = f'Cannot parse combination {combine}. Valid literals: {", ".join(_combination_ops)}'
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

            msg = f'Cannot parse operation {op} in {ops}. Valid literals: ' + (
                '\n'.join([', '.join(list(_combination_ops) + list(_elementwise_ops))])
            )
            raise ValueError(msg)

        return CompositeCombination(tuple(ops))


class EinsOp:
    """A computation on tensors, defined abstractly without specific inputs."""

    def __init__(
        self,
        op: str,
        /,
        *,
        reduce: ReduceArg = 'sum',
        combine: CombineArg = 'multiply',
        transform: Optional[Mapping[str, TransformArg]] = None,
        symbol_values: Optional[Mapping[str, int]] = None,
    ):
        """
        A tensor operation that takes in tensors of the given shapes and outputs a tensor of the
        given shape. Has a generic, powerful shape description language that supports the majority
        of tensor operations.

        Parameters
        ----------
        op: str or sequence of strs
            The description of the operation.

            If a list of strings is passed in, the last string is assumed to be the output, and the
            others are assumed to be inputs.

            For example, `'a b, b c -> a c'` performs matrix multiplication. There is a lot of
            supported syntax: consult the tutorial for more information.

        reduce: function, Reduction, str, sequence of ops, or mapping from axes to previous
            Describes how axes that appear in the input but not the output are eliminated: use
            [eins.Reductions] to get an autocomplete-friendly list of options.

            The default is `'sum'`, like in `einsum`. Common alternatives are `'mean'`, `'std'`,
            `'max'`, and `'min'`. This can also be a combine operation, which is reduced in the
            functional programming sense. For example, `'add'` would perform the same reduction as
            `'sum'` but in a less efficient loop. You can also pass in a function: it should be
            callable as `func(arr, axis=0)` and return an array with that axis eliminated. `eins`
            makes no guarantees about the order reductions are performed.

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

        combine: function, Combination, str, sequence of ops, or mapping from axes to previous
            Describes how the elements of different input tensors are combined: use
            [eins.Combinations] to get an autocomplete-friendly list of options.

            The default is `'multiply'`, which is what `einsum` does. This can be a list of
            elementwise operations and a single combination operation, like reduce: `('log', 'add',
            'exp')` would be a less efficient equivalent operation to `'logaddexp'`.

            A custom callable should be callable as `func(arr1, arr2)` and return an array of the
            same shape as the two inputs. `eins` makes no guarantees about the order combinations
            are performed, so this function should be commutative and associative.

        transform: mapping from axes to: function, Transformation, str, or sequence of previous

        symbol_values: mapping from symbols to integers or None
            An alternative to using = to specify axis values.
        """
        if transform is None:
            transform = {}

        if '->' not in op and len(transform) == 0:
            msg = f'Einsop "{op}" has no "->", which is required unless transform is given.'
            raise ValueError(msg)
        elif '->' in op and len(transform) > 0:
            msg = f'Einsop "{op}" has "->", which is not allowed with transform.'
            raise ValueError(msg)

        self.is_transform = len(transform) > 0

        if isinstance(op, str):
            self.op_str = op
        elif len(op) == 0:
            msg = 'EinsOp must have at least one operation.'
            raise ValueError(msg)
        elif len(op) == 1:
            self.op_str = op[0]
        else:
            *inputs, output = op[-1]
            self.op_str = ', '.join(inputs) + ' -> ' + output

        if self.is_transform:
            self.transform = {k: _parse_transform_arg(v) for k, v in transform.items()}
            self.program = TransformProgram.parse(self.op_str, transform=self.transform)
        else:
            if isinstance(reduce, Mapping):
                self.reduce = {k: _parse_reduce_arg(v) for k, v in reduce.items()}
            else:
                self.reduce = _parse_reduce_arg(reduce)

            self.combine = _parse_combine_arg(combine)

            self.program = Program.parse(self.op_str, combine=self.combine, reduce=self.reduce)

        self.symbol_values = symbol_values or {}
        for k, v in self.symbol_values.items():
            self.program.constr.add_constraint(Symbol(k), Constant(v))

    def __repr__(self) -> str:
        if self.is_transform:
            return f'EinsOp({self.op_str}, transform={self.transform})'
        else:
            return f'EinsOp({self.op_str}, reduce={self.reduce}, combine={self.combine})'

    def __str__(self) -> str:
        names = {}

        def get_name(tens_id, names=names):
            if tens_id in names:
                return names[tens_id]
            else:
                name_symbols = ascii_uppercase
                if len(names) < len(name_symbols):
                    suffix = ''
                else:
                    suffix = str(len(names) // len(name_symbols) - 1)

                name_sym_base = name_symbols[len(names) % len(name_symbols)]
                name_sym = f'{name_sym_base}{suffix}'
                names[tens_id] = name_sym
                return name_sym

        out = [repr(self)]
        if hasattr(self, 'instructions') and self.instructions:
            # has graph
            out.append('Execution Graph:\n')
            for op, sources, sinks in self.instructions:
                source_str = ', '.join(
                    [get_name(i) + '{' + str(self.abstract.get(i, 'NA')) + '}' for i in sources]
                )

                sink_str = ', '.join(
                    [get_name(i) + '{' + str(self.abstract.get(i, 'NA')) + '}' for i in sinks]
                )
                out.append(f'{op!s:>60}\t{source_str:>40} → {sink_str:<25}')

        out.append('Constraints:')
        out.append(str(self.program.constr))
        return '\n'.join(out)

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

        self.strat = BaseStrategy(self.program)

        self.program.make_path(self.strat)

        self.backend = ArrayBackend(self.program.constr)
        self.instructions = []

        self.abstract: MutableMapping[int, Tensor] = {}
        self.concrete: MutableMapping[int, Array] = {}

        def fill_from(src: Tensor, arr: Array, instructions=self.instructions):
            """Fills forward from the known tensor. Returns the answer, if found."""
            self.concrete[id(src)] = arr
            self.abstract[id(src)] = src

            # print(src, src.children)

            if len(src.children) == 0:
                # we're done
                return [arr]

            leaves = []

            for op, children in src.children:
                if len(children) == 1:
                    # could be a many-to-one op
                    child = children[0]
                    concrete_parents = [self.concrete.get(id(parent)) for parent in child.parents]
                    if all(p is not None for p in concrete_parents):
                        # we can fill in this operation
                        result = self.backend.do(concrete_parents, op, child.parents, [child])[0]
                        instructions.append((op, list(map(id, child.parents)), [id(child)]))
                        # type: ignore
                        res = fill_from(child, result)
                        leaves.extend(res)

                else:
                    # no many-to-many ops: we can fill in all of these
                    child_results = self.backend.do([arr], op, [src], children)
                    instructions.append((op, [id(src)], list(map(id, children))))
                    for child, result in zip(children, child_results):
                        res = fill_from(child, result)
                        leaves.extend(res)

            return leaves

        try:
            leaves = []
            for src, arr in zip(self.program.sources, tensors):
                ans = fill_from(src, arr)
                leaves.extend(ans)

            self.out_shape = [
                self.program.constr.value_of(ax) for ax in self.program.orig_sink.axes
            ]
            potential_returns = [l for l in leaves if list(l.shape) == self.out_shape]
            if potential_returns:
                ans = potential_returns[-1]
                self.abstract[id(ans)] = self.program.orig_sink
                # the concrete arrays are large: we don't want to store them indefinitely
                self.concrete.clear()
                return ans
        except ValueError as err:
            msg = f"""
Error occurred during computation.
{self}
"""
            raise ValueError(msg) from err

        msg = f'Could not solve tensor graph: \n{self}'
        raise ValueError(msg)


def einsop(
    *tensors_and_pattern: Union[Array, str],
    reduce: ReduceArg = 'sum',
    combine: CombineArg = 'multiply',
    transform: Optional[Mapping[str, TransformArg]] = None,
    **kwargs,
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

    op = EinsOp(pattern, reduce=reduce, combine=combine, transform=transform, symbol_values=kwargs)
    return op(*tensors)
