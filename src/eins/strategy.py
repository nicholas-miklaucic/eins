"""Abstraction of different algorithmic choices made during program execution."""

from collections import defaultdict
from typing import Sequence, Union

from eins.common_types import Transformation
from eins.parsing import Constant, Symbol
from eins.program import Program
from eins.symbolic import Combine, ExpandTo, Reduce, ShapeOp, Tensor, Tile, Transform, Transpose


class BaseStrategy:
    """Base strategy: implements simple defaults and implements higher-order logic that doesn't
    usually need to change."""

    def __init__(self, prog: Program, reduce_early: bool = True) -> None:  # noqa: FBT001, FBT002
        self.prog = prog
        self.constr = prog.constr
        self.reduce_early = reduce_early
        self.smart_combine = True

    def optimal_combined_shape(self, t1: Tensor, t2: Tensor) -> Sequence[str]:
        """Gets shape to combine input tensors into."""
        if self.smart_combine:
            # We want to find an order of axes that can avoid transpositions. For example, the best
            # way to broadcast a b c and a d c is a b d c, not a b c d or a d c b.
            ax1 = t1.axes_list()
            ax2 = t2.axes_list()
            t1_common = [ax for ax in ax1 if ax in ax2]
            common = set(t1_common)

            gaps = defaultdict(list)
            for axes in (ax1, ax2):
                curr_ax = None
                for ax in axes:
                    if ax in common:
                        curr_ax = ax
                        continue
                    else:
                        gaps[curr_ax].append(ax)
            out_shape = gaps[None]
            for ax in t1_common:
                out_shape.append(ax)
                out_shape.extend(gaps[ax])
            return out_shape

        else:
            out_shape = t1.axes_list()
            for ax in t2.axes_list():
                if ax not in out_shape:
                    out_shape.append(ax)
            return out_shape

    def combine_mismatched(self, combine: Combine, t1: Tensor, t2: Tensor) -> Tensor:
        """Combines tensors together, broadcasting and transposing as needed. Assumes tensors are in
        normal form."""
        reshaped = []
        new_shape = self.optimal_combined_shape(t1, t2)
        for t in (t1, t2):
            axs = t.axes_list()
            perm = tuple(sorted(range(len(axs)), key=lambda x: new_shape.index(axs[x])))
            transposed_t = Transpose(perm)(t)[0]
            new_ax_op = ExpandTo(tuple(map(Symbol, new_shape)))
            expanded_t = self.apply_op(new_ax_op, transposed_t)[0]
            reshaped.append(expanded_t)

        out = self.apply_op(combine, reshaped)[0]
        return out

    def apply_op(self, op: ShapeOp, tensors: Union[Tensor, Sequence[Tensor]]):
        return self.prog.apply_op(op, tensors)

    def connect(self, start: Sequence[Tensor], goal: Tensor) -> Tensor:
        """Connects the start tensors and the goal. Returns tensor of same shape as goal, connected
        in the graph to the start inputs."""
        goal_axes = goal.axes_list()
        start_axes = start[0].axes_list()
        for other in start[1:]:
            for ax in other.axes_list():
                if ax not in start_axes:
                    start_axes.append(ax)

        reduce_axes = set(start_axes) - set(goal_axes)

        if self.reduce_early:
            transposed = start
            exp_axes = [set(exp.axes_list()) for exp in transposed]
            combined = transposed[0]
            for i, exp in enumerate(transposed):
                if i == 0:
                    continue
                else:
                    combined = self.combine_mismatched(Combine(self.prog.combine), combined, exp)

                in_rest = set()
                for remaining_axes in exp_axes[i + 1 :]:
                    in_rest |= remaining_axes

                # axes in the input, to reduce, that aren't in future tensors
                to_reduce = (reduce_axes - in_rest) & set(combined.axes_list())
                for r_ax in to_reduce:
                    combined = self.apply_op(
                        Reduce(self.prog.reduce[r_ax], Symbol(r_ax)), combined
                    )[0]
            reduced = combined
        else:
            transposed = []
            for s in start:
                axs = s.axes_list()
                perm = tuple(sorted(range(len(axs)), key=lambda x: start_axes.index(axs[x])))
                transposed.extend(self.apply_op(Transpose(perm), s))
            expanded = []
            for t in transposed:
                expanded.extend(self.apply_op(ExpandTo(tuple(map(Symbol, start_axes))), t))

            if len(expanded) > 1:
                combined = self.apply_op(Combine(self.prog.combine), tuple(expanded))[0]
            else:
                combined = expanded[0]

            reduced = combined

        for ax in reduce_axes:
            if ax in reduced.axes_list():
                reduced = self.apply_op(Reduce(self.prog.reduce[ax], Symbol(ax)), reduced)[0]

        r_axs = reduced.axes_list()

        perm = tuple(sorted(range(len(r_axs)), key=lambda x: goal_axes.index(r_axs[x])))

        out = self.apply_op(Transpose(perm), reduced)[0]

        # expansion if necessary
        # TODO should there be some kind of warning here? this makes some syntax ambiguous
        if set(out.axes_list()) < set(goal_axes):
            out = self.apply_op(ExpandTo(tuple(goal.axes)), out)[0]
            out = self.apply_op(Tile(tuple(goal.axes)), out)[0]

        if out.axes_list() != goal_axes:
            msg = f'{out.axes_list()} != {goal_axes}'
            raise ValueError(msg)

        return out
