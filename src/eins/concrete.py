"""Concrete implementations of the shape operations."""

from typing import Sequence, cast

from array_api_compat import array_namespace, is_jax_array

from eins.common_types import Array
from eins.constraint import Constraints
from eins.parsing import Expr
from eins.symbolic import (
    Combine,
    Concat,
    ExpandTo,
    Reduce,
    Reshape,
    ShapeOp,
    Split,
    Tensor,
    Tile,
    Transform,
    Transpose,
)


class ArrayBackend:
    def __init__(self, constr: Constraints):
        self.constr = constr

    def do(
        self, x: Sequence[Array], op: ShapeOp, ins: Sequence[Tensor], _outs: Sequence[Tensor]
    ) -> Sequence['Array']:
        "Apply the op, which goes from i to o, on x, an actual array."
        if len(x) == 0:
            msg = 'Cannot call operation on empty inputs'
            raise ValueError(msg)

        xp = array_namespace(*x)

        try:
            if op.is_identity_for(ins):
                # no-op
                return x
            elif isinstance(op, Reshape):
                new_shape = tuple(map(self.constr.value_of, op.new_shape))
                return [xp.reshape(x[0], shape=new_shape)]
            elif isinstance(op, Transpose):
                return [xp.permute_dims(x[0], axes=op.perm)]
            elif isinstance(op, Split):
                split_ax = ins[0].axes[op.axis_num]
                if not (isinstance(split_ax, Expr) and split_ax.op == '+'):
                    msg = f'Tried to split on {split_ax}, which is not a sum.'
                    raise ValueError

                sizes = []
                for child in split_ax.children:
                    val = self.constr.value_of(child)
                    if val is None:
                        msg = f'Could not compute value of {child} for split operation.'
                        raise ValueError(msg)
                    sizes.append(val)

                out = []
                curr = 0
                for size in sizes:
                    slc = [slice(None) for _ in range(x[0].ndim)]
                    slc[op.axis_num] = slice(curr, curr + size)
                    out.append(x[0][tuple(slc)])
                    curr += size
                return out
            elif isinstance(op, Concat):
                return [xp.concat(x, axis=op.axis_num)]
            elif isinstance(op, ExpandTo):
                slc = []
                i = 0
                for out_ax in op.new_shape:
                    if i < len(ins[0].axes) and out_ax == ins[0].axes[i]:
                        slc.append(slice(None))
                        i += 1
                    else:
                        slc.append(None)

                return [x[0][tuple(slc)]]
            elif isinstance(op, Tile):
                # Tile isn't in the version of the Array API that the Array API supports,
                # annoyingly. We try to wrap it as best we can.
                tiles = []
                if len(op.new_shape) != len(x[0].shape):
                    msg = f'Shape {x[0].shape} cannot be tiled to{op.new_shape}: shapes must match.'
                    raise ValueError(msg)

                for out_ax, in_len in zip(op.new_shape, x[0].shape):
                    out_len = self.constr.value_of(out_ax)
                    if out_len % in_len != 0:
                        msg = f'Cannot tile: {in_len} does not divide {out_len}'
                        msg += f': {x[0]}, {op.new_shape}'
                        raise ValueError(msg)
                    tiles.append(out_len // in_len)

                if hasattr(xp, 'tile'):
                    return [xp.tile(x[0], tuple(tiles))]
                elif is_jax_array(x[0]):
                    # workaround: use jnp.tile
                    from jax.numpy import tile
                    from jax.typing import ArrayLike

                    return [tile(cast(ArrayLike, x[0]), tuple(tiles))]
                else:
                    msg = (
                        f'Array of type {type(x[0])!s} does not support tiling. '
                        'Repeat functionality will not be supported.'
                    )
                    raise ValueError(msg)
            elif isinstance(op, Combine):
                # return [op.method(*xp.broadcast_arrays(*x))]
                return [op.method(*x)]
            elif isinstance(op, Reduce):
                # print(ins, _outs, op, id(_outs[0]))
                return [op.method(x[0], axis=ins[0].axes.index(op.axis))]
            elif isinstance(op, Transform):
                return [op.method(x[0], axis=ins[0].axes.index(op.axis))]
            else:
                msg = 'Op not supported: ' + str(op)
                raise TypeError(msg)
        except TypeError as err:
            msg = (
                'An error occurred when applying the operation.\n'
                f'Operation: {op}\n'
                f'Concrete inputs: {type(x[0])}, {[x_a.shape for x_a in x]}\n'
                f'Inputs: {ins}\n'
                f'Outputs: {_outs}\n'
                f'Variable values: {self.constr.known_vars}\n'
            )

            raise ValueError(msg) from err
