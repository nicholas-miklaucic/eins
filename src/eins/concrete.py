"""Concrete implementations of the shape operations."""

from abc import ABCMeta
from typing import Sequence

from eins.constraint import Constraints
from eins.symbolic import Combine, Concat, ExpandTo, Reduce, Reshape, ShapeOp, Transpose, Split, Tensor

from numpy import array_api as aa
from numpy.array_api import _array_object as ao

Array = ao.Array

# transpose, split, concat, onehot, expandto, combine, reduce

class ArrayBackend:
    def __init__(self, constr: Constraints):
        self.constr = constr

    def do(self, x: Sequence[Array], op: ShapeOp, ins: Sequence[Tensor], outs: Sequence[Tensor]) -> Sequence[Array]:
        "Apply the op, which goes from i to o, on x, an actual array."
        # print(x, op, ins, outs)
        if isinstance(op, Reshape):
            new_shape = tuple(map(self.constr.value_of, op.new_shape))
            return [aa.reshape(x[0], shape=new_shape)]
        elif isinstance(op, Transpose):
            return [aa.permute_dims(x[0], op.perm)]
        elif isinstance(op, Split):
            sizes = map(self.constr.value_of, ins[0].axes[op.axis_num].children)
            out = []
            curr = 0            
            for size in sizes:
                slc = [slice(None) for _ in range(x[0].ndim)]
                slc[op.axis_num] = slice(curr, curr + size)
                out.append(x[0][tuple(slc)])
                curr += size
            return out
        elif isinstance(op, Concat):
            return [aa.concat(x, axis=op.axis_num)]
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
        elif isinstance(op, Combine):
            return [op.method.reduce(aa.stack(aa.broadcast_arrays(*x)))]
        elif isinstance(op, Reduce):
            return [op.method.reduce(x[0], axis=ins[0].axes.index(op.axis))]