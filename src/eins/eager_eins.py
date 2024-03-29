"""Putting it all together."""

from multiprocessing import Value

from eins.concrete import ArrayBackend
from eins.parsing import Constant
from eins.symbolic import Program, Tensor


def eager_eins(op_str: str, *tensors):
    prog = Program.parse(op_str)
    for concrete, shape in zip(tensors, prog.sources):
        for lhs, rhs in zip(concrete.shape, shape.axes):
            prog.constr.add_constraint(Constant(lhs), rhs)

    prog.constr.solve()
    if prog.constr.free_vars:
        # print(prog.constr)
        msg = f'Could not solve: free variables {prog.constr.free_vars} remain'
        raise ValueError(msg)

    # print(prog.constr)
    backend = ArrayBackend(prog.constr)

    abstract = {}
    concrete = {}
    frontier = {}

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
                    frontier[key] = (op, (child,))
            else:
                frontier[(id(src),)] = (op, tuple(children))
        return False

    for src, arr in zip(prog.sources, tensors):
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
                op, outputs = frontier[input_ids]
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
        print(list(map(id, prog.sinks)))
        raise ValueError(msg)
    return None
