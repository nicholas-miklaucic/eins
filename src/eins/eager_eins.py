"""Putting it all together."""


from regex import P
from eins.concrete import ArrayBackend
from eins.parsing import Constant
from eins.symbolic import Program, Tensor
import numpy as np


def eager_eins(op_str: str, *tensors):
    prog = Program.parse(op_str)
    for concrete, shape in zip(tensors, prog.sources):        
        for lhs, rhs in zip(concrete.shape, shape.axes):
            prog.constr.add_constraint(Constant(lhs), rhs)
    
    prog.constr.solve()
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

    while frontier:        
        input_ids, (op, outputs) = next(iter(frontier.items()))
        if all(input_id in concrete for input_id in input_ids):                                    
            x = [concrete[input_id] for input_id in input_ids]
            i = [abstract[input_id] for input_id in input_ids]
            o = list(outputs)
            concrete_outputs = backend.do(x, op, i, o)
            # print('co', concrete_outputs)
            for concrete_out, out in zip(concrete_outputs, outputs):
                if fill(out, concrete_out, frontier):
                    return concrete_out

            del frontier[input_ids]  
        else:
            pass
            # print(input_ids, concrete.keys())   

    return None


import numpy.array_api as xp
X = xp.reshape(xp.linspace(0, 1, 3 * 7), (3, 7))
Y = xp.reshape(xp.linspace(0, 1, 7 * 4), (7, 4))
Z = X @ Y
Z2 = eager_eins('a b, b c -> a c', X, Y)
print(xp.mean(Z2 - Z))