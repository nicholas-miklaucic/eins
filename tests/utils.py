from copy import deepcopy

import numpy as np

from eins import Combinations as C
from eins import ElementwiseOps as E
from eins import Reductions as R
from eins import Transformations as T
from eins.combination import ops as combine_ops
from eins.elementwise import ops as elem_ops
from eins.reduction import ops as reduce_ops
from eins.transformation import ops as transform_ops

COMBINE_OPS = deepcopy(combine_ops)

ELEM_OPS = deepcopy(elem_ops)

reals = (-10, -2, -1, -0.5, 0, 0.5, 1, 2, 10)
for scale in reals:
    for shift in reals:
        ELEM_OPS[f'x*{scale}+{shift}'] = E.Affine(scale=scale, shift=shift)

REDUCE_OPS = deepcopy(reduce_ops)

norms = (-1, 1, -1.5, 1.5, -2, 2, -3, 3, -4, 4, float('-inf'), float('inf'))
for norm in norms:
    REDUCE_OPS[f'{norm}-norm'] = R.PowerNorm(power=norm)

TRANSFORM_OPS = deepcopy(transform_ops)

for norm in norms:
    TRANSFORM_OPS[f'{norm}-normalize'] = T.PowerNormalize(power=norm)

for temp in (0.5, 2):
    TRANSFORM_OPS[f'softmax-{temp}'] = T.Softmax(temperature=temp)

SEEDS = np.random.randint(0, 10000, 3)
SIZES = (2, 3, 5, 10, 100)
