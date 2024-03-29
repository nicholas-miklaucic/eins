import numpy as np
import pytest
from pyparsing import ParseException

from eins.einsop import einsop
from eins.parsing import expr


def run_with_randn(op_str: str, *shapes, **kwargs):
    tensors = [np.random.randn(*shape) for shape in shapes]
    return tensors, einsop(op_str, *tensors, **kwargs)


def test_matrix_mul():
    (x, y), z1 = run_with_randn('a b, b c -> a c', (4, 5), (5, 6))
    z2 = x @ y
    assert np.allclose(z1, z2)


def test_big_kahuna():
    run_with_randn(
        'b ((n p) (n p)) c d=c, b p*p*d*c h, h g k -> b (n^2 g) k',
        (6, 64, 3, 3),
        (6, 36, 7),
        (7, 1, 8),
    )
