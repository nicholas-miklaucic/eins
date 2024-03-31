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


def test_sum():
    (x, y), xy = run_with_randn('a b, c b -> a+c b', (4, 5), (6, 5))
    assert np.allclose(xy, np.concatenate((x, y), axis=0))


def test_sum_after_mat():
    (ab, cb, bd), acd = run_with_randn('a b, c b, b d -> a+c d', (4, 5), (6, 5), (5, 2))
    assert np.allclose(acd, np.concatenate((ab, cb), axis=0) @ bd)


def test_big_kahuna():
    run_with_randn(
        'b ((n p) (n p)) c d=c, b p*p*d*c h, h g k, i (h k) -> b (n^2 g+i) k',
        (6, 64, 3, 3),
        (6, 36, 7),
        (7, 1, 8),
        (3, 56),
    )
