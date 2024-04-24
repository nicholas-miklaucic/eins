import numpy as np

from .utils import COMBINE_OPS, ELEM_OPS, REDUCE_OPS, SEEDS, SIZES


def commutes_elemwise(seed: int = 123):
    xs = []
    ys = []
    rng = np.random.default_rng(seed)
    for n in SIZES:
        xs.append(rng.standard_normal(n))
        ys.append(rng.standard_normal(n))

        for elem_name, elem in ELEM_OPS.items():
            if 'bitwise' in elem_name:
                continue

            for x_base, y_base in zip(xs, ys):
                x = x_base.copy()
                y = y_base.copy()
                if elem_name in ('acosh', 'log', 'log1p', 'log2', 'log10', 'sqrt'):
                    x = np.abs(x)
                    y = np.abs(y)
                    if elem_name == 'acosh':
                        x += 1
                        y += 1
                elif elem_name in ('atanh', 'asin', 'acos'):
                    x = np.tanh(x)
                    y = np.tanh(y)
                elif elem_name in ('conj', 'imag', 'real'):
                    x = x + 1j * np.flip(x)
                    y = y + 1j * np.flip(y)

                for combo_name, combo in COMBINE_OPS.items():
                    if (
                        'bitwise' in combo_name
                        or combo_name not in ('add', 'multiply')
                        and elem_name in ('conj', 'imag', 'real')
                    ):
                        continue

                    try:
                        np.seterr('raise')
                        z1 = combo(elem(x), elem(y))
                        z2 = elem(combo(x, y))
                        does_commute = np.allclose(z1, z2)
                    except FloatingPointError:
                        does_commute = False

                    assert does_commute or not elem.commutes_with(
                        combo
                    ), f'{combo_name}, {elem_name}'

                for reduce_name, reduce in REDUCE_OPS.items():
                    for a in (x, y):
                        try:
                            np.seterr('raise')
                            z1 = reduce(elem(a), axis=0)
                            z2 = elem(reduce(a, axis=0))
                            does_commute = np.allclose(z1, z2)
                        except FloatingPointError:
                            does_commute = False

                        assert does_commute or not elem.commutes_with(
                            reduce
                        ), f'{reduce_name}, {elem_name}'


def test_commutes_elemwise():
    for seed in SEEDS:
        commutes_elemwise(seed)
