import pytest
from pyparsing import ParseException

from eins.parsing import expr


def test_basic_parses():
    cases = {
        'a b, b c -> a c': [[['a', 'b'], ['b', 'c']], ['a', 'c']],
        'b (d d d) c -> b c': [[['b', [['d', 'd', 'd'], '*'], 'c']], ['b', 'c']],
        """batch height width chan_in, chan_in chan_out ->
batch (height width) chan_out""": [
            [['batch', 'height', 'width', 'chan_in'], ['chan_in', 'chan_out']],
            ['batch', [['height', 'width'], '*'], 'chan_out'],
        ],
    }

    failures = {'b (d, b) c -> b c', 'b (d -> c) a', '!@# b -> a', ''}
    for case, expected in cases.items():
        assert expr.parse_string(case).as_list() == expected

    for failure in failures:
        pytest.raises(ParseException, expr.parse_string, failure)
