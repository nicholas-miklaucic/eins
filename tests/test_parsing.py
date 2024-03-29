import re

from eins.parsing import parens, unpack_parens, unpack_shorthands


def test_unpack_parens():
    expr = 'a (b b c), (2 c) -> a 2'.replace('(', '{').replace(')', '}')
    unpacked = re.sub(parens, unpack_parens, expr)
    assert unpacked == 'a ((b)*(b)*(c)), ((2)*(c)) -> a 2'
