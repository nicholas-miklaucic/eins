"""Parsing the einops expression into a structured form with constraint variables."""

import pprint
from collections import defaultdict
from dataclasses import dataclass

import pyparsing as pp
from pyparsing import DelimitedList
from pyparsing import pyparsing_common as ppc

pp.ParserElement.enablePackrat()
pp.ParserElement.setDefaultWhitespaceChars('')

integer = ppc.integer
symbol = ppc.identifier
operand = (integer | symbol).set_whitespace_chars('')


def wrap_ws(elem):
    ws = pp.Suppress(pp.ZeroOrMore(pp.one_of(list(' \t\n\r'))))
    return ws + elem + ws


arrow = wrap_ws('->')
delimit = wrap_ws(',')
eq = pp.Suppress('=')
lpar = pp.Suppress('(')
rpar = pp.Suppress(')')
lbrack = pp.Suppress('[')
rbrack = pp.Suppress(']')
space = pp.Suppress(' ')
paren_op = pp.one_of('* +')

add_op = pp.one_of('+')
sub_op = pp.one_of('-')
mul_op = pp.one_of('*')
div_op = pp.one_of('/')
pow_op = pp.one_of('^')

ops = [add_op, sub_op, mul_op, div_op, pow_op]
for i, op in enumerate(ops):
    ops[i] = op.set_whitespace_chars('')

dim_expr = pp.Group(
    pp.infix_notation(
        operand,
        [
            (ops[-1], 2, pp.opAssoc.RIGHT),
            (ops[-2], 2, pp.opAssoc.LEFT),
            (ops[-3], 2, pp.opAssoc.LEFT),
            (ops[-4], 2, pp.opAssoc.LEFT),
            (ops[-5], 2, pp.opAssoc.LEFT),
        ],
        '#(',
        ')#',
    )
)('dim_expr')

dim_seq = pp.Group(pp.DelimitedList(dim_expr, space))

grouping = pp.Group(lpar + dim_seq('grouped_dims') + pp.Opt(paren_op, default='*')('group_op') + rpar)

dim_no_constr = pp.Group(dim_expr | grouping)
dim = pp.Group(dim_no_constr('eq_lhs') + eq + dim_no_constr('eq_rhs'))('constr') | pp.Group(
    dim_no_constr('dim_no_constr')
)

arr_shape = pp.Group(pp.DelimitedList(dim, space, min=1))('arr_shape')
index_arr = pp.Group(pp.Group(dim('index_dim')) + lbrack + pp.Group(arr_shape('inner')) + rbrack)('index_arr')
arr = pp.Group(index_arr | arr_shape)
arr_list = pp.Group(DelimitedList(arr, delimit, min=1))

expr = arr_list('lhs') + arrow + arr('rhs')


def parse_einop(in_str):
    return expr.parse_string(in_str).as_dict()


parse = parse_einop('a b=a c , c[a k] , b[a g] -> a*g+1 k')
pprint.pprint(parse)
