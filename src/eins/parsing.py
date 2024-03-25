import pprint
import re
from dataclasses import dataclass
from typing import Callable, Union

import pyparsing as pp
import pyparsing.common as ppc

lparen = '('
rparen = ')'
parens = re.compile(r'\{\s*([^{} ,]+ )*?([^{}, ]+)\s*\}')
index = re.compile(r'([^[ ,]+)\[([^\]]*)\]')
pows_paren = re.compile(r'(\{[^{}]+\})\^(\d+)')
pows_atomic = re.compile(r'([^{} +*/-]+)\^(\d+)')


def unpack_parens(m: re.Match):
    return lparen + '*'.join(f'{lparen}{g.strip()}{rparen}' for g in m.groups() if g is not None) + rparen


def unpack_index(m: re.Match):
    ind, *rest = m.groups()
    return ' '.join(rest) + f' @ {ind}'


def unpack_pow(m: re.Match):
    left, right = m.groups()
    return '{' + '*'.join(left.strip() for g in range(int(right))) + '}'


MAX_NESTED_PARENS = 100


def unpack_shorthands(expr: str):
    expr = expr.replace('(', '{').replace(')', '}')
    expr = re.sub(pows_paren, unpack_pow, expr)
    expr = re.sub(pows_atomic, unpack_pow, expr)

    new_expr = re.sub(parens, unpack_parens, expr)
    num_times = 0
    while new_expr != expr:
        num_times += 1
        if num_times > MAX_NESTED_PARENS:
            msg = 'Stack overflow'
            raise ValueError(msg)
        new_expr, expr = re.sub(parens, unpack_parens, new_expr), new_expr

    expr = re.sub(index, unpack_index, new_expr)

    return expr


# parse = parse_einop('b (d=(n p) d) c, b p*p*c h, h[k] -> b n n k')
# parse = parse_einop('b k=(a a) a -> a b')
# pprint.pprint(parse)

unpacked = unpack_shorthands('b ( d=(n p ) d) c, b p*p*c h, h[g k], h[i k] -> b (n^2 g+i) k')

# b ((d=((n)*(p)))*(d)) c, b p*p*c h, h[] k -> b n n k


pp.ParserElement.enable_packrat()
pp.ParserElement.set_default_whitespace_chars('\n\t\r')
spaces = pp.Suppress(pp.ZeroOrMore(' '))
eq_op = pp.one_of('=')
comma_op = spaces + pp.Literal(',') + spaces
seq_op = pp.Literal(' ')
add_op = pp.one_of('+ -')
mul_op = pp.one_of('* /')
pow_op = pp.one_of('^')
arrow = spaces + pp.Literal('->') + spaces
index_op = spaces + pp.one_of('@') + spaces

symbol = pp.Word(pp.pyparsing_unicode.identchars, pp.pyparsing_unicode.identbodychars, exclude_chars='+-*/^()[] ->')
literal = pp.Word(pp.nums)

operand = symbol | literal

expr = pp.infix_notation(
    operand,
    [
        (pow_op, 2, pp.OpAssoc.LEFT),
        (mul_op, 2, pp.OpAssoc.LEFT),
        (add_op, 2, pp.OpAssoc.LEFT),
        (seq_op, 2, pp.OpAssoc.LEFT),
        (eq_op, 2, pp.OpAssoc.LEFT),
        (index_op, 2, pp.OpAssoc.LEFT),
        (comma_op, 2, pp.OpAssoc.LEFT),
        (arrow, 2, pp.OpAssoc.LEFT),
    ],
)

print(unpacked)
pprint.pprint(expr.parse_string(unpacked).as_list())


@dataclass
class Constant:
    value: int

    def __repr__(self):
        return repr(self.value)


@dataclass
class Symbol:
    value: str

    def __repr__(self):
        return repr(self.value)


Node = Union[Constant, Symbol, 'Expr']


@dataclass
class Expr:
    op: str
    children: list[Node]

    def tree_map(self, op: Callable) -> 'Expr':
        op(self)
        for child in self.children:
            if isinstance(child, Expr):
                child.tree_map(op)
            else:
                op(child)
        return self

    def replace_with(self, new: 'Expr'):
        self.op, self.children = new.op, new.children


def make_expr(parsed: list | str) -> Expr:
    if isinstance(parsed, str):
        if parsed.isdigit():
            return Constant(int(parsed))
        else:
            return Symbol(parsed)
    else:
        if len(parsed) == 1:
            return make_expr(parsed[0])

        lhs, op, *rhs = parsed
        return Expr(op, [make_expr(lhs), make_expr(rhs)])


# ast = make_expr(expr.parse_string(unpacked).as_list())
# pprint.pprint(ast)

equations = []


def flatten(node: Node) -> Node:
    if isinstance(node, Expr):
        can_flatten = True
        while can_flatten:
            to_flatten = []
            for i, child in enumerate(node.children):
                if isinstance(child, Expr) and child.op == node.op:
                    to_flatten.append(i)

            for i in to_flatten[::-1]:
                node.children = node.children[:i] + node.children[i].children + node.children[i + 1 :]

            can_flatten = len(to_flatten) > 0

    return node


class Constraints:
    def __init__(self):
        self.equations = []

    def process_constraints(self, node: Node):
        if isinstance(node, Expr) and node.op == '=':
            if len(node.children) != 2:  # noqa: PLR2004
                msg = 'Must have exactly two parts to equality'
                raise ValueError(msg)
            lhs, rhs = node.children
            self.equations.append((lhs, rhs))
            node.replace_with(rhs)

    def replace_referents(self, node: Node):
        if isinstance(node, Expr):
            for i, child in enumerate(node.children):
                for lhs, rhs in self.equations:
                    if child == lhs:
                        node.children[i] = rhs


def postprocess_ast(ast: Node):
    constraints = Constraints()
    for func in (constraints.process_constraints, constraints.replace_referents, flatten):
        ast.tree_map(func)

    return constraints


# pprint.pprint(ast)
# pprint.pprint(equations)
