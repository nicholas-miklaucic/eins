import pprint
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Union

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
        (eq_op, 2, pp.OpAssoc.LEFT),
        (seq_op, 2, pp.OpAssoc.LEFT),
        (index_op, 2, pp.OpAssoc.LEFT),
        (comma_op, 2, pp.OpAssoc.LEFT),
        (arrow, 2, pp.OpAssoc.LEFT),
    ],
)


@dataclass(unsafe_hash=True)
class Constant:
    value: int

    def __repr__(self):
        return str(self.value)


@dataclass(unsafe_hash=True)
class Symbol:
    value: str

    def __repr__(self):
        return str(self.value)


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

    def __str__(self):
        return f'({self.op} ' + ' '.join(map(str, self.children)) + ')'


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

    def __repr__(self) -> str:
        lines = []
        for lhs, rhs in self.equations:
            lines.append(f'{lhs!s:>20} = {rhs!s}')
        return '\n'.join(lines)

    def add_constraint(self, lhs: Node, rhs: Node):
        if (lhs, rhs) not in self.equations:
            self.equations.append((lhs, rhs))

    def process_constraints(self, node: Node):
        if isinstance(node, Expr) and node.op == '=':
            if len(node.children) != 2:  # noqa: PLR2004
                msg = 'Must have exactly two parts to equality'
                raise ValueError(msg)
            lhs, rhs = node.children
            self.equations.append((lhs, rhs))
            node.replace_with(Expr(' ', [lhs]))

    def replace_referents(self, node: Node):
        if isinstance(node, Expr):
            for i, child in enumerate(node.children):
                for lhs, rhs in self.equations:
                    if child == lhs:
                        node.children[i] = rhs

    def disambiguate_axes(self, node: Node, curr_axes: Optional[list[Node]] = None):
        if curr_axes is None:
            curr_axes = []
        if isinstance(node, Expr):
            if node.op in (',', '->', '@'):
                for i, child in enumerate(node.children):
                    new = self.disambiguate_axes(child, curr_axes=[])
                    if new != child:
                        node.children[i] = new
            elif node.op in ('+', '*', ' ', '^'):
                for i, child in enumerate(node.children):
                    new = self.disambiguate_axes(child, curr_axes=curr_axes)
                    if new != child:
                        node.children[i] = new
        elif isinstance(node, Symbol):
            num = 1
            orig_value = node.value
            node_value = orig_value
            while node_value in curr_axes:
                num += 1
                # use - because it's not allowed in user identifiers
                node_value = f'{orig_value}-{num}'

            curr_axes.append(node_value)
            if num > 1:
                new_node = Symbol(node_value)
                self.add_constraint(new_node, node)
                return new_node
        return node


def postprocess_ast(ast: Node):
    constraints = Constraints()
    for func in (
        constraints.process_constraints,
        # constraints.replace_referents,
        flatten,
        constraints.disambiguate_axes,
    ):
        ast.tree_map(func)

    return constraints


# unpacked = unpack_shorthands('b ((n p) (n p)) c d=c, b p*p*c*d h, h[g k], h[i k] -> b (n^2 g+i) k')
# print(unpacked)
# pprint.pprint(expr.parse_string(unpacked).as_list())
# ast = make_expr(expr.parse_string(unpacked).as_list())
# print(ast)
# constr = postprocess_ast(ast)
# print(ast)
# print(constr)
