"""Constraint solver for einops expressions."""

from typing import Mapping, Optional, Union

from eins.parsing import Constant, Expr, Node, Symbol, expr, flatten, make_expr, unpack_shorthands

MAX_STEPS = 100


class Constraints:
    def __init__(self):
        self.equations = []
        self.free_vars = []
        self.known_vars = {}

    def __repr__(self) -> str:
        lines = []
        for lhs, rhs in self.equations:
            lines.append(f'{lhs!s:>20} = {rhs!s}')

        lines.append(str(self.free_vars))
        lines.append(str(self.known_vars))
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

    def add_variables(self, variables: list[str]):
        for v in variables:
            if v in self.known_vars:
                continue
            elif v in self.free_vars:
                continue
            else:
                self.free_vars.append(v)

    def fill_in(self, values: Mapping[Symbol, int]):
        for k, v in values.items():
            self.free_vars.remove(k.value)
            self.known_vars[k.value] = v

    def value_of(self, node: Union[Node, str, int]) -> Optional[int]:
        if isinstance(node, str):
            return self.known_vars.get(node)
        elif isinstance(node, int):
            return node
        elif isinstance(node, Symbol):
            return self.known_vars.get(node.value)
        elif isinstance(node, Constant):
            return node.value
        else:
            match node.op:
                case '+':
                    s = 0
                    for child in node.children:
                        if self.value_of(child) is not None:
                            s += self.value_of(child)
                        else:
                            return None
                    return s
                case '*':
                    p = 1
                    for child in node.children:
                        if self.value_of(child) is not None:
                            p *= self.value_of(child)
                        else:
                            return None
                    return p

    def reduce_eqn(self, lhs, rhs):
        inv = {'+': lambda x, y: x - y, '*': lambda x, y: x // y}

        log = {'+': lambda x, n: x // n, '*': lambda x, n: int(x ** (1 / n))}
        lhs_val = self.value_of(lhs)
        rhs_val = self.value_of(rhs)
        if lhs_val is not None and rhs_val is not None:
            if lhs_val != rhs_val:
                msg = f"""Incompatible shapes given.
The operation requires that {lhs} = {rhs}, but the deduced values of {lhs_val} and {rhs_val} are not equal.

Deduced values:
""" + str(self.known_vars)
                raise ValueError(msg)
            return []
        elif lhs_val is not None:
            return self.reduce_eqn(rhs, lhs)
        elif (lhs_val, rhs_val) == (None, None):
            return [(lhs, rhs)]

        if isinstance(lhs, Expr):
            unknowns = []
            new_rhs = rhs_val
            for child in lhs.children:
                val = self.value_of(child)
                if val is None:
                    unknowns.append(child)
                else:
                    new_rhs = inv[lhs.op](new_rhs, val)

            # special case that requires attention:
            # (n n n) = 64 is solvable
            if all(isinstance(x, Symbol) for x in unknowns):
                if len(unknowns) == 1:
                    self.fill_in({unknowns[0]: new_rhs})
                    return []
                else:
                    # (n n) = 100 gets preprocessed to (n n-2) = 100, n-2 = n

                    # we want to reduce this down to (n n) so it gets recognized as solvable
                    reduced_unknowns = []
                    for unk in unknowns:
                        for eq_lhs, eq_rhs in self.equations:
                            if unk == eq_lhs:
                                reduced_unknowns.append(eq_rhs)
                                break
                        else:
                            reduced_unknowns.append(unk)

                    if len(set(reduced_unknowns)) == 1:
                        # special case: we can solve (n n) = 100, for example
                        self.fill_in({reduced_unknowns[0]: log[lhs.op](new_rhs, len(unknowns))})
                        return []
                    else:
                        # even if we could solve, say, (n+4 n+4 p) = 75 as n=1, p=3, assuming n is natural, we don't
                        # want to. That's a recipe for subtle data breaking later, in a way that is very hard to defend
                        # against. You suddenly go from having 3 channels to 4, and now eins assumes 1 channel and twice
                        # the input size!
                        return [(Expr(lhs.op, unknowns), new_rhs)]
            else:
                return [(Expr(lhs.op, unknowns), new_rhs)]
        elif isinstance(lhs, Symbol):
            self.fill_in({lhs: rhs_val})
            return []
        else:
            return []

    def solve(self):
        step = 0
        while self.free_vars and step < MAX_STEPS:
            step += 1
            new_equations = []
            for eqn in self.equations:
                new_equations.extend(self.reduce_eqn(*eqn))

            if new_equations == self.equations:
                break
            else:
                self.equations = new_equations

        if step == MAX_STEPS:
            msg = f'Solution failed!\n{self}'
            raise ValueError(msg)


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


# unpacked = unpack_shorthands('a (b b c), (2 c) -> a 2')
# # unpacked = unpack_shorthands('b ((n p) (n p)) c d=c, b p*p*d*c h, h[g k], h[i k] -> b (n^2 g+i) k')
# # unpacked = unpack_shorthands('b ((n p) (n p)) c d=c, b p*p*d*c h, h g k, i (g k) -> b (n^2 g+i) k')
# ast = make_expr(expr.parse_string(unpacked).as_list())
# constr = postprocess_ast(ast)
# print(unpacked)
# print(ast)
# print(constr)

# b, p, d, c, h, i, k, n, g = map(Symbol, 'bpdchikng')
# C = Constant

# # 13 100 4 4, 13 64 3, 7 8, 9 8

# constr.add_constraint(b, 13)
# constr.add_constraint(Expr('*', [n, p, n, p]), Constant(100))
# constr.add_constraint(c, 4)
# constr.add_constraint(d, 4)
# constr.add_constraint(Expr('*', [p, p, d, c]), Constant(64))
# constr.add_constraint(h, 3)
# constr.add_constraint(g, 7)
# constr.add_constraint(k, 8)
# constr.add_constraint(i, 9)
# constr.add_variables(list('bpdchikng'))
# constr.add_variables(['n-2', 'p-2'])
# constr.solve()
# print(constr)
