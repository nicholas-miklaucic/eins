"""Constraint solver for einops expressions."""

from typing import Callable, Mapping, MutableSequence, Optional, Union

from eins.parsing import Constant, Expr, Node, Symbol, flatten

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
        if isinstance(node, Expr):
            for i, child in enumerate(node.children):
                if isinstance(child, Expr) and child.op == '=':
                    if len(child.children) != 2:
                        msg = 'Must have exactly two parts to equality'
                        raise ValueError(msg)
                    lhs, rhs = child.children
                    self.equations.append((lhs, rhs))
                    node.children[i] = lhs

    def replace_referents(self, node: Node):
        if isinstance(node, Expr):
            for i, child in enumerate(node.children):
                for lhs, rhs in self.equations:
                    if child == lhs:
                        node.children[i] = rhs

    def replace_constants(self, node: Node):
        if isinstance(node, Expr):
            for i, child in enumerate(node.children):
                if isinstance(child, Constant):
                    sym = Symbol(str(child.value))
                    node.children[i] = Symbol(str(child.value))
                    self.equations.append((sym, child))

    def disambiguate_axes(self, node: Node, curr_axes: Optional[MutableSequence[str]] = None):
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
                # if you ever change this, you need to change the logic in Program that adds
                # the constraints!
                node_value = f'{orig_value}-{num}'

            curr_axes.append(node_value)
            if num > 1:
                new_node = Symbol(node_value)
                self.add_constraint(new_node, node)
                return new_node

        return node

    def add_variables(self, variables: MutableSequence[str]):
        for v in variables:
            if v in self.known_vars:
                continue
            elif v in self.free_vars:
                continue
            else:
                self.free_vars.append(v)

    def fill_in(self, values: Mapping[Symbol, int]):
        for k, v in values.items():
            if k.value in self.free_vars:
                self.free_vars.remove(k.value)
            else:
                old_v = self.known_vars[k.value]
                if old_v != v:
                    msg = f'Conflicting values for {k.value}: {old_v} and {v}'
                    raise ValueError(msg)
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
        elif node.op == '+':
            s = 0
            for child in node.children:
                v = self.value_of(child)
                if v is not None:
                    s += v
                else:
                    return None
            return s
        elif node.op == '*':
            p = 1
            for child in node.children:
                v = self.value_of(child)
                if v is not None:
                    p *= v
                else:
                    return None
            return p
        else:
            return None

    def reduce_eqn(self, lhs, rhs):
        inv: dict[str, Callable[[int, int], int]] = {
            '+': lambda x, y: x - y,
            '*': lambda x, y: x // y,
        }

        log: dict[str, Callable[[int, int], int]] = {
            '+': lambda x, n: x // n,
            '*': lambda x, n: int(x ** (1 / n)),
        }
        lhs_val = self.value_of(lhs)
        rhs_val = self.value_of(rhs)
        if lhs_val is not None and rhs_val is not None:
            if lhs_val != rhs_val:
                eqns = '\n'.join(f'{lhs} = {rhs}' for lhs, rhs in self.equations)
                msg = f"""Incompatible shapes given. The operation requires that {lhs} = {rhs}, but
the deduced values of {lhs_val} and {rhs_val} are not equal.

Deduced values: {self.known_vars}
Equations:\n{eqns}
"""
                raise ValueError(msg)
            return []
        elif lhs_val is not None:
            return self.reduce_eqn(rhs, lhs)
        elif (lhs_val, rhs_val) == (None, None):
            return [(lhs, rhs)]
        elif rhs_val is not None:
            if isinstance(lhs, Expr):
                unknowns = []
                new_rhs = rhs_val
                for child in lhs.children:
                    val = self.value_of(child)
                    if val is None:
                        unknowns.append(child)
                    else:
                        new_rhs = inv[lhs.op](new_rhs, val)

                # special case that requires attention: (n n n) = 64 is solvable
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
                            # even if we could solve, say, (n+4 n+4 p) = 75 as n=1, p=3, assuming n
                            # is natural, we don't want to. That's a recipe for subtle data breaking
                            # later, in a way that is very hard to defend against. You suddenly go
                            # from having 3 channels to 4, and now eins assumes 1 channel and twice
                            # the input size!
                            return [(Expr(lhs.op, unknowns), new_rhs)]
                else:
                    return [(Expr(lhs.op, unknowns), new_rhs)]
            elif isinstance(lhs, Symbol):
                self.fill_in({lhs: rhs_val})
                return []
            else:
                return []
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


def postprocess_ast(ast: Expr):
    constraints = Constraints()
    for func in (
        constraints.process_constraints,
        # constraints.replace_referents,
        flatten,
        constraints.replace_constants,
        constraints.disambiguate_axes,
    ):
        ast.tree_map(func)

    return constraints
