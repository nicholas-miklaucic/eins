# {py:mod}`eins.constraint`

```{py:module} eins.constraint
```

```{autodoc2-docstring} eins.constraint
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Constraints <eins.constraint.Constraints>`
  - ```{autodoc2-docstring} eins.constraint.Constraints
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`postprocess_ast <eins.constraint.postprocess_ast>`
  - ```{autodoc2-docstring} eins.constraint.postprocess_ast
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAX_STEPS <eins.constraint.MAX_STEPS>`
  - ```{autodoc2-docstring} eins.constraint.MAX_STEPS
    :summary:
    ```
````

### API

````{py:data} MAX_STEPS
:canonical: eins.constraint.MAX_STEPS
:value: >
   100

```{autodoc2-docstring} eins.constraint.MAX_STEPS
```

````

`````{py:class} Constraints()
:canonical: eins.constraint.Constraints

```{autodoc2-docstring} eins.constraint.Constraints
```

```{rubric} Initialization
```

```{autodoc2-docstring} eins.constraint.Constraints.__init__
```

````{py:method} __repr__() -> str
:canonical: eins.constraint.Constraints.__repr__

````

````{py:method} add_constraint(lhs: eins.parsing.Node, rhs: eins.parsing.Node)
:canonical: eins.constraint.Constraints.add_constraint

```{autodoc2-docstring} eins.constraint.Constraints.add_constraint
```

````

````{py:method} process_constraints(node: eins.parsing.Node)
:canonical: eins.constraint.Constraints.process_constraints

```{autodoc2-docstring} eins.constraint.Constraints.process_constraints
```

````

````{py:method} replace_referents(node: eins.parsing.Node)
:canonical: eins.constraint.Constraints.replace_referents

```{autodoc2-docstring} eins.constraint.Constraints.replace_referents
```

````

````{py:method} disambiguate_axes(node: eins.parsing.Node, curr_axes: typing.Optional[list[eins.parsing.Node]] = None)
:canonical: eins.constraint.Constraints.disambiguate_axes

```{autodoc2-docstring} eins.constraint.Constraints.disambiguate_axes
```

````

````{py:method} add_variables(variables: list[str])
:canonical: eins.constraint.Constraints.add_variables

```{autodoc2-docstring} eins.constraint.Constraints.add_variables
```

````

````{py:method} fill_in(values: typing.Mapping[eins.parsing.Symbol, int])
:canonical: eins.constraint.Constraints.fill_in

```{autodoc2-docstring} eins.constraint.Constraints.fill_in
```

````

````{py:method} value_of(node: typing.Union[eins.parsing.Node, str, int]) -> typing.Optional[int]
:canonical: eins.constraint.Constraints.value_of

```{autodoc2-docstring} eins.constraint.Constraints.value_of
```

````

````{py:method} reduce_eqn(lhs, rhs)
:canonical: eins.constraint.Constraints.reduce_eqn

```{autodoc2-docstring} eins.constraint.Constraints.reduce_eqn
```

````

````{py:method} solve()
:canonical: eins.constraint.Constraints.solve

```{autodoc2-docstring} eins.constraint.Constraints.solve
```

````

`````

````{py:function} postprocess_ast(ast: eins.parsing.Node)
:canonical: eins.constraint.postprocess_ast

```{autodoc2-docstring} eins.constraint.postprocess_ast
```
````
