# {py:mod}`eins.parsing`

```{py:module} eins.parsing
```

```{autodoc2-docstring} eins.parsing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Constant <eins.parsing.Constant>`
  - ```{autodoc2-docstring} eins.parsing.Constant
    :summary:
    ```
* - {py:obj}`Symbol <eins.parsing.Symbol>`
  - ```{autodoc2-docstring} eins.parsing.Symbol
    :summary:
    ```
* - {py:obj}`Expr <eins.parsing.Expr>`
  - ```{autodoc2-docstring} eins.parsing.Expr
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`unpack_parens <eins.parsing.unpack_parens>`
  - ```{autodoc2-docstring} eins.parsing.unpack_parens
    :summary:
    ```
* - {py:obj}`unpack_index <eins.parsing.unpack_index>`
  - ```{autodoc2-docstring} eins.parsing.unpack_index
    :summary:
    ```
* - {py:obj}`unpack_pow <eins.parsing.unpack_pow>`
  - ```{autodoc2-docstring} eins.parsing.unpack_pow
    :summary:
    ```
* - {py:obj}`unpack_shorthands <eins.parsing.unpack_shorthands>`
  - ```{autodoc2-docstring} eins.parsing.unpack_shorthands
    :summary:
    ```
* - {py:obj}`make_expr <eins.parsing.make_expr>`
  - ```{autodoc2-docstring} eins.parsing.make_expr
    :summary:
    ```
* - {py:obj}`flatten <eins.parsing.flatten>`
  - ```{autodoc2-docstring} eins.parsing.flatten
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`lparen <eins.parsing.lparen>`
  - ```{autodoc2-docstring} eins.parsing.lparen
    :summary:
    ```
* - {py:obj}`rparen <eins.parsing.rparen>`
  - ```{autodoc2-docstring} eins.parsing.rparen
    :summary:
    ```
* - {py:obj}`parens <eins.parsing.parens>`
  - ```{autodoc2-docstring} eins.parsing.parens
    :summary:
    ```
* - {py:obj}`index <eins.parsing.index>`
  - ```{autodoc2-docstring} eins.parsing.index
    :summary:
    ```
* - {py:obj}`pows_paren <eins.parsing.pows_paren>`
  - ```{autodoc2-docstring} eins.parsing.pows_paren
    :summary:
    ```
* - {py:obj}`pows_atomic <eins.parsing.pows_atomic>`
  - ```{autodoc2-docstring} eins.parsing.pows_atomic
    :summary:
    ```
* - {py:obj}`MAX_NESTED_PARENS <eins.parsing.MAX_NESTED_PARENS>`
  - ```{autodoc2-docstring} eins.parsing.MAX_NESTED_PARENS
    :summary:
    ```
* - {py:obj}`spaces <eins.parsing.spaces>`
  - ```{autodoc2-docstring} eins.parsing.spaces
    :summary:
    ```
* - {py:obj}`eq_op <eins.parsing.eq_op>`
  - ```{autodoc2-docstring} eins.parsing.eq_op
    :summary:
    ```
* - {py:obj}`comma_op <eins.parsing.comma_op>`
  - ```{autodoc2-docstring} eins.parsing.comma_op
    :summary:
    ```
* - {py:obj}`seq_op <eins.parsing.seq_op>`
  - ```{autodoc2-docstring} eins.parsing.seq_op
    :summary:
    ```
* - {py:obj}`add_op <eins.parsing.add_op>`
  - ```{autodoc2-docstring} eins.parsing.add_op
    :summary:
    ```
* - {py:obj}`mul_op <eins.parsing.mul_op>`
  - ```{autodoc2-docstring} eins.parsing.mul_op
    :summary:
    ```
* - {py:obj}`pow_op <eins.parsing.pow_op>`
  - ```{autodoc2-docstring} eins.parsing.pow_op
    :summary:
    ```
* - {py:obj}`arrow <eins.parsing.arrow>`
  - ```{autodoc2-docstring} eins.parsing.arrow
    :summary:
    ```
* - {py:obj}`index_op <eins.parsing.index_op>`
  - ```{autodoc2-docstring} eins.parsing.index_op
    :summary:
    ```
* - {py:obj}`symbol <eins.parsing.symbol>`
  - ```{autodoc2-docstring} eins.parsing.symbol
    :summary:
    ```
* - {py:obj}`literal <eins.parsing.literal>`
  - ```{autodoc2-docstring} eins.parsing.literal
    :summary:
    ```
* - {py:obj}`operand <eins.parsing.operand>`
  - ```{autodoc2-docstring} eins.parsing.operand
    :summary:
    ```
* - {py:obj}`expr <eins.parsing.expr>`
  - ```{autodoc2-docstring} eins.parsing.expr
    :summary:
    ```
* - {py:obj}`Node <eins.parsing.Node>`
  - ```{autodoc2-docstring} eins.parsing.Node
    :summary:
    ```
* - {py:obj}`equations <eins.parsing.equations>`
  - ```{autodoc2-docstring} eins.parsing.equations
    :summary:
    ```
````

### API

````{py:data} lparen
:canonical: eins.parsing.lparen
:value: >
   '('

```{autodoc2-docstring} eins.parsing.lparen
```

````

````{py:data} rparen
:canonical: eins.parsing.rparen
:value: >
   ')'

```{autodoc2-docstring} eins.parsing.rparen
```

````

````{py:data} parens
:canonical: eins.parsing.parens
:value: >
   'compile(...)'

```{autodoc2-docstring} eins.parsing.parens
```

````

````{py:data} index
:canonical: eins.parsing.index
:value: >
   'compile(...)'

```{autodoc2-docstring} eins.parsing.index
```

````

````{py:data} pows_paren
:canonical: eins.parsing.pows_paren
:value: >
   'compile(...)'

```{autodoc2-docstring} eins.parsing.pows_paren
```

````

````{py:data} pows_atomic
:canonical: eins.parsing.pows_atomic
:value: >
   'compile(...)'

```{autodoc2-docstring} eins.parsing.pows_atomic
```

````

````{py:function} unpack_parens(m: re.Match)
:canonical: eins.parsing.unpack_parens

```{autodoc2-docstring} eins.parsing.unpack_parens
```
````

````{py:function} unpack_index(m: re.Match)
:canonical: eins.parsing.unpack_index

```{autodoc2-docstring} eins.parsing.unpack_index
```
````

````{py:function} unpack_pow(m: re.Match)
:canonical: eins.parsing.unpack_pow

```{autodoc2-docstring} eins.parsing.unpack_pow
```
````

````{py:data} MAX_NESTED_PARENS
:canonical: eins.parsing.MAX_NESTED_PARENS
:value: >
   100

```{autodoc2-docstring} eins.parsing.MAX_NESTED_PARENS
```

````

````{py:function} unpack_shorthands(expr: str)
:canonical: eins.parsing.unpack_shorthands

```{autodoc2-docstring} eins.parsing.unpack_shorthands
```
````

````{py:data} spaces
:canonical: eins.parsing.spaces
:value: >
   'Suppress(...)'

```{autodoc2-docstring} eins.parsing.spaces
```

````

````{py:data} eq_op
:canonical: eins.parsing.eq_op
:value: >
   'one_of(...)'

```{autodoc2-docstring} eins.parsing.eq_op
```

````

````{py:data} comma_op
:canonical: eins.parsing.comma_op
:value: >
   None

```{autodoc2-docstring} eins.parsing.comma_op
```

````

````{py:data} seq_op
:canonical: eins.parsing.seq_op
:value: >
   'Literal(...)'

```{autodoc2-docstring} eins.parsing.seq_op
```

````

````{py:data} add_op
:canonical: eins.parsing.add_op
:value: >
   'one_of(...)'

```{autodoc2-docstring} eins.parsing.add_op
```

````

````{py:data} mul_op
:canonical: eins.parsing.mul_op
:value: >
   'one_of(...)'

```{autodoc2-docstring} eins.parsing.mul_op
```

````

````{py:data} pow_op
:canonical: eins.parsing.pow_op
:value: >
   'one_of(...)'

```{autodoc2-docstring} eins.parsing.pow_op
```

````

````{py:data} arrow
:canonical: eins.parsing.arrow
:value: >
   None

```{autodoc2-docstring} eins.parsing.arrow
```

````

````{py:data} index_op
:canonical: eins.parsing.index_op
:value: >
   None

```{autodoc2-docstring} eins.parsing.index_op
```

````

````{py:data} symbol
:canonical: eins.parsing.symbol
:value: >
   'Word(...)'

```{autodoc2-docstring} eins.parsing.symbol
```

````

````{py:data} literal
:canonical: eins.parsing.literal
:value: >
   'Word(...)'

```{autodoc2-docstring} eins.parsing.literal
```

````

````{py:data} operand
:canonical: eins.parsing.operand
:value: >
   None

```{autodoc2-docstring} eins.parsing.operand
```

````

````{py:data} expr
:canonical: eins.parsing.expr
:value: >
   'infix_notation(...)'

```{autodoc2-docstring} eins.parsing.expr
```

````

`````{py:class} Constant
:canonical: eins.parsing.Constant

```{autodoc2-docstring} eins.parsing.Constant
```

````{py:attribute} value
:canonical: eins.parsing.Constant.value
:type: int
:value: >
   None

```{autodoc2-docstring} eins.parsing.Constant.value
```

````

````{py:method} __repr__()
:canonical: eins.parsing.Constant.__repr__

````

`````

`````{py:class} Symbol
:canonical: eins.parsing.Symbol

```{autodoc2-docstring} eins.parsing.Symbol
```

````{py:attribute} value
:canonical: eins.parsing.Symbol.value
:type: str
:value: >
   None

```{autodoc2-docstring} eins.parsing.Symbol.value
```

````

````{py:method} __repr__()
:canonical: eins.parsing.Symbol.__repr__

````

`````

````{py:data} Node
:canonical: eins.parsing.Node
:value: >
   None

```{autodoc2-docstring} eins.parsing.Node
```

````

`````{py:class} Expr
:canonical: eins.parsing.Expr

```{autodoc2-docstring} eins.parsing.Expr
```

````{py:attribute} op
:canonical: eins.parsing.Expr.op
:type: str
:value: >
   None

```{autodoc2-docstring} eins.parsing.Expr.op
```

````

````{py:attribute} children
:canonical: eins.parsing.Expr.children
:type: list[eins.parsing.Node]
:value: >
   None

```{autodoc2-docstring} eins.parsing.Expr.children
```

````

````{py:method} tree_map(op: typing.Callable) -> eins.parsing.Expr
:canonical: eins.parsing.Expr.tree_map

```{autodoc2-docstring} eins.parsing.Expr.tree_map
```

````

````{py:method} replace_with(new: eins.parsing.Expr)
:canonical: eins.parsing.Expr.replace_with

```{autodoc2-docstring} eins.parsing.Expr.replace_with
```

````

````{py:method} __str__()
:canonical: eins.parsing.Expr.__str__

````

`````

````{py:function} make_expr(parsed: list | str) -> eins.parsing.Expr
:canonical: eins.parsing.make_expr

```{autodoc2-docstring} eins.parsing.make_expr
```
````

````{py:data} equations
:canonical: eins.parsing.equations
:value: >
   []

```{autodoc2-docstring} eins.parsing.equations
```

````

````{py:function} flatten(node: eins.parsing.Node) -> eins.parsing.Node
:canonical: eins.parsing.flatten

```{autodoc2-docstring} eins.parsing.flatten
```
````
