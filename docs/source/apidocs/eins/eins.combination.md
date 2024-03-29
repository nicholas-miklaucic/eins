# {py:mod}`eins.combination`

```{py:module} eins.combination
```

```{autodoc2-docstring} eins.combination
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Combination <eins.combination.Combination>`
  - ```{autodoc2-docstring} eins.combination.Combination
    :summary:
    ```
* - {py:obj}`ArrayCombination <eins.combination.ArrayCombination>`
  - ```{autodoc2-docstring} eins.combination.ArrayCombination
    :summary:
    ```
* - {py:obj}`UserCombination <eins.combination.UserCombination>`
  - ```{autodoc2-docstring} eins.combination.UserCombination
    :summary:
    ```
* - {py:obj}`CompositeCombination <eins.combination.CompositeCombination>`
  - ```{autodoc2-docstring} eins.combination.CompositeCombination
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_combination <eins.combination.parse_combination>`
  - ```{autodoc2-docstring} eins.combination.parse_combination
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ARRAY_COMBINE_OPS <eins.combination.ARRAY_COMBINE_OPS>`
  - ```{autodoc2-docstring} eins.combination.ARRAY_COMBINE_OPS
    :summary:
    ```
* - {py:obj}`CombineLiteral <eins.combination.CombineLiteral>`
  - ```{autodoc2-docstring} eins.combination.CombineLiteral
    :summary:
    ```
````

### API

`````{py:class} Combination
:canonical: eins.combination.Combination

```{autodoc2-docstring} eins.combination.Combination
```

````{py:method} parse(name: str)
:canonical: eins.combination.Combination.parse
:classmethod:

```{autodoc2-docstring} eins.combination.Combination.parse
```

````

````{py:method} __call__(*arrs: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.combination.Combination.__call__
:abstractmethod:

```{autodoc2-docstring} eins.combination.Combination.__call__
```

````

`````

````{py:data} ARRAY_COMBINE_OPS
:canonical: eins.combination.ARRAY_COMBINE_OPS
:value: >
   ['add', 'hypot', 'logaddexp', 'maximum', 'minimum', 'multiply', 'bitwise_xor', 'bitwise_and', 'bitwi...

```{autodoc2-docstring} eins.combination.ARRAY_COMBINE_OPS
```

````

````{py:data} CombineLiteral
:canonical: eins.combination.CombineLiteral
:value: >
   None

```{autodoc2-docstring} eins.combination.CombineLiteral
```

````

`````{py:class} ArrayCombination
:canonical: eins.combination.ArrayCombination

Bases: {py:obj}`eins.combination.Combination`

```{autodoc2-docstring} eins.combination.ArrayCombination
```

````{py:attribute} func_name
:canonical: eins.combination.ArrayCombination.func_name
:type: str
:value: >
   None

```{autodoc2-docstring} eins.combination.ArrayCombination.func_name
```

````

````{py:method} __str__()
:canonical: eins.combination.ArrayCombination.__str__

````

````{py:method} parse(name: str)
:canonical: eins.combination.ArrayCombination.parse
:classmethod:

````

````{py:method} __call__(*arrs: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.combination.ArrayCombination.__call__

````

`````

`````{py:class} UserCombination
:canonical: eins.combination.UserCombination

Bases: {py:obj}`eins.combination.Combination`

```{autodoc2-docstring} eins.combination.UserCombination
```

````{py:attribute} func
:canonical: eins.combination.UserCombination.func
:type: typing.Callable
:value: >
   None

```{autodoc2-docstring} eins.combination.UserCombination.func
```

````

````{py:method} __call__(*arrs: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.combination.UserCombination.__call__

````

`````

`````{py:class} CompositeCombination
:canonical: eins.combination.CompositeCombination

Bases: {py:obj}`eins.combination.Combination`

```{autodoc2-docstring} eins.combination.CompositeCombination
```

````{py:attribute} ops
:canonical: eins.combination.CompositeCombination.ops
:type: typing.Sequence[typing.Union[eins.elementwise.ElementwiseOp, eins.combination.Combination]]
:value: >
   None

```{autodoc2-docstring} eins.combination.CompositeCombination.ops
```

````

````{py:method} __call__(*arrs: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.combination.CompositeCombination.__call__

````

`````

````{py:function} parse_combination(name: str) -> eins.combination.Combination
:canonical: eins.combination.parse_combination

```{autodoc2-docstring} eins.combination.parse_combination
```
````
