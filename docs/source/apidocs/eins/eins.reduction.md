# {py:mod}`eins.reduction`

```{py:module} eins.reduction
```

```{autodoc2-docstring} eins.reduction
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Reduction <eins.reduction.Reduction>`
  - ```{autodoc2-docstring} eins.reduction.Reduction
    :summary:
    ```
* - {py:obj}`ArrayReduction <eins.reduction.ArrayReduction>`
  - ```{autodoc2-docstring} eins.reduction.ArrayReduction
    :summary:
    ```
* - {py:obj}`Fold <eins.reduction.Fold>`
  - ```{autodoc2-docstring} eins.reduction.Fold
    :summary:
    ```
* - {py:obj}`CompositeReduction <eins.reduction.CompositeReduction>`
  - ```{autodoc2-docstring} eins.reduction.CompositeReduction
    :summary:
    ```
* - {py:obj}`UserReduction <eins.reduction.UserReduction>`
  - ```{autodoc2-docstring} eins.reduction.UserReduction
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_reduction <eins.reduction.parse_reduction>`
  - ```{autodoc2-docstring} eins.reduction.parse_reduction
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ARRAY_REDUCE_OPS <eins.reduction.ARRAY_REDUCE_OPS>`
  - ```{autodoc2-docstring} eins.reduction.ARRAY_REDUCE_OPS
    :summary:
    ```
* - {py:obj}`ReductionLiteral <eins.reduction.ReductionLiteral>`
  - ```{autodoc2-docstring} eins.reduction.ReductionLiteral
    :summary:
    ```
* - {py:obj}`REDUNDANT_FOLDS <eins.reduction.REDUNDANT_FOLDS>`
  - ```{autodoc2-docstring} eins.reduction.REDUNDANT_FOLDS
    :summary:
    ```
````

### API

`````{py:class} Reduction
:canonical: eins.reduction.Reduction

```{autodoc2-docstring} eins.reduction.Reduction
```

````{py:method} parse(_name: str)
:canonical: eins.reduction.Reduction.parse
:classmethod:

```{autodoc2-docstring} eins.reduction.Reduction.parse
```

````

````{py:method} __call__(arr: eins.common_types.Array, axis: int = 0) -> eins.common_types.Array
:canonical: eins.reduction.Reduction.__call__
:abstractmethod:

```{autodoc2-docstring} eins.reduction.Reduction.__call__
```

````

`````

````{py:data} ARRAY_REDUCE_OPS
:canonical: eins.reduction.ARRAY_REDUCE_OPS
:value: >
   ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

```{autodoc2-docstring} eins.reduction.ARRAY_REDUCE_OPS
```

````

````{py:data} ReductionLiteral
:canonical: eins.reduction.ReductionLiteral
:value: >
   None

```{autodoc2-docstring} eins.reduction.ReductionLiteral
```

````

`````{py:class} ArrayReduction
:canonical: eins.reduction.ArrayReduction

Bases: {py:obj}`eins.reduction.Reduction`

```{autodoc2-docstring} eins.reduction.ArrayReduction
```

````{py:attribute} func_name
:canonical: eins.reduction.ArrayReduction.func_name
:type: str
:value: >
   None

```{autodoc2-docstring} eins.reduction.ArrayReduction.func_name
```

````

````{py:method} __str__()
:canonical: eins.reduction.ArrayReduction.__str__

````

````{py:method} parse(name: str)
:canonical: eins.reduction.ArrayReduction.parse
:classmethod:

````

````{py:method} __call__(arr: eins.common_types.Array, axis: int = 0)
:canonical: eins.reduction.ArrayReduction.__call__

````

`````

````{py:data} REDUNDANT_FOLDS
:canonical: eins.reduction.REDUNDANT_FOLDS
:value: >
   None

```{autodoc2-docstring} eins.reduction.REDUNDANT_FOLDS
```

````

`````{py:class} Fold
:canonical: eins.reduction.Fold

Bases: {py:obj}`eins.reduction.Reduction`

```{autodoc2-docstring} eins.reduction.Fold
```

````{py:attribute} combination
:canonical: eins.reduction.Fold.combination
:type: eins.combination.Combination
:value: >
   None

```{autodoc2-docstring} eins.reduction.Fold.combination
```

````

````{py:method} parse(name: str)
:canonical: eins.reduction.Fold.parse
:classmethod:

````

````{py:method} __call__(arr: eins.common_types.Array, axis: int = 0)
:canonical: eins.reduction.Fold.__call__

````

`````

`````{py:class} CompositeReduction
:canonical: eins.reduction.CompositeReduction

Bases: {py:obj}`eins.reduction.Reduction`

```{autodoc2-docstring} eins.reduction.CompositeReduction
```

````{py:attribute} ops
:canonical: eins.reduction.CompositeReduction.ops
:type: typing.Sequence[typing.Union[eins.elementwise.ElementwiseOp, eins.reduction.Reduction]]
:value: >
   None

```{autodoc2-docstring} eins.reduction.CompositeReduction.ops
```

````

````{py:method} parse(_name: str)
:canonical: eins.reduction.CompositeReduction.parse
:classmethod:

````

````{py:method} __call__(arr: eins.common_types.Array, axis: int = 0) -> eins.common_types.Array
:canonical: eins.reduction.CompositeReduction.__call__

````

`````

`````{py:class} UserReduction
:canonical: eins.reduction.UserReduction

Bases: {py:obj}`eins.reduction.Reduction`

```{autodoc2-docstring} eins.reduction.UserReduction
```

````{py:attribute} func
:canonical: eins.reduction.UserReduction.func
:type: typing.Callable
:value: >
   None

```{autodoc2-docstring} eins.reduction.UserReduction.func
```

````

````{py:method} __call__(arr: eins.common_types.Array, axis: int = 0) -> eins.common_types.Array
:canonical: eins.reduction.UserReduction.__call__

````

`````

````{py:function} parse_reduction(name: str) -> eins.reduction.Reduction
:canonical: eins.reduction.parse_reduction

```{autodoc2-docstring} eins.reduction.parse_reduction
```
````
