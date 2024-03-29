# {py:mod}`eins.einsop`

```{py:module} eins.einsop
```

```{autodoc2-docstring} eins.einsop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EinsOp <eins.einsop.EinsOp>`
  - ```{autodoc2-docstring} eins.einsop.EinsOp
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_reduce_arg <eins.einsop.parse_reduce_arg>`
  - ```{autodoc2-docstring} eins.einsop.parse_reduce_arg
    :summary:
    ```
* - {py:obj}`parse_combine_arg <eins.einsop.parse_combine_arg>`
  - ```{autodoc2-docstring} eins.einsop.parse_combine_arg
    :summary:
    ```
* - {py:obj}`einsop <eins.einsop.einsop>`
  - ```{autodoc2-docstring} eins.einsop.einsop
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ElementwiseKind <eins.einsop.ElementwiseKind>`
  - ```{autodoc2-docstring} eins.einsop.ElementwiseKind
    :summary:
    ```
* - {py:obj}`ReductionKind <eins.einsop.ReductionKind>`
  - ```{autodoc2-docstring} eins.einsop.ReductionKind
    :summary:
    ```
* - {py:obj}`CombinationKind <eins.einsop.CombinationKind>`
  - ```{autodoc2-docstring} eins.einsop.CombinationKind
    :summary:
    ```
* - {py:obj}`GeneralReductionKind <eins.einsop.GeneralReductionKind>`
  - ```{autodoc2-docstring} eins.einsop.GeneralReductionKind
    :summary:
    ```
* - {py:obj}`ReduceArg <eins.einsop.ReduceArg>`
  - ```{autodoc2-docstring} eins.einsop.ReduceArg
    :summary:
    ```
* - {py:obj}`CombineArg <eins.einsop.CombineArg>`
  - ```{autodoc2-docstring} eins.einsop.CombineArg
    :summary:
    ```
````

### API

````{py:data} ElementwiseKind
:canonical: eins.einsop.ElementwiseKind
:value: >
   None

```{autodoc2-docstring} eins.einsop.ElementwiseKind
```

````

````{py:data} ReductionKind
:canonical: eins.einsop.ReductionKind
:value: >
   None

```{autodoc2-docstring} eins.einsop.ReductionKind
```

````

````{py:data} CombinationKind
:canonical: eins.einsop.CombinationKind
:value: >
   None

```{autodoc2-docstring} eins.einsop.CombinationKind
```

````

````{py:data} GeneralReductionKind
:canonical: eins.einsop.GeneralReductionKind
:value: >
   None

```{autodoc2-docstring} eins.einsop.GeneralReductionKind
```

````

````{py:data} ReduceArg
:canonical: eins.einsop.ReduceArg
:value: >
   None

```{autodoc2-docstring} eins.einsop.ReduceArg
```

````

````{py:data} CombineArg
:canonical: eins.einsop.CombineArg
:value: >
   None

```{autodoc2-docstring} eins.einsop.CombineArg
```

````

````{py:function} parse_reduce_arg(reduce: eins.einsop.GeneralReductionKind) -> eins.reduction.Reduction
:canonical: eins.einsop.parse_reduce_arg

```{autodoc2-docstring} eins.einsop.parse_reduce_arg
```
````

````{py:function} parse_combine_arg(combine: eins.einsop.CombineArg) -> eins.combination.Combination
:canonical: eins.einsop.parse_combine_arg

```{autodoc2-docstring} eins.einsop.parse_combine_arg
```
````

`````{py:class} EinsOp(op: str, /, *, reduce: eins.einsop.ReduceArg = 'sum', combine: eins.einsop.CombineArg = 'multiply')
:canonical: eins.einsop.EinsOp

```{autodoc2-docstring} eins.einsop.EinsOp
```

```{rubric} Initialization
```

```{autodoc2-docstring} eins.einsop.EinsOp.__init__
```

````{py:method} __repr__() -> str
:canonical: eins.einsop.EinsOp.__repr__

````

````{py:method} __call__(*tensors: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.einsop.EinsOp.__call__

```{autodoc2-docstring} eins.einsop.EinsOp.__call__
```

````

`````

````{py:function} einsop(op_str: str, *tensors: eins.common_types.Array, reduce: eins.einsop.ReduceArg = 'sum', combine: eins.einsop.CombineArg = 'multiply') -> eins.common_types.Array
:canonical: eins.einsop.einsop

```{autodoc2-docstring} eins.einsop.einsop
```
````
