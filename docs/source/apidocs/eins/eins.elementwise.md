# {py:mod}`eins.elementwise`

```{py:module} eins.elementwise
```

```{autodoc2-docstring} eins.elementwise
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ElementwiseOp <eins.elementwise.ElementwiseOp>`
  - ```{autodoc2-docstring} eins.elementwise.ElementwiseOp
    :summary:
    ```
* - {py:obj}`ArrayElementwiseOp <eins.elementwise.ArrayElementwiseOp>`
  - ```{autodoc2-docstring} eins.elementwise.ArrayElementwiseOp
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_elementwise <eins.elementwise.parse_elementwise>`
  - ```{autodoc2-docstring} eins.elementwise.parse_elementwise
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ARRAY_ELEMWISE_OPS <eins.elementwise.ARRAY_ELEMWISE_OPS>`
  - ```{autodoc2-docstring} eins.elementwise.ARRAY_ELEMWISE_OPS
    :summary:
    ```
* - {py:obj}`ElementwiseLiteral <eins.elementwise.ElementwiseLiteral>`
  - ```{autodoc2-docstring} eins.elementwise.ElementwiseLiteral
    :summary:
    ```
````

### API

`````{py:class} ElementwiseOp
:canonical: eins.elementwise.ElementwiseOp

```{autodoc2-docstring} eins.elementwise.ElementwiseOp
```

````{py:method} parse(_name: str)
:canonical: eins.elementwise.ElementwiseOp.parse
:classmethod:

```{autodoc2-docstring} eins.elementwise.ElementwiseOp.parse
```

````

````{py:method} __call__(arr: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.elementwise.ElementwiseOp.__call__
:abstractmethod:

```{autodoc2-docstring} eins.elementwise.ElementwiseOp.__call__
```

````

`````

````{py:data} ARRAY_ELEMWISE_OPS
:canonical: eins.elementwise.ARRAY_ELEMWISE_OPS
:value: >
   ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'bitwise_invert', 'ceil', 'conj', 'cos', ...

```{autodoc2-docstring} eins.elementwise.ARRAY_ELEMWISE_OPS
```

````

````{py:data} ElementwiseLiteral
:canonical: eins.elementwise.ElementwiseLiteral
:value: >
   None

```{autodoc2-docstring} eins.elementwise.ElementwiseLiteral
```

````

`````{py:class} ArrayElementwiseOp
:canonical: eins.elementwise.ArrayElementwiseOp

Bases: {py:obj}`eins.elementwise.ElementwiseOp`

```{autodoc2-docstring} eins.elementwise.ArrayElementwiseOp
```

````{py:attribute} func_name
:canonical: eins.elementwise.ArrayElementwiseOp.func_name
:type: str
:value: >
   None

```{autodoc2-docstring} eins.elementwise.ArrayElementwiseOp.func_name
```

````

````{py:method} __str__()
:canonical: eins.elementwise.ArrayElementwiseOp.__str__

````

````{py:method} parse(name: str)
:canonical: eins.elementwise.ArrayElementwiseOp.parse
:classmethod:

````

````{py:method} __call__(arr: eins.common_types.Array) -> eins.common_types.Array
:canonical: eins.elementwise.ArrayElementwiseOp.__call__

````

`````

````{py:function} parse_elementwise(name: str) -> eins.elementwise.ElementwiseOp
:canonical: eins.elementwise.parse_elementwise

```{autodoc2-docstring} eins.elementwise.parse_elementwise
```
````
