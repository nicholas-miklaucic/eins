# {py:mod}`eins.symbolic`

```{py:module} eins.symbolic
```

```{autodoc2-docstring} eins.symbolic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tensor <eins.symbolic.Tensor>`
  - ```{autodoc2-docstring} eins.symbolic.Tensor
    :summary:
    ```
* - {py:obj}`ShapeOp <eins.symbolic.ShapeOp>`
  - ```{autodoc2-docstring} eins.symbolic.ShapeOp
    :summary:
    ```
* - {py:obj}`Reshape <eins.symbolic.Reshape>`
  - ```{autodoc2-docstring} eins.symbolic.Reshape
    :summary:
    ```
* - {py:obj}`Transpose <eins.symbolic.Transpose>`
  - ```{autodoc2-docstring} eins.symbolic.Transpose
    :summary:
    ```
* - {py:obj}`Split <eins.symbolic.Split>`
  - ```{autodoc2-docstring} eins.symbolic.Split
    :summary:
    ```
* - {py:obj}`Concat <eins.symbolic.Concat>`
  - ```{autodoc2-docstring} eins.symbolic.Concat
    :summary:
    ```
* - {py:obj}`OneHot <eins.symbolic.OneHot>`
  - ```{autodoc2-docstring} eins.symbolic.OneHot
    :summary:
    ```
* - {py:obj}`ExpandTo <eins.symbolic.ExpandTo>`
  - ```{autodoc2-docstring} eins.symbolic.ExpandTo
    :summary:
    ```
* - {py:obj}`Combine <eins.symbolic.Combine>`
  - ```{autodoc2-docstring} eins.symbolic.Combine
    :summary:
    ```
* - {py:obj}`Reduce <eins.symbolic.Reduce>`
  - ```{autodoc2-docstring} eins.symbolic.Reduce
    :summary:
    ```
* - {py:obj}`Program <eins.symbolic.Program>`
  - ```{autodoc2-docstring} eins.symbolic.Program
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`expanded_shape <eins.symbolic.expanded_shape>`
  - ```{autodoc2-docstring} eins.symbolic.expanded_shape
    :summary:
    ```
* - {py:obj}`normalize_step <eins.symbolic.normalize_step>`
  - ```{autodoc2-docstring} eins.symbolic.normalize_step
    :summary:
    ```
* - {py:obj}`normalize_until_done <eins.symbolic.normalize_until_done>`
  - ```{autodoc2-docstring} eins.symbolic.normalize_until_done
    :summary:
    ```
* - {py:obj}`reverse_graph <eins.symbolic.reverse_graph>`
  - ```{autodoc2-docstring} eins.symbolic.reverse_graph
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEFAULT_COMBINE <eins.symbolic.DEFAULT_COMBINE>`
  - ```{autodoc2-docstring} eins.symbolic.DEFAULT_COMBINE
    :summary:
    ```
* - {py:obj}`DEFAULT_REDUCE <eins.symbolic.DEFAULT_REDUCE>`
  - ```{autodoc2-docstring} eins.symbolic.DEFAULT_REDUCE
    :summary:
    ```
````

### API

`````{py:class} Tensor(expr: eins.parsing.Node)
:canonical: eins.symbolic.Tensor

```{autodoc2-docstring} eins.symbolic.Tensor
```

```{rubric} Initialization
```

```{autodoc2-docstring} eins.symbolic.Tensor.__init__
```

````{py:method} deepcopy() -> eins.symbolic.Tensor
:canonical: eins.symbolic.Tensor.deepcopy

```{autodoc2-docstring} eins.symbolic.Tensor.deepcopy
```

````

````{py:method} axes_list() -> typing.Sequence[str]
:canonical: eins.symbolic.Tensor.axes_list

```{autodoc2-docstring} eins.symbolic.Tensor.axes_list
```

````

````{py:method} add_child_op(children: Sequence[Tensor], op: eins.symbolic.ShapeOp)
:canonical: eins.symbolic.Tensor.add_child_op

```{autodoc2-docstring} eins.symbolic.Tensor.add_child_op
```

````

````{py:method} is_same_shape(other: eins.symbolic.Tensor) -> bool
:canonical: eins.symbolic.Tensor.is_same_shape

```{autodoc2-docstring} eins.symbolic.Tensor.is_same_shape
```

````

````{py:method} __repr__()
:canonical: eins.symbolic.Tensor.__repr__

````

`````

`````{py:class} ShapeOp
:canonical: eins.symbolic.ShapeOp

```{autodoc2-docstring} eins.symbolic.ShapeOp
```

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.ShapeOp.apply
:abstractmethod:

```{autodoc2-docstring} eins.symbolic.ShapeOp.apply
```

````

````{py:method} __call__(tensors: typing.Union[eins.symbolic.Tensor, typing.Sequence[eins.symbolic.Tensor]]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.ShapeOp.__call__

```{autodoc2-docstring} eins.symbolic.ShapeOp.__call__
```

````

````{py:method} is_identity_for(_tensors: typing.Sequence[eins.symbolic.Tensor]) -> bool
:canonical: eins.symbolic.ShapeOp.is_identity_for

```{autodoc2-docstring} eins.symbolic.ShapeOp.is_identity_for
```

````

`````

`````{py:class} Reshape
:canonical: eins.symbolic.Reshape

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.Reshape
```

````{py:attribute} new_shape
:canonical: eins.symbolic.Reshape.new_shape
:type: tuple[eins.parsing.Node]
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Reshape.new_shape
```

````

````{py:method} apply(_tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.Reshape.apply

```{autodoc2-docstring} eins.symbolic.Reshape.apply
```

````

`````

`````{py:class} Transpose
:canonical: eins.symbolic.Transpose

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.Transpose
```

````{py:attribute} perm
:canonical: eins.symbolic.Transpose.perm
:type: typing.Sequence[int]
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Transpose.perm
```

````

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.Transpose.apply

```{autodoc2-docstring} eins.symbolic.Transpose.apply
```

````

````{py:method} is_identity_for(_tensors: typing.Sequence[eins.symbolic.Tensor]) -> bool
:canonical: eins.symbolic.Transpose.is_identity_for

```{autodoc2-docstring} eins.symbolic.Transpose.is_identity_for
```

````

`````

`````{py:class} Split
:canonical: eins.symbolic.Split

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.Split
```

````{py:attribute} axis_num
:canonical: eins.symbolic.Split.axis_num
:type: int
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Split.axis_num
```

````

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.Split.apply

```{autodoc2-docstring} eins.symbolic.Split.apply
```

````

`````

`````{py:class} Concat
:canonical: eins.symbolic.Concat

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.Concat
```

````{py:attribute} axis_num
:canonical: eins.symbolic.Concat.axis_num
:type: int
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Concat.axis_num
```

````

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.Concat.apply

```{autodoc2-docstring} eins.symbolic.Concat.apply
```

````

`````

`````{py:class} OneHot
:canonical: eins.symbolic.OneHot

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.OneHot
```

````{py:attribute} idx_axis
:canonical: eins.symbolic.OneHot.idx_axis
:type: eins.parsing.Node
:value: >
   None

```{autodoc2-docstring} eins.symbolic.OneHot.idx_axis
```

````

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.OneHot.apply

```{autodoc2-docstring} eins.symbolic.OneHot.apply
```

````

`````

`````{py:class} ExpandTo
:canonical: eins.symbolic.ExpandTo

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.ExpandTo
```

````{py:attribute} new_shape
:canonical: eins.symbolic.ExpandTo.new_shape
:type: tuple[eins.parsing.Node]
:value: >
   None

```{autodoc2-docstring} eins.symbolic.ExpandTo.new_shape
```

````

````{py:method} apply(_tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.ExpandTo.apply

```{autodoc2-docstring} eins.symbolic.ExpandTo.apply
```

````

`````

`````{py:class} Combine
:canonical: eins.symbolic.Combine

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.Combine
```

````{py:attribute} method
:canonical: eins.symbolic.Combine.method
:type: eins.reduction.Reduction
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Combine.method
```

````

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.Combine.apply

```{autodoc2-docstring} eins.symbolic.Combine.apply
```

````

````{py:method} is_identity_for(tensors: typing.Sequence[eins.symbolic.Tensor]) -> bool
:canonical: eins.symbolic.Combine.is_identity_for

```{autodoc2-docstring} eins.symbolic.Combine.is_identity_for
```

````

`````

`````{py:class} Reduce
:canonical: eins.symbolic.Reduce

Bases: {py:obj}`eins.symbolic.ShapeOp`

```{autodoc2-docstring} eins.symbolic.Reduce
```

````{py:attribute} method
:canonical: eins.symbolic.Reduce.method
:type: eins.reduction.Reduction
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Reduce.method
```

````

````{py:attribute} axis
:canonical: eins.symbolic.Reduce.axis
:type: eins.parsing.Node
:value: >
   None

```{autodoc2-docstring} eins.symbolic.Reduce.axis
```

````

````{py:method} apply(tensors: typing.Sequence[eins.symbolic.Tensor]) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.Reduce.apply

```{autodoc2-docstring} eins.symbolic.Reduce.apply
```

````

`````

````{py:function} expanded_shape(node: eins.parsing.Node) -> typing.Sequence[eins.parsing.Node]
:canonical: eins.symbolic.expanded_shape

```{autodoc2-docstring} eins.symbolic.expanded_shape
```
````

````{py:function} normalize_step(tensor: eins.symbolic.Tensor) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.normalize_step

```{autodoc2-docstring} eins.symbolic.normalize_step
```
````

````{py:function} normalize_until_done(tensor: eins.symbolic.Tensor) -> typing.Sequence[eins.symbolic.Tensor]
:canonical: eins.symbolic.normalize_until_done

```{autodoc2-docstring} eins.symbolic.normalize_until_done
```
````

````{py:function} reverse_graph(root: eins.symbolic.Tensor)
:canonical: eins.symbolic.reverse_graph

```{autodoc2-docstring} eins.symbolic.reverse_graph
```
````

````{py:data} DEFAULT_COMBINE
:canonical: eins.symbolic.DEFAULT_COMBINE
:value: >
   'ArrayCombination(...)'

```{autodoc2-docstring} eins.symbolic.DEFAULT_COMBINE
```

````

````{py:data} DEFAULT_REDUCE
:canonical: eins.symbolic.DEFAULT_REDUCE
:value: >
   'ArrayReduction(...)'

```{autodoc2-docstring} eins.symbolic.DEFAULT_REDUCE
```

````

`````{py:class} Program(expr: eins.parsing.Expr, constr: eins.constraint.Constraints, combine: eins.combination.Combination = DEFAULT_COMBINE, reduce: eins.reduction.Reduction | typing.Mapping[str, eins.reduction.Reduction] = DEFAULT_REDUCE)
:canonical: eins.symbolic.Program

```{autodoc2-docstring} eins.symbolic.Program
```

```{rubric} Initialization
```

```{autodoc2-docstring} eins.symbolic.Program.__init__
```

````{py:method} apply_op(op: eins.symbolic.ShapeOp, tensors: typing.Union[eins.symbolic.Tensor, typing.Sequence[eins.symbolic.Tensor]])
:canonical: eins.symbolic.Program.apply_op
:staticmethod:

```{autodoc2-docstring} eins.symbolic.Program.apply_op
```

````

````{py:method} parse(op: str)
:canonical: eins.symbolic.Program.parse
:classmethod:

```{autodoc2-docstring} eins.symbolic.Program.parse
```

````

````{py:method} connect(start: typing.Sequence[eins.symbolic.Tensor], goal: eins.symbolic.Tensor)
:canonical: eins.symbolic.Program.connect

```{autodoc2-docstring} eins.symbolic.Program.connect
```

````

````{py:method} __repr__()
:canonical: eins.symbolic.Program.__repr__

````

`````
