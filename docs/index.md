# Eins Documentation
## Introduction: one tensor operation is all you need

Eins is a library that generalizes `einops`-like syntax to almost any operation you would want to do with tensors.

As an example, here's [the patch embedding from a vision transformer](https://nn.labml.ai/transformers/vit/index.html#PatchEmbeddings):
```python
linear = EinsOp('b (n_p patch) (n_p patch) c, (patch patch c) emb -> b (n_p n_p) emb')
add_bias = EinsOp('b (n_p n_p) emb, emb -> b (n_p n_p) emb', combine='add')

kernel = randn(5 * 5 * 3, 12)
bias = randn(12)
images = randn(4, 55, 55, 3)
patches = add_bias(linear(images, kernel), bias)
print(patches.shape)  # (4, 121, 12)
```

## Installation

Eins works with anything that implements the Array API: that includes `numpy`, `jax`, and `torch`. Install one of those libraries, and then simply run

```console
pip install eins
```

## Next Steps

To get started, check out the [tutorial](tutorial.md), which walks you through the most important parts of Eins and shows how you can use Eins to manipulate data in a readable, reliable way.

The [cookbook](cookbook.md) shows how many common operations are implemented in Eins in short code snippets you can easily copy and paste. It's best to read the [tutorial](tutorial.md) first, so you understand how to adapt that code for your own use cases.

[Advanced Eins](in-depth.md) explains advanced features and functionality—not always essential, but if you want Eins to be a one-stop shop for all of your tensor manipulation needs you should give it a read.

Finally, if you want to see exactly what's happening under the hood, you can consult the [API documentation](api/eins/). Ideally, there would be no reason to read the API docs unless you're interested in the internals, but it's there if you need it.

If you run into problems, think something should be done differently, or want to suggest improvements to Eins, feel free to [file an issue.](https://github.com/nicholas-miklaucic/eins/issues/new/choose)