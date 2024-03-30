# eins
## One tensor operation is all you need

[![PyPI - Version](https://img.shields.io/pypi/v/eins.svg)](https://pypi.org/project/eins)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eins.svg)](https://pypi.org/project/eins)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

What if most of your machine learning model code could be replaced by a single operation? `eins` gives you a powerful language to describe tensor manipulation, making it a one-stop shop for all of your AI needs.

As an example, here's [the patch embedding from a vision transformer](https://nn.labml.ai/transformers/vit/index.html#PatchEmbeddings):
```python
linear = EinsOp('b (n_p patch) (n_p patch) c, (patch patch c) emb -> b (n_p n_p) emb')

kernel = randn(5 * 5 * 3, 12)
images = randn(4, 55, 55, 3)
patches = linear(images, kernel)
print(patches.shape)  # (4, 121, 12)
```

This takes in a batch of square images: `b` images in a batch, `c` channels, and height and width both divisible into `n_p` patches of size `patch`. It combines those images with an embedding layer that takes each of the `patch * patch * c` values in a patch and produces a vector of length `emb`.

If you've used the wonderful [`einops`](https://github.com/arogozhnikov/einops), then think of `eins` as `einops` with a more ambitious goal—being the only operation you should need for your next deep learning project.

## Installation

```console
pip install eins
```

Then, just use `eins` with any library that implements the [Array API](https://data-apis.org/array-api/latest/index.html#), including NumPy, Torch, JAX, CuPy, and Dask.

One of the design goals of `eins` is to be painless to use in almost any Python project. If `pip install eins` is not the only thing you need to do to use Eins in your project, feel free to [file an issue](https://github.com/nicholas-miklaucic/eins/issues/new).

## Roadmap

`eins` is still in heavy development. Here's a sense of where we're headed:

- `...` for batching over dynamic numbers of batch axes
- Specifying intermediate results to control the order of execution
- Implementing `repeat`
- Automatically optimizing the execution of a specific EinsOp for a specific computer and input size
- Completing full support for tensor indexing
- Annotating the output with its shape
- Tabulating the model FLOPs/memory usage as a function of named axes

## Acknowledgements

The excellent [`einops`](https://github.com/arogozhnikov/einops) library inspired this project and its syntax: consider `eins` an attempt at "`einops` on steroids." Einstein did a pretty good job coming up with the summation notation, so big shoutout to him.

## Contributing

Any contributions to `eins` are welcomed and appreciated! If you're interested in making serious changes or extensions to the syntax of operations, consider reaching out first to make sure we're on the same page. For any code changes, do make sure you're using the project Ruff settings.

## License

`eins` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


To do:

- Debug issues with sums in output
- Support for overriding parameter values
- Passing in constants: this needs to be desugared to named values, I think?
- When we duplicate n to n-2, also copy n's reduction
- Start building a test library
- Go through asserts and build out error reporting
- Test with JIT and PyTorch
- Get a basic perf comparison
- Docs website, maybe rewrite a few big models
- Example with non-Latin characters
- Unit array
- `pyproject.toml` fun stuff
- Put constraints in a separate dict, to free up kwargs for further changes

v0.1 new functionality:
- Parsing elementwise/reduction/combination
- Some kind of namespace for those ops or typed creation
- Repeat: one-to-one where inputs are larger than outputs
- Get ops from `nn` libraries or logit or something
- Intermediate chaining syntax
- Static typechecking

Later functionality:
- Give useful error message when order of reductions matters and is not specified: reduce -> reduce with different axes and incompatible reductions
- We want to optimize reductions going forward, so some kind of error when not commutative with combination. `a b, b c, c d -> a d` relies on distributivity to be unambiguous
- `-` and `/` operations: maybe these are just simple rewrites?
- More general indexing
- `...` variadic axes
- Packaging more generally
- Make compatible with Python 3.8
- Pack and unpack replacements: this should probably be unified as "solve for the batch axes" and then substituting
- `@local` syntax a la pandas query?
- Strategies/autotuning
- Graph visualization
- Shape type annotations?
- Flop annotations