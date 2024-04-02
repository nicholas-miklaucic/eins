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

`eins` is still in heavy development. Here's a sense of where we're headed.

### Short-Term (days)

- [x] Better error reporting
- [ ] `...` for batching over dynamic numbers of batch axes
- [ ] Specifying intermediate results to control the order of reduction
- [ ] Support `-` and `/` as natural pairs to `+` and `*`
- [ ] Implementing `repeat`
- [x] Adding support for reductions that aren't in Array API (e.g., p-norm)
- [x] Adding support for "transformations" that are elementwise but use an axis (standardize)
- [x] Flip order of inputs (logsumexp is log + sum + exp)
- [ ] Unit array to indicate zero-dimensional tensors
- [ ] Updating indexing syntax to match `eindex`
- [/] Much more thorough documentation and tests for existing functionality
- [ ] Better visualization of the program graph
- [ ] `@local` syntax

### Near-Term (weeks)
- [ ] Easy performance gains
- [ ] Much better error reporting
- [ ] Completing full support for tensor indexing
- [ ] Warnings or errors for ambiguous expressions
- [ ] Ways of visualizing and inspecting the computation graph

### Long-Term (months)
- [ ] Layers for popular ML frameworks?
- [ ] Automatically optimizing the execution of a specific EinsOp for a specific computer and input size
- [ ] Static typing support that shows the array shapes
- [ ] Tabulating the model FLOPs/memory usage as a function of named axes
- [ ] Functionality akin to `pack` and `unpack`?

## Acknowledgements

The excellent [`einops`](https://github.com/arogozhnikov/einops) library inspired this project and its syntax. After working on my own extension to handle indexing, I realized that [`eindex`](https://github.com/arogozhnikov/eindex) already had a more coherent vision for what indexing can look like, and so much of that syntax in this library borrows from that one.

## Contributing

Any contributions to `eins` are welcomed and appreciated! If you're interested in making serious changes or extensions to the syntax of operations, consider reaching out first to make sure we're on the same page. For any code changes, do make sure you're using the project Ruff settings.

## License

`eins` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


To do:

- Passing in constants: this needs to be desugared to named values, I think?
- When we duplicate n to n-2, also copy n's reduction
- Test with JIT and PyTorch
- Get a basic perf comparison
- Docs website, maybe rewrite a few big models
- Example with non-Latin characters
- `pyproject.toml` fun stuff
- Get ops from `nn` libraries or logit or something
- Intermediate chaining syntax
- Constants file with op characters and similar