# eins

## One tensor operation is all you need

[![PyPI - Version](https://img.shields.io/pypi/v/eins.svg)](https://pypi.org/project/eins)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eins.svg)](https://pypi.org/project/eins)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

What if most of your machine learning model code could be replaced by a single operation? Eins gives
you a powerful language to describe array manipulation, making it a one-stop shop for all of your AI
needs.

For a sample of what Eins does, let's do [the patch embedding from a vision
transformer](https://nn.labml.ai/transformers/vit/index.html#PatchEmbeddings). That means breaking
up an image into patches and then linearly embedding each patch.

```python
from eins import EinsOp
patchify = EinsOp('''b (n_p patch) (n_p patch) c, (patch patch c) emb -> b (n_p n_p) emb''')

kernel = randn(5 * 5 * 3, 12)
images = randn(4, 55, 55, 3)
patches = linear(images, kernel)
print(patches.shape)  # (4, 121, 12)
```

You input the shapes and Eins will figure out what to do.

If you've used [`einops`](https://github.com/arogozhnikov/einops), then think of Eins as `einops`
with a more ambitious goal—being the only operation you should need for your next deep learning
project.

Interested? Check out the [tutorial](https://nicholas-miklaucic.github.io/eins/tutorial/), which
walks you through the highlights of Eins with examples of how Eins can make the array operations you
know and love more readable, portable, and robust.

To learn more, consult the [documentation](https://nicholas-miklaucic.github.io/eins/) or
the [examples](examples/README.md).

## Installation

```console
pip install eins
```

Eins works with anything that implements the [Array
API](https://data-apis.org/array-api/latest/index.html), and Eins explicitly promises to support
NumPy, PyTorch, and JAX—including full differentiability. You will need one of those libraries to
actually use Eins operations.

## Features

- 🧩 A solver that can handle duplicate axes, named constants, and constraints
- 🚀 Compilation and optimization for high performance without sacrificing readability
- Split, concatenate, stack, flatten, transpose, normalize, reduce, broadcast, and more
- Works across frameworks
- A composable set of unified array operations for portable softmax, power normalization, and more

## Roadmap

Eins is still in heavy development. Here's a sense of where we're headed.

### Near-Term (weeks)

- [ ] Updating indexing syntax to match `eindex`
- [ ] Unit array to indicate zero-dimensional tensors
- [ ] `...` for batching over dynamic numbers of batch axes
- [ ] Specifying intermediate results to control the order of reduction
- [ ] Support `-` and `/`
- [ ] Better error reporting
- [ ] Ways of visualizing and inspecting the computation graph
- [ ] Typo checking in errors about axes
- [ ] Multiple outputs, either through arrows or commas

### Long-Term (months)

- [ ] Layers for popular ML frameworks
- [ ] Automatically optimizing the execution of a specific EinsOp for a specific
      computer and input size
- [ ] Completing full support for tensor indexing
- [ ] Static typing support
- [ ] Tabulating the model FLOPs/memory usage as a function of named axes
- [ ] Functionality akin to `pack` and `unpack`

## Acknowledgements

The excellent [`einops`](https://github.com/arogozhnikov/einops) library
inspired this project and its syntax. After working on my own extension to
handle indexing, I realized that
[`eindex`](https://github.com/arogozhnikov/eindex) already had a more coherent
vision for what indexing can look like, and so much of that syntax in this
library borrows from that one.

## Contributing

Any contributions to Eins are welcomed and appreciated! If you're interested
in making serious changes or extensions to the syntax of operations, consider
reaching out first to make sure we're on the same page. For any code changes, do
make sure you're using the project Ruff settings.

## License

Eins is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.