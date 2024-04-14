# Eins Documentation
## Introduction: One Tensor Operation is All You Need

What if most of your machine learning model code could be replaced by a single operation? Eins gives
you a powerful language to describe array manipulation, making it a one-stop shop for all of your AI
needs.

Let's say you want to compute batched pairwise distance between two batches of arrays of points:

```python
from eins import EinsOp
EinsOp('batch n1 d, batch n2 d -> batch n1 n2',
       combine='add', reduce='l2_norm')(pts1, -pts2)
```

If you're more interested in deep learning, here is [the patch embedding from a vision
transformer](https://nn.labml.ai/transformers/vit/index.html#PatchEmbeddings). That means breaking
up an image into patches and then linearly embedding each patch.

```python
from eins import EinsOp
patchify = EinsOp('''b (n_p patch) (n_p patch) c, (patch patch c) emb -> b (n_p n_p) emb''')
patches = patchify(images, kernel)
```

## Installation

Eins just has a few pure Python dependencies, and installation should be as easy as:

```console
pip install eins
```

Eins works with anything that implements the [Array
API](https://data-apis.org/array-api/latest/index.html), and Eins explicitly promises to support
NumPy, PyTorch, and JAX—including full differentiability. You will need one of those libraries to
actually use Eins operations.

## Next Steps

To get started, check out the [tutorial](tutorial.md), which walks you through the most important parts of Eins and shows how you can use Eins to manipulate data in a readable, reliable way.

[Advanced Eins](in-depth.md) explains advanced features and functionality—not always essential, but if you want Eins to be a one-stop shop for all of your tensor manipulation needs you should give it a read.

Finally, if you want to see exactly what's happening under the hood, you can consult the [API documentation](api/eins/). Ideally, there would be no reason to read the API docs unless you're interested in the internals, but it's there if you need it.

If you run into problems, think something should be done differently, or want to suggest improvements to Eins, feel free to [file an issue.](https://github.com/nicholas-miklaucic/eins/issues/new/choose)