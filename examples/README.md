# Examples Library

Here, you can find implementations of common layers and modules in deep learning using Eins. The
goal is to show how Eins can make real-world code more readable, more reliable, and sometimes even
more performant.

The reference for these implementations is the excellent site [nn.labml.ai](https://nn.labml.ai/).
The point is not for this library to be a way to learn about machine learning itself, or to directly
replicate all of their work. Instead, this is intended to show what code with Eins looks like, in
comparison to a well-known benchmark.

Flax and JAX are used for the modules. This is because Flax and JAX are functional, which pairs well
with the functional nature of Eins. It is certainly possible to wrap Eins functions in PyTorch
modules, and perhaps in the future a set of modules/layers with Eins interfaces can be written for
other frameworks, but the goal here is to focus on manipulating data and not storing parameters.