# eins
# One tensor operation is all you need

[![PyPI - Version](https://img.shields.io/pypi/v/eins.svg)](https://pypi.org/project/eins)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eins.svg)](https://pypi.org/project/eins)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install eins
```

## License

`eins` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


To do:

- Parsing, typing for combine/reduce/elementwise
- Debug issues with sums in output
- Typing for compiled operations, including custom functions
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