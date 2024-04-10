# Introduction

If you're reading this, that's great! ✨ Contributions of any kind are always more than welcome.
Reading these guidelines before contributing will make it easier both for you to make an impact and
for the maintainers to integrate your work.

There are many ways to contribute to Eins: updating the library documentation, writing standalone
tutorials or guides, identifying bugs, outlining new proposals for Eins syntax and semantics,
writing tests, and writing new code.

# Contributor Responsibilities

## Be decent.

The [Code of Conduct](CODE_OF_CONDUCT.md) elaborates on this point.

## Keep Eins portable and keep Eins installation painless.

Eins is a library, not a framework. Users may be adding Eins only after deciding on a Python
version, a tensor framework, and writing a lot of code. As such, Eins prioritizes a painless
installation process for systems on any common OS, any supported tensor framework, and Python 3.8+,
with or without type checking enabled.

A recommendation is to test/develop in a Python 3.8 environment with Mypy type checking enabled.
This will help ensure that your contributions don't inadvertently break other code.

## Keep the Eins user-facing API as small as possible.

If proposed new functionality would often require users to import something other than `EinsOp`, it
may not be a good fit for the library. Feel free to support typed alternatives to string
configuration, like how the `reduce` argument has `eins.Reductions` to increase discoverability, but
if you can't write motivating examples without an additional import you should be clear about why you think the extra cognitive load on users is worth it.

## Follow the project style.

Eins has a Ruff config that should mostly do this for you. Look at the `EinsOp` code to see the
docstring format. When writing new documentation, in lieu of an explicit style guide, just look at
existing documentation and try to ensure your contributions fit harmoniously within them.

# New Feature Requests

Eins is still in rapid development, so a new feature request may already be in the works. If not,
please file an issue, even if you plan on submitting a pull request yourself—this will make sure
your time is not spent in vain.

A great feature request answers several questions:

- **What would new code using the feature do?** Give at least one example with output or equivalent
  code in a tensor framework.
- **If the new feature involves modifying the EinsOp syntax, how would that work?** Is there any
  situation in which your new proposed syntax would conflict with existing functionality?

  A lot of thought has gone into Eins syntax, and new syntax proposals will get similar
  consideration. If you considered alternative syntax, explain why you decided on what you did. It's
  often quite educational to try to find similar projects or languages and look at how they notate
  specific operations. A "metaphor", even if not completely analogous to another common library, can
  do a lot to help users. For example, the `@local` syntax is taken from Pandas `query()`/`eval()`,
  which is a different library with different purposes but gives users familiar with it a hook.
- **How would the feature be implemented?** New primitives (reshape, transpose, etc.) are
  significantly more work to support and optimize, but sometimes necessary. Remember that Eins
  supports many different tensor frameworks: you can't rely on functionality that only exists in
  Torch.

  **In particular**: note that some frameworks do not have support for dynamic shapes. Eins is built
  around shapes, so this is not an issue for now, but if you intend to support functionality like
  `.nonzero()` do try and think about how it would work in libraries like JAX.

# Filing Bug Reports

When reporting a bug, please answer these questions:

1. What environment are you using (OS, Python version, tensor framework)?
2. What code did you run? If you want bonus points, write a minimal example that can be understood
   outside of your code.
3. What did you expect to happen?
4. What happened instead?