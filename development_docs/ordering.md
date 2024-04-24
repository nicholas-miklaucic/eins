# Ordering

Ordering operations is relatively complicated.


```
a b, b c, c d -> a d
```

We can combine this as `a b, b c -> a b c` and then `a b c, c d -> a b c d` or in the other order.
As long as the combination is commutative/associative, this won't matter.

Sometimes, we can save memory by reducing early:

```
a b, b c -> a c
a c, c d -> a d
```

But this isn't generally true.

Here, it works because

$$
\begin{aligned}
{AD}_{ad} &= \sum_{b \in B} \sum_{c \in C} {AB}_{ab} {BC}_{bc} {CD}_{cd} \\
&= \sum_{b \in B} {AB}_{ab} \sum_{c \in C} {BC}_{bc} {CD}_{cd} \\
&= \sum_{c \in C} {CD}_{cd} \sum_{b \in B} {AB}_{ab} {BC}_{bc}
\end{aligned}
$$

The reordering of the summations is fine, because we assume that reductions are commutative. But moving a term outside the sum only works because

```
sum(x * y, x * z) = x * sum(y, z)
```

The vast majority of reductions are folded binary operations. Here, this is expressing the distributive property, and saying that this needs to form a semiring.

There are a couple other examples of common reductions that work with this pattern:

- The tropical semiring: max(x + y, x + z) = x + max(y, z), similarly with minimum.
- Min/minimum, max/maximum. Note that sum/add and multiply/prod don't work, because the difference in order affects whether elements get repeated and then reduced. Max and minimum are idempotent so they don't care.

Many other reductions can be decomposed into elementwise ops and one of the basic central operations. (Basically all of them can: there are basically no other valid operations.) We can use that to generalize a bit.

For example, consider $\max(\sqrt{x + y}, \sqrt{x + z}) = \sqrt{x + \max(y, z)}$.

This is true because of a couple things:
- Max-plus is a semiring, as seen above.
- Square root is monotonic—it commutes with max.

We would like for this to also be true with the Euclidean distance version that squares the inputs. It's only true then if we order the squaring before the max. Such small differences are challenging from a user interface perspective.

Note that the version of this with logs works cleanly, because both log and exp are monotonic.

Note that there are going to be numerical differences in how these are handled. I'm fine with that.