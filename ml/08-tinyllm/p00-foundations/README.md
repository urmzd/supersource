# Part 0: Foundations

The model side of the tracer and, from Pass 2, the autograd engine every later model trains with. Pass 1 has one chapter: a byte-level bigram fitted by counting, whose logits go through your C matmul and whose weights you write as a safetensors checkpoint the Rust engine serves. Pass 2 retrains the same bigram with your autograd (L0.5 takes over `bigram.py`) behind the same checkpoint contract.

**Course passes**: 1 (L0.0, gate [MS-P1](../../../paths/course-p01-tracer/milestone.md)), 2 (L0.1 to L0.6, gate MS-P2).

**Before you start**: [Python and numpy](../../../software-craftsmanship/12-language-and-tool-primers/01-python-and-numpy.md), the C ABI ([rt.01](../p09-kernels/01-the-c-abi.md)), and the matmul ([M03.1](../../../math/03-linear-algebra/01-vectors-matrices-and-matmul-in-c.md)).

<!-- ss:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L0.0` | [Byte bigram: counts, logits through C, safetensors v0](00-byte-bigram.md) | build | 1 |
<!-- /ss:chapters -->
