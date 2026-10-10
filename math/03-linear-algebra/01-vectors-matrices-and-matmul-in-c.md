<!-- ss:module M03.1 -->
# Vectors, matrices, row-major layout, and a naive matmul in C

## Overview

| | |
|---|---|
| **Module** | `M03.1` · build · C · Pass 1 · 3 to 4 h |
| **You build** | `c/src/kernels/matmul.c`: `tl_matmul_f32`, version 0 (a direct triple loop), computing $C \leftarrow \alpha\, A\, \mathrm{op}(B) + \beta\, C$ on row-major float32 matrices with explicit leading dimensions |
| **Contract** | [`course/contracts/c/include/tinyllm/matmul.h`](../../course/contracts/c/include/tinyllm/matmul.h) · rules: [`c/ABI.md`](../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/M03.1/`: `test_matmul.c` (C, under ASan and UBSan) and `test_matmul_ctypes.py` (Python, through your `rt.01` loader, against numpy) (what they check: section 4) |
| **Needs** | `rt.01` the error slot and the ctypes loader (or `--ref-deps`). Reading: `lang.03` (pointers and arrays in C) |
| **Used by** | `L0.0` computes its bigram logits as `onehot(ids) @ weight` with it · `L10.0` your Rust engine calls it through `extern "C"` · later `L9.1` takes the file over with a tiled, packed, batch-invariant version behind the same signature · later: `L0.5` |
| **Milestone** | `MS-P1` (the tracer: every layer is yours and runs end to end) |
| **Optional depth** | Hefferon, *Linear Algebra* (free), ch. 3, sections IV.1 and IV.2 (matrix multiplication); Strang, MIT 18.06 lecture 3; Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008) for what `L9.1` does next |

## Key Takeaways

- A **matrix product** $C = AB$ is one **dot product** per output entry: $C_{ij} = \sum_k A_{ik} B_{kj}$, defined only when $A$'s column count equals $B$'s row count (`hand_example`).
- **Row-major storage** puts entry $(i, k)$ of a matrix at offset $i \cdot \mathit{ld} + k$, where the **leading dimension** $\mathit{ld}$ is the distance between rows, at least the row length; a view into a wider buffer has a larger $\mathit{ld}$ (`leading_dimensions_skip_padding`, `test_strided_views_use_leading_dimensions`).
- **`trans_b`** reads $B$ as stored $N \times K$, the way a neural network layer stores its weight, without copying it (`trans_b_reads_rows_of_b`).
- **$\beta = 0$ means overwrite**: the output is never read, so garbage or NaN already in $C$ cannot leak into the result (`beta_zero_ignores_garbage_in_c`).
- **Floating-point sums depend on order.** Your result differs from numpy's in the last bits, so it is compared with a float64 reference within a tolerance that grows like $\sqrt{K}$ (`test_matches_numpy_across_shapes`).

## How to work this chapter

```bash
ss start M03.1              # stubs c/src/kernels/matmul.c into your repo
ss tests M03.1              # read the test catalog first: rung R0, you write no tests here
ss check M03.1              # exit code is the verdict
ss check M03.1 --ref-deps   # only if your rt.01 is not passing yet
ss diff  M03.1              # after passing: your code against the reference
```

---

## 1. Why now

Your system can load a C library from Python (`rt.01`), but the library computes nothing. The next module, `L0.0`, trains a language model whose whole forward pass is one matrix product: the logits for a sequence of $T$ bytes are a $T \times 256$ matrix obtained by multiplying a one-hot matrix by the model's $256 \times 256$ weight table. Your Rust engine (`L10.0`) computes the same product to serve the model over HTTP. Both call `tl_matmul_f32`, and right now that symbol is a stub that returns `TL_EUNSUPPORTED`. Every model in the rest of the course, up to a Llama-family transformer, spends most of its time in this one operation, so it is also where your C performance work starts in `L9.1`. This module defines vectors, matrices, and their product from the beginning, fixes how a matrix is laid out in memory, and implements the product in the most direct correct way, under sanitizers and against numpy.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\mathbb{R}^n$ | the set of lists of $n$ real numbers (vectors of length $n$) | |
| $u, v \in \mathbb{R}^n$ | vectors; $u_k$ is entry $k$, counted from 0 | `float[n]` |
| $u \cdot v$ | dot product, $\sum_{k=0}^{n-1} u_k v_k$ | scalar |
| $A \in \mathbb{R}^{M \times K}$ | a matrix of $M$ rows and $K$ columns; $A_{ik}$ is row $i$, column $k$ | `float[M][K]` |
| $B \in \mathbb{R}^{K \times N}$ | the right factor | `float[K][N]` |
| $C \in \mathbb{R}^{M \times N}$ | the product, or the output buffer | `float[M][N]` |
| $B^\top$ | the transpose: $(B^\top)_{jk} = B_{kj}$ | $N \times K$ |
| $\mathrm{op}(B)$ | $B$ when `trans_b == 0`, $B^\top$ of the stored matrix when `trans_b != 0` | $K \times N$ |
| $\alpha, \beta$ | scalars: `alpha` scales the product, `beta` scales the old $C$ | `float` |
| $\mathit{lda}, \mathit{ldb}, \mathit{ldc}$ | leading dimensions: elements between the starts of consecutive rows | `int64_t` |
| $\varepsilon$ | float32 unit roundoff, $2^{-24} \approx 6 \times 10^{-8}$ | |

### 2.1 Vectors and the dot product

A **vector** of length $n$ is an ordered list of $n$ numbers, $v = (v_0, \dots, v_{n-1})$. Two vectors of the same length add entry by entry, and a number $c$ (a **scalar**) scales every entry: $(u + v)_k = u_k + v_k$, $(cv)_k = c\, v_k$. The **dot product** multiplies matching entries and adds the results:

$$u \cdot v = \sum_{k=0}^{n-1} u_k v_k .$$

For $u = (1, 2, 3)$ and $v = (7, 9, 11)$: $1 \cdot 7 + 2 \cdot 9 + 3 \cdot 11 = 58$. The dot product measures how much two vectors point the same way ($u \cdot u$ is the squared length); `M03.6` builds cosine similarity on it. Here it is the unit of work: everything below is many dot products.

### 2.2 Matrices and their product

A **matrix** $A$ with $M$ rows and $K$ columns is a grid of $M \cdot K$ numbers; row $i$ is a vector of length $K$. Multiplying $A$ by a vector $x \in \mathbb{R}^K$ gives the vector of dot products of each row with $x$: $(Ax)_i = \sum_k A_{ik} x_k$. That is a **linear map** from $\mathbb{R}^K$ to $\mathbb{R}^M$: it respects addition and scaling, $A(x + y) = Ax + Ay$ and $A(cx) = c\,Ax$.

The **product** of $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ applies $A$ to each column of $B$. Entry $(i, j)$ is the dot product of row $i$ of $A$ with column $j$ of $B$:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}, \qquad 0 \le i < M,\ 0 \le j < N .$$

It exists only when the **inner dimensions** agree ($A$ has $K$ columns, $B$ has $K$ rows), and the result has $A$'s rows and $B$'s columns. It is associative, $(AB)D = A(BD)$, which is why composing two linear maps is one matrix. It is **not** commutative: $AB$ and $BA$ usually differ, and one of them may not even exist. The **transpose** swaps rows and columns, $(B^\top)_{jk} = B_{kj}$, and reverses products: $(AB)^\top = B^\top A^\top$.

Two products you will meet in `L0.0` and every later model:

- **Selecting rows.** If row $t$ of $X$ is the one-hot vector $e_{c}$ (1 at position $c$, 0 elsewhere), then row $t$ of $XW$ is $\sum_k (e_c)_k W_{kj} = W_{cj}$: exactly row $c$ of $W$. A bigram model's logits are $\mathrm{onehot}(\mathit{ids})\, W$.
- **A linear layer.** A layer with $K$ inputs and $N$ outputs stores its weight as $W \in \mathbb{R}^{N \times K}$ (one row per output, the PyTorch and safetensors convention) and computes $y = x W^\top$. With `trans_b` the kernel uses $W$ as stored.

### 2.3 The GEMM convention

Numerical libraries since BLAS (1979) expose one general matrix multiply, **GEMM**, that also scales and accumulates:

$$C \leftarrow \alpha\, A\, \mathrm{op}(B) + \beta\, C .$$

$\alpha = 1, \beta = 0$ is the plain product; $\beta = 1$ adds the product into $C$ (a residual connection, or a sum over chunks of $K$). Two edge rules matter. **$\beta = 0$ means "overwrite", and $C$ is not read**: a caller passes a fresh, uninitialized buffer, and computing $0 \cdot C_{ij}$ would turn any NaN already there into NaN, because IEEE 754 defines $0 \cdot \mathrm{NaN} = \mathrm{NaN}$ (and $0 \cdot \infty = \mathrm{NaN}$). **$K = 0$ is an empty sum**, which is 0, so $C \leftarrow \beta C$; with $M = 0$ or $N = 0$, $C$ has no entries and nothing happens.

### 2.4 Row-major layout and leading dimensions

Memory is one-dimensional (`lang.03` 2.1). **Row-major** order stores row 0, then row 1, and so on, so in a tightly packed $M \times K$ matrix entry $(i, k)$ is at element offset $iK + k$. The kernel takes the distance between rows as a separate parameter, the **leading dimension**:

$$\mathrm{addr}(A_{ik}) = A + (i \cdot \mathit{lda} + k), \qquad \mathit{lda} \ge K ,$$

and likewise $C_{ij}$ at $i \cdot \mathit{ldc} + j$ with $\mathit{ldc} \ge N$. A packed matrix has $\mathit{lda} = K$. A **view** has a larger one: columns 2 to 6 of a $6 \times 11$ array are a $6 \times 5$ matrix whose rows are still 11 elements apart, so $\mathit{lda} = 11$ and the bytes between rows belong to someone else. Engines live on views (one attention head's slice of a fused projection), so the kernel must never assume $\mathit{lda} = K$, and must never write $C$'s padding. In numpy, a float32 view's row stride in bytes divided by 4 is its leading dimension.

`trans_b` changes only how $B$ is read. With `trans_b == 0`, $B$ is stored $K \times N$ and $\mathrm{op}(B)_{kj}$ is at $k \cdot \mathit{ldb} + j$ ($\mathit{ldb} \ge N$). With `trans_b != 0`, $B$ is stored $N \times K$ and $\mathrm{op}(B)_{kj} = B_{jk}$ is at $j \cdot \mathit{ldb} + k$ ($\mathit{ldb} \ge K$). No copy is made.

### 2.5 The naive algorithm and its cost

The definition is the algorithm: for each $i$, for each $j$, sum over $k$. Each output entry costs $K$ multiplications and $K$ additions, so the product costs $2MNK$ floating-point operations: $2 \cdot 512^3 \approx 2.7 \times 10^8$ for $512 \times 512$ matrices. The triple loop is correct but slow, because with `trans_b == 0` the inner loop walks down a column of $B$, $\mathit{ldb}$ elements apart, touching a new cache line on almost every step. `L9.1` keeps the same signature and reorganizes the loops into tiles that stay in cache (at least 10 times faster at $512^3$). For now, correctness is the whole job.

Validation comes first and touches nothing: a negative dimension, a leading dimension below its minimum, or a NULL pointer that would be read or written returns `TL_EINVAL` with a message in the error slot (`rt.01`), and $C$ is left exactly as it was. A pointer may be NULL when nothing is read through it ($A$ and $B$ when $K = 0$; all three when $M = 0$ or $N = 0$). Do not even compute `A + i * lda` from a NULL `A`: pointer arithmetic on NULL is undefined behavior, and UBSan reports it.

### 2.6 Why your answer differs from numpy's

A float32 number keeps 24 significant bits, so every addition rounds, with a relative error of at most $\varepsilon = 2^{-24}$. Rounding makes addition non-associative: $(a + b) + c$ and $a + (b + c)$ can differ in the last bit. numpy calls an optimized BLAS that adds the $K$ products in blocks and in a different order than your loop, so the two results agree to about 6 or 7 significant digits, not bit for bit. Each of the $K$ additions contributes an error of order $\varepsilon$ times the running sum; for random signs they partly cancel, and the typical total grows like $\sqrt{K}\,\varepsilon$ (the worst case like $K\varepsilon$). So the tests compare your float32 result with a **float64** reference within the course's float32 tolerances scaled by $\sqrt{K}$: $|c - c^\star| \le \sqrt{K}\,(10^{-6} + 10^{-5}\,|c^\star|)$, from the frozen `course/tests/_lib/close.py`. `M09.3` derives these bounds properly. The worked example and every test with small integers are exact, because integers below $2^{24}$ and their sums are represented exactly in float32.

## 3. Worked example by hand

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \quad B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}, \quad M = 2,\ K = 3,\ N = 2 .$$

Packed row-major: `A = {1, 2, 3, 4, 5, 6}` with $\mathit{lda} = 3$, `B = {7, 8, 9, 10, 11, 12}` with $\mathit{ldb} = 2$, $C$ with $\mathit{ldc} = 2$. Each entry, with the memory offsets the loop reads:

| $(i, j)$ | $\sum_k A_{ik} B_{kj}$ | offsets in A | offsets in B | $C_{ij}$ |
|---|---|---|---|---|
| $(0, 0)$ | $1 \cdot 7 + 2 \cdot 9 + 3 \cdot 11 = 7 + 18 + 33$ | 0, 1, 2 | 0, 2, 4 | **58** |
| $(0, 1)$ | $1 \cdot 8 + 2 \cdot 10 + 3 \cdot 12 = 8 + 20 + 36$ | 0, 1, 2 | 1, 3, 5 | **64** |
| $(1, 0)$ | $4 \cdot 7 + 5 \cdot 9 + 6 \cdot 11 = 28 + 45 + 66$ | 3, 4, 5 | 0, 2, 4 | **139** |
| $(1, 1)$ | $4 \cdot 8 + 5 \cdot 10 + 6 \cdot 12 = 32 + 50 + 72$ | 3, 4, 5 | 1, 3, 5 | **154** |

The call is `tl_matmul_f32(A, B, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL)`: the arguments are $A, B, C$, then $M, N, K$, then $\mathit{lda}, \mathit{ldb}, \mathit{ldc}$, then $\alpha, \beta$, `trans_b`, and the pool. Result `{58, 64, 139, 154}`: the first test, `hand_example`, and its Python twin `test_hand_example_through_ctypes`.

**The same product with $B$ stored transposed.** Store $B^\top = \begin{pmatrix} 7 & 9 & 11 \\ 8 & 10 & 12 \end{pmatrix}$ as `{7, 9, 11, 8, 10, 12}`, $N \times K$ with $\mathit{ldb} = 3$, and pass `trans_b = 1`. Entry $(1, 0)$ reads $\mathrm{op}(B)_{k0}$ at $0 \cdot 3 + k$ = offsets 0, 1, 2: values 7, 9, 11, the same column as before. Result unchanged (`trans_b_reads_rows_of_b`).

**The same product inside wider buffers.** Put $A$ in a $2 \times 5$ buffer whose last two columns hold $10^{30}$ (so any read of them is obvious), $B$ in a $3 \times 4$ buffer, and $C$ in a $2 \times 3$ buffer filled with $-1$: $\mathit{lda} = 5$, $\mathit{ldb} = 4$, $\mathit{ldc} = 3$. $A_{12} = 6$ is now at offset $1 \cdot 5 + 2 = 7$; $C_{10}$ is written at $1 \cdot 3 + 0 = 3$. Offsets 2 and 5 of $C$ (its padding) must still hold $-1$ afterwards (`leading_dimensions_skip_padding`).

**Scaling and accumulating.** With $C = \begin{pmatrix} 2 & 4 \\ 6 & 8 \end{pmatrix}$, $\alpha = 2$, $\beta = 0.5$: $C_{00} = 2 \cdot 58 + 0.5 \cdot 2 = 117$, and the rest are $130, 281, 312$ (`alpha_and_beta_scale_and_accumulate`).

## 4. The interface

```c
/* tinyllm/matmul.h */
tl_status tl_matmul_f32(const float *A, const float *B, float *C,
                        int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc,
                        float alpha, float beta, int trans_b, tl_pool *tp);
/* C = alpha * A @ op(B) + beta * C. TL_OK, or TL_EINVAL (slot set, C untouched) for a
   negative dimension, lda < K, ldb < N (ldb < K with trans_b), ldc < N, or a NULL
   pointer that would be read or written. beta == 0: C is written, never read.
   tp may be NULL; v0 ignores it (the thread pool arrives with rt.03). */
```

From Python, through your `rt.01` loader:

```python
from tinyllm.ffi.libtinyllm import load, f32_ptr
C = np.zeros((M, N), dtype=np.float32)
load().tl_matmul_f32(f32_ptr(A), f32_ptr(B), f32_ptr(C), M, N, K,
                     A.strides[0] // 4, B.strides[0] // 4, C.strides[0] // 4, 1.0, 0.0, 0, None)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3's numbers exactly | you and the tests agree on the definition |
| `trans_b_reads_rows_of_b` | unit | `trans_b = 1` on $B^\top$ gives the same product | weights stored `[out, in]` in `L10.0` and later layers |
| `beta_zero_ignores_garbage_in_c` | boundary | NaN and infinity in $C$ do not reach the result when $\beta = 0$ | uninitialized output buffers everywhere |
| `alpha_and_beta_scale_and_accumulate` | unit | $\alpha = 2$, $\beta = 0.5$ exactly | accumulation over chunks of $K$ |
| `leading_dimensions_skip_padding` | unit | reads through `lda`, `ldb`, writes through `ldc`, padding untouched | views into larger buffers |
| `empty_dimensions` | boundary | $M = 0$ and $N = 0$ are no-ops; $K = 0$ gives $\beta C$ | empty prompts and batches |
| `bad_arguments_are_einval` | boundary | every invalid argument gives `TL_EINVAL`, a message naming `tl_matmul_f32`, and $C$ unchanged | errors instead of crashes across the ABI |
| `test_hand_example_through_ctypes` | unit, smoke | section 3 from numpy arrays through `load()` | exactly how `L0.0` calls it |
| `test_matches_numpy_across_shapes` | differential | shapes from $1 \times 1 \times 1$ to $2 \times 65 \times 128$, both `trans_b`, within the $\sqrt{K}$ bound against float64 | any correct summation order passes, any wrong index fails |
| `test_strided_views_use_leading_dimensions` | differential | numpy row-strided views in and out; nothing outside $C$'s columns changes | `L10.0` and attention slices |
| `test_alpha_beta_property` | property | $\alpha = 2$ doubles the result bit for bit; $\beta = 1$ adds $C$ once | exactness of scaling by powers of two |
| `test_einval_raises_through_the_loader` | boundary | a short `lda` arrives in Python as `TlError` with `TL_EINVAL` | the error path end to end |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| computing `alpha * acc + beta * c[j]` when $\beta = 0$ | NaN in a fresh output buffer becomes NaN in the logits | `beta_zero_ignores_garbage_in_c` (mutant `s03`) |
| indexing $C$ as `i * N + j` (or $A$ as `i * K + k`) | correct on packed matrices, wrong on every view, and writes into the neighbour's columns | `leading_dimensions_skip_padding`, `test_strided_views_use_leading_dimensions` (mutants `s05`, `s06`) |
| ignoring `trans_b` | a Linear layer's weight is used transposed; shapes may even match when $N = K$ | `trans_b_reads_rows_of_b` (mutant `s02`) |
| returning early when $K = 0$ | $C$ keeps its old values instead of $\beta C$ (zeros when $\beta = 0$) | `empty_dimensions` (mutant `s07`) |
| declaring the accumulator once per row | each entry starts from the previous entry's sum | `hand_example` (mutant `s01`) |
| skipping argument checks | a short leading dimension reads past the buffer: ASan in tests, silent garbage in production | `bad_arguments_are_einval`, `test_einval_raises_through_the_loader` (mutant `s08`) |
| expecting numpy's float32 answer bit for bit | your correct kernel "fails" by $10^{-7}$; the fix is the $\sqrt{K}$ bound, not a different loop | `test_matches_numpy_across_shapes` |

## 6. Where it's used next
| Forward | `L0.5` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.01` | the error slot behind `TL_EINVAL`, and the ctypes loader the Python tests call through |
| Back | `lang.03` | pointers, arrays, and `const float *` in C (reading) |
| Forward | `L0.0` | `BigramLM.logits` = `onehot(ids) @ weight` with $M = T$, $N = K = 256$ |
| Forward | `L10.0` | the Rust engine computes the same logits by calling `tl_matmul_f32` through `extern "C"` |
| Forward | `L9.1` | takes `c/src/kernels/matmul.c` over: cache tiling, packing, and batch invariance behind this signature; this module's tests become its regression |
| Forward | `M03.2` | Gaussian elimination and LU work on the same row-major matrices, in Python |

If you skip this module, `ss check L0.0` stops with `BLOCKED ... needs M03.1`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_matmul_f32` v0 | BLAS `sgemm` (reference BLAS, OpenBLAS, BLIS, Apple Accelerate) | the same $\alpha, \beta$, transposes, and leading dimensions, but column-major; packing and register-blocked micro-kernels reach most of the chip's peak | BLIS `kernels/`; `cblas_sgemm` in Accelerate |
| triple loop | Goto's algorithm | blocks sized for L1, L2, and L3 caches; your `L9.1` | Goto and van de Geijn (2008) |
| float32 accumulation | mixed-precision GEMM (bf16 or fp8 inputs, fp32 accumulators) | 2 to 4 times the throughput on matrix units | NVIDIA cuBLASLt; `L9.5` quantized matmul |
| CPU kernel behind a C ABI | llama.cpp `ggml_mul_mat` | dispatch over quantized formats and backends (CPU, Metal, CUDA) | `ggml/src/ggml-cpu/` |
