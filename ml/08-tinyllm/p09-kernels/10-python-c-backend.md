<!-- ss:module L9.7 -->
# The Python C backend: the Llama forward through libtinyllm, with a load-time op check

## Overview

| | |
|---|---|
| **Module** | `L9.7` · build · Python · Pass 6 · 4 to 6 h |
| **You build** | `python/tinyllm/backend/c.py`: one wrapper per C op (`embedding`, `rmsnorm`, `linear`, `rope`, `attention`, `silu_mul`, `add`, `argmax`), `signatures`, the load-time op check (`check_ops`, `OpCheck`, `OpCheckError`), and `CBackend` with its cache (`forward`, `forward_varlen`, `new_cache`) |
| **Contract** | [`course/contracts/py/tinyllm/backend/c.pyi`](../../../course/contracts/py/tinyllm/backend/c.pyi) · the kernel headers [`elementwise.h`](../../../course/contracts/c/include/tinyllm/elementwise.h), [`attention.h`](../../../course/contracts/c/include/tinyllm/attention.h), [`matmul.h`](../../../course/contracts/c/include/tinyllm/matmul.h) |
| **Tests** | `course/tests/L9.7/test_backend_c.py` (what they check: section 4); your own tests in `python/tests/l9-7-backend/` are graded at rung R7 |
| **Needs** | `L9.1` batch-invariant matmul · `L9.3` FlashAttention forward · `L9.6` elementwise kernels · `rt.01` the ctypes loader · `M09.3` error bounds · `L7.9` the Llama model whose forward this runs (or `--ref-deps`) |
| **Used by** | `L10.1`: the Rust runner's logits are compared with `--backend c` logits (it joins the registry in Pass 7) · your `{tinyllm} generate --backend c` |
| **Milestone** | `MS-L9` (your kernels run SmolLM2 and match HF token for token) |
| **Optional depth** | llama.cpp's `ggml` backend scheduler (`ggml-backend.c`); PyTorch's dispatcher ("Let's talk about the PyTorch dispatcher", ezyang's blog); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 3 |

## Key Takeaways

- A **backend** keeps the model's math and changes where each op runs. Your C backend reuses L7.9's parameters by name and sends every op of the forward through one ctypes call (`test_tiny_hf_logits_golden`).
- **A Linear with a bias is one GEMM**: fill the output with the bias rows, then `tl_matmul_f32` with $\beta = 1$. The buffer must be fresh: `ascontiguousarray(broadcast_to(b, (1, M)))` is `b` itself, and the kernel would write into your weights (`test_hand_example_linear_bias_and_rope`).
- **A load-time op check** runs each op once on the loaded model's own shapes and divides its error by the `M09.3` bound. A kernel wrong by $10^{-5}$ relative, invisible in logits, is named before the first token (`test_op_check_names_a_wrong_kernel`).
- **Invariance is a property you can test bitwise**: chunked decode through the cache, a packed batch of different lengths, and a row of a batch all give exactly the logits of the sequence alone (`test_cached_decode_is_bitwise_the_full_forward`, `test_varlen_batch_equals_per_sequence_calls`).
- A backend **refuses what its kernels cannot compute** (latent attention, experts, GeLU, sink tokens) instead of producing plausible wrong text (`test_rejects_unsupported_configs`).

## How to work this chapter

```bash
ss start L9.7              # stubs backend/c.py into your repo, contract alongside
ss tests L9.7              # read the test catalog first
ss check L9.7              # exit code is the verdict; then your R7 tests are graded by mutation
ss check L9.7 --ref-deps   # only if L9.1, L9.3, or L9.6 are not passing yet
ss diff  L9.7              # after passing: your code against the reference
```

`ss check` builds `libtinyllm` from your C units (the references with `--ref-deps`) and puts it in `TINYLLM_LIB`; your backend reaches it through `rt.01`'s `load()`. Your own tests in `python/tests/l9-7-backend/` run against the reference backend with one planted bug at a time; rung R7 asks for 0.90 of them, including every tagged one.

---

## 1. Why now

You have two halves that have never met. Part 7 gave you a Llama-family model in Python (`L7.9`) that matches Hugging Face on SmolLM2, but every op runs as numpy on autograd tensors. Part 9 gave you C kernels (`L9.1` matmul, `L9.3` FlashAttention, `L9.6` RMSNorm, RoPE, SiLU, embedding, add, argmax), each proven against a Python oracle on random inputs. Nothing yet proves that **together, on a real model's shapes**, they compute the same forward. In Pass 7 your Rust engine (`L10.1`) will run that forward over the same kernels, and when its logits disagree with Python you will not know whether the bug is in Rust, in a kernel, or in how a kernel is called. This module removes two of the three suspects first: the Python backend calls the kernels exactly as the engine will, and `--backend c` becomes the oracle `L10.1` is compared with. It also adds the check that every production backend wants and few have: before serving a model, prove each kernel is correct for *that* model.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $N$ | rows of activations in one call: $B \cdot T$ tokens, or the packed tokens of a varlen batch | integer |
| $d, H, H_{kv}, D$ | hidden size, query heads, key/value heads, head dim | integers |
| $x \in \mathbb{R}^{N \times K}$, $W \in \mathbb{R}^{M \times K}$, $b \in \mathbb{R}^{M}$ | a Linear's input, weight (stored `[out, in]`), bias | `float32` |
| $\beta$ | GEMM's scale on the existing output: $C \leftarrow x W^\top + \beta C$ | 0 or 1 |
| $s$ | the cache length before this chunk; the first new query sits at position $s$ | integer |
| $u = 2^{-24}$ | the float32 unit roundoff (`M09.3`) | scalar |
| $\gamma_k = k u / (1 - k u)$ | the accumulated relative error of $k$ roundings (`M09.3`) | scalar |
| $\rho = \max_i \lvert \hat y_i - y_i \rvert / \mathrm{bound}_i$ | an op's bound ratio: $\rho \le 1$ is inside the budget | scalar |

### 2.1 What a backend is

L7.9's `LlamaForCausalLM.forward` is a fixed sequence of ops: embed, then per layer RMSNorm, three projections, RoPE on $q$ and $k$, attention, the output projection, a residual add, RMSNorm, gate and up projections, SiLU-mul, down projection, a residual add; then a final RMSNorm and the head. A backend is a second implementation of exactly that sequence over different primitives. It owns no parameters of its own: `CBackend` copies `model.state_dict()` once, by its Hugging Face names (`model.layers.3.self_attn.q_proj.weight`), as contiguous float32, and reads the rest from `config` and `config.rope_spec()`. Two backends of one model can therefore only disagree because an op disagrees, which is what makes the comparison useful.

### 2.2 Declaring the kernels

`rt.01`'s loader knows the v0 header only. Every later kernel is declared once per `Lib` with `lib.declare(name, restype, argtypes)`, from a table written off the headers: `int64_t` is `c_int64`, `int` is `c_int`, `float *` is `POINTER(c_float)`, `int32_t *` is `POINTER(c_int32)`, `tl_arena *` and `tl_pool *` are `c_void_p`, a `void` function has `restype = None`, and `tl_flash_attn_fwd_f32` returns `STATUS` so a failure raises `TlError`. ctypes believes the table, not the header: one `c_int` where the header says `int64_t` passes the wrong bits with no error at all.

Every wrapper converts its inputs to C-contiguous float32 (int32 for ids and positions) and allocates a fresh output. Contiguity matters because the kernels take raw pointers plus dimensions: a transposed numpy view is the same memory read in the wrong order.

### 2.3 One forward, op by op

The rows of every activation matrix are tokens, so one call covers all $N$ of them:

| Step | Op | Shapes |
|---|---|---|
| $h = E[\text{ids}]$ | `embedding` | `[N, d]` |
| $q, k, v$ | `rmsnorm` then three `linear` | `[N, H D]`, `[N, H_kv D]` |
| rotate $q$, $k$ | `rope` on `[N, H, D]` and `[N, H_kv, D]`, one position per row | in place on a copy |
| $o$ | `attention` on `[B, H, T, D]` against `[B, H_kv, s + T, D]` | causal, `q_offset` $= s$ |
| $h \mathrel{+}= o W_o^\top$ | `linear`, `add` | `[N, d]` |
| MLP | `rmsnorm`, `linear` (gate, up), `silu_mul`, `linear` (down), `add` | `[N, d_{ff}]` |
| logits | `rmsnorm`, `linear` with the head (or the embedding when tied) | `[N, V]` |

**The bias rides in the GEMM.** `tl_matmul_f32` computes $C \leftarrow \alpha\, x W^\top + \beta C$ with `trans_b = 1`, so the weight is used as stored. With a bias, the output is first filled with $b$ in every row and the call runs with $\beta = 1$; without one, $\beta = 0$ and the kernel never reads the output, so `np.empty` is safe. The fill must go into a **new** buffer. `np.ascontiguousarray(np.broadcast_to(b, (N, M)))` copies when $N > 1$, but for one row the broadcast view already is contiguous, so numpy returns it unchanged: it *is* `b`, and the kernel writes $x W^\top + b$ over your bias. Decode runs one row per step, so the second token reads a corrupted bias.

**Attention** gets `scale` $= D^{-1/2}$, `causal = 1`, `q_offset` $= s$ (query $i$ is at absolute position $s + i$), `window` from `config.sliding_window` (0 for none), and `sink_logits` from the layer's learned `sinks` when it has them. RoPE gets `rope_spec()`'s `inv_freq`, `attention_scaling` (YaRN's mscale), `rotary_dim` (a partial rotary model rotates only the first dims), and the layout flag (0 half, 1 interleaved).

**What the kernels cannot do, the backend refuses.** Latent attention (`attention = "mla"`) has queries and keys wider than values, which `tl_flash_attn_fwd_f32` does not take; mixture-of-experts layers need routing no kernel does; `gelu_tanh` is not `silu_mul`; and sink tokens are exempt from the window, which the kernel's window rule cannot express. Each is a `ValueError` naming the feature, raised before any C call.

### 2.4 The load-time op check

`L9.1` to `L9.6` were tested on shapes the tests chose. A model brings its own: a head dim of 6, a vocabulary of 49152, positions up to 8191, a sliding window of 4. `check_ops(model)` runs each op once on the loaded model's shapes, with its layer 0 and head weights where the op has weights, and inputs from a fixed formula (no RNG, so the check is the same everywhere). It computes the same op in numpy float64 and divides the error by its bound from `M09.3`:

| Op | Bound per output |
|---|---|
| `embedding`, `argmax` | 0: exact (a copy, an index) |
| `add` | $u \lvert a + b \rvert$: one rounding |
| `matmul` (8 calls: $q, k, v, o$, gate, up, down, head) | $\gamma_K \lvert x \rvert \lvert W \rvert^\top$: `matmul_error_bound` |
| `rmsnorm` | $\gamma_{d+8} \lvert y \rvert$: a $d$-term sum of squares, a square root, a division, two products, with slack for `L9.6`'s 2-ulp `tl_rsqrtf` |
| `silu_mul` | $\gamma_{12} \lvert y \rvert$: a 2-ulp `tl_expf`, an add, a divide, a multiply |
| `rope` | $\gamma_4 (\lvert a c \rvert + \lvert b s \rvert)$ for $a' = ac - bs$, and $\gamma_4 (\lvert a s \rvert + \lvert b c \rvert)$ for $b' = as + bc$ |
| `attention` | $(2E + \gamma_{3T_k + 16}) (P \lvert V \rvert)$, with $E = \gamma_{D+2}\, \text{scale} \max_j \lvert q \rvert \cdot \lvert k_j \rvert + u (\max_j s_j - \min_j s_j)$ |

The attention bound follows the error through: each score $s_j$ is a $D$-term dot product (error at most $\gamma_{D+2}$ times its absolute value), and subtracting the running max rounds the exponent by $u \lvert s_j - m \rvert$; a softmax shifts its probabilities by at most about twice the largest score error; the weighted sum of $T_k$ values adds $\gamma_{T_k}$, and the online rescales and exponentials stay inside $\gamma_{3T_k + 16}$. Each row's bound is relative to $P \lvert V \rvert$, the weighted magnitude of the values, so cancellation in $PV$ cannot make it zero.

Each op gets one `OpCheck(op, calls, ratio, ok, detail)` with $\rho$ the worst ratio over its calls. `CBackend(model, check=True)` raises `OpCheckError` when any row has $\rho > 1$, and a C function that raises `TlError` (a stubbed unit) is a failed row with $\rho = \infty$, so `--check` lists every missing kernel at once instead of crashing on the first. The bounds hold for every input; a correct kernel can never cross them, and a kernel off by $10^{-5}$ relative crosses them by a factor of 10 or more.

### 2.5 Invariance

Batched serving is only safe if batching changes no one's output. Two kernel properties give that: `L9.1`'s matmul reduces each output element over $K$ in one fixed order whatever $M$ and the row's position (batch invariance), and `L9.3`'s attention reduces over keys in tiles aligned to absolute key positions whatever the query chunking (chunk invariance). Every other op is per row. So, **bitwise**:

- running a sequence through the cache in chunks of 1, 4, 1, and the rest gives the logits of running it whole: each chunk's rows meet the same reductions;
- `forward_varlen` packs sequences of lengths 3, 12, and 7 into one 22-row pass (every projection one call), restarts positions at 0 for each sequence, and runs attention per sequence on its own rows: each sequence's logits equal its own `forward`;
- row $i$ of a `[B, T]` batch equals row $i$ run alone, provided every row's positions restart at $s$.

The cache stores `[B, H_kv, capacity, D]` per layer and doubles its capacity when full, copying what it has; attention reads a contiguous copy of the first $s + T$ positions, because the kernel takes a packed `[B, H_kv, T_k, D]` tensor.

## 3. Worked example by hand

**A Linear with a bias is one GEMM.** $x = (1, 2)$, $W = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}$ (stored `[out, in]`), $b = (0.5, -1)$.

| Step | Computation | Result |
|---|---|---|
| fill the output | a fresh `[1, 2]` buffer set to $b$ | $(0.5, -1)$ |
| `tl_matmul_f32(trans_b=1, beta=1)` | $C_j = (x \cdot W_j) + 1 \cdot C_j$: $(1 \cdot 1 + 2 \cdot 0) + 0.5$, $(1 \cdot 3 + 2 \cdot 1) - 1$ | $(1.5, 4)$ |
| the bound ($K = 2$) | $\gamma_2 (\lvert x \rvert \lvert W \rvert^\top) = \frac{2u}{1 - 2u} (1, 5) \approx (1.19 \cdot 10^{-7}, 5.96 \cdot 10^{-7})$ | |
| the ratio | every product and sum here is exact in float32 | $\rho = 0$ |
| the same call again | a fresh buffer again: $b$ is untouched | $(1.5, 4)$ |

With the aliasing bug, the first call returns $(1.5, 4)$ but leaves $b = (1.5, 4)$ (the view *was* $b$); the second call returns $(1 + 1.5, 5 + 4) = (2.5, 9)$.

**One RoPE pair.** Position 1, `inv_freq` $= 1$, layout half, $D = 2$: the angle is $1 \cdot 1 = 1$ (formed in double, rounded once to float), $c = \cos 1 = 0.5403023$, $s = \sin 1 = 0.8414710$, and the pair $(a, b) = (1, 0)$ becomes $(a c - b s,\ a s + b c) = (0.5403023, 0.8414710)$.

**One RMSNorm row.** $x = (3, 4)$, gain 1, $\epsilon = 0$: $\sqrt{(9 + 16)/2} = \sqrt{12.5} = 3.5355339$, so $y = (3, 4) / 3.5355339 = (0.8485281, 1.1313708)$.

All three are `test_hand_example_linear_bias_and_rope`.

## 4. The interface

```python
OPS = ("embedding", "rmsnorm", "matmul", "rope", "attention", "silu_mul", "add", "argmax")
def signatures() -> dict[str, tuple[restype, list[argtype]]]       # declared on each Lib once
def embedding(table, ids, lib=None) -> NDArray                       # ValueError for an id outside [0, V)
def rmsnorm(x, w, eps, lib=None) -> NDArray
def linear(x, W, b=None, lib=None) -> NDArray                        # one tl_matmul_f32, trans_b=1, beta 1 over a fresh bias fill
def rope(x, positions, inv_freq, attention_scaling=1.0, layout="half", rotary_dim=None, lib=None) -> NDArray
def attention(q, k, v, scale, q_offset=0, causal=True, window=None, sinks=None, lib=None) -> NDArray
def silu_mul(gate, up, lib=None) -> NDArray
def add(a, b, lib=None) -> NDArray
def argmax(x, lib=None) -> int                                       # ties to the lowest index, NaN skipped

@dataclass
class OpCheck: op: str; calls: int; ratio: float; ok: bool; detail: str
class OpCheckError(RuntimeError): report: list[OpCheck]
def check_ops(model, lib=None, seed=0, T=7) -> list[OpCheck]         # never raises for a wrong kernel

class CCache:  def seq_len(self, layer=0) -> int
class CBackend:
    def __init__(self, model, lib=None, check=False, seed=0)         # ValueError: unsupported; OpCheckError: failed check
    def new_cache(self) -> CCache
    def forward(self, ids, positions=None, cache=None) -> NDArray     # [B, T, V] or [T, V], float32
    def forward_varlen(self, seqs) -> list[NDArray]                   # one [T_i, V] per sequence, bitwise == forward(seq)
```

Your CLI (entry-point territory) gains `--backend {numpy,c}` and `--check` on `generate` and `logits`, and `bench decode --backend numpy,c`; `MS-L9` fixes their flags and final lines.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_linear_bias_and_rope` | unit, smoke | section 3: the bias GEMM twice, the bound, one RoPE pair, one RMSNorm row | you and the test agree on the definitions |
| `test_signatures_match_the_headers` | unit | every declared argtype has the header's width; void functions have no restype | `L10.1` declares the same functions in Rust |
| `test_op_check_passes_on_every_tiny_model` | differential | on five tiny checkpoints, every op's ratio is at most 1; matmul checks 8 weights | a correct library always loads |
| `test_op_check_names_a_wrong_kernel` | fault | an `add`, `rmsnorm`, or `silu_mul` off by $10^{-3}$ to $10^{-5}$ relative is the one failed row, and `check=True` refuses to load | `--check` points at the kernel to fix |
| `test_op_check_reports_a_stubbed_kernel` | fault | a `TlError` from a stubbed `L9.3` is a failed row with ratio inf and the error text | every missing kernel is listed at once |
| `test_tiny_hf_logits_golden` | golden, smoke | five tiny HF checkpoints (GQA, MQA, window, biases, Llama-3 and YaRN rope) give HF's logits within $10^{-4}$ of their range | the oracle `L10.1` is compared with is itself right |
| `test_logits_match_the_numpy_backend` | differential | C logits equal L7.9's on a batch of two | the design's bar for `--backend c` |
| `test_config_features_match_the_numpy_backend` | differential | learned sinks with a window, tied heads with partial interleaved rotary, biases with MQA, MHA with a large eps | features no committed checkpoint has |
| `test_greedy_matches_hf_tokens` | golden | 32 greedy tokens on every margin-filtered prompt equal HF's | `MS-L9`'s first step |
| `test_cached_decode_is_bitwise_the_full_forward` | property | chunks of 1, 4, 1, and the rest equal the whole sequence bitwise, on all five models | cached decode equals recompute |
| `test_varlen_batch_equals_per_sequence_calls` | property | a packed batch of 3, 12, 7 tokens equals three calls bitwise | continuous batching in `L10.2` |
| `test_batch_rows_equal_single_calls` | property | each row of a `[2, 12]` batch equals that row alone bitwise | batched requests stay independent |
| `test_rejects_unsupported_configs` | boundary | MLA, MoE, GeLU, and sink tokens are `ValueError`s naming the feature | no plausible wrong text |
| `test_rejects_bad_ids_and_empty_sequences` | boundary | id 256 of 256, -1, float ids, 3-D ids, and an empty varlen sequence are `ValueError`s | the kernel trusts its ids |
| `test_cache_belongs_to_one_batch_size` | boundary | a cache filled for a batch of 2 refuses a batch of 1 | sequences never mix |
| `test_argmax_ties_and_nan` | boundary | ties go to the lowest index, NaN is skipped, all NaN is -1 | greedy agrees across Python and Rust |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. filling the bias with `ascontiguousarray(broadcast_to(b, (N, M)))` | for one row it is `b` itself: decode corrupts the bias after the first token | `test_hand_example_linear_bias_and_rope`, `test_cached_decode_is_bitwise_the_full_forward` (mutant `s01`) |
| 2. $\beta = 1$ over a zero buffer, or $\beta = 0$ over the bias | the bias vanishes (Qwen2's logits drift) | `test_tiny_hf_logits_golden` (mutant `s02`) |
| 3. `q_offset = 0` after a cached prefix | the new query is treated as position 0: it sees only key 0 | `test_greedy_matches_hf_tokens` (mutant `s03`) |
| 4. not passing `config.sliding_window` to the kernel | Mistral attends past its window | `test_tiny_hf_logits_golden` (mutant `s04`) |
| 5. dropping a layer's learned sinks | attention rows sum to 1 when they should leave weight on the sink | `test_config_features_match_the_numpy_backend` (mutant `s05`) |
| 6. RoPE `attention_scaling` forced to 1 | YaRN models' logits scale wrong | `test_tiny_hf_logits_golden` (mutant `s06`) |
| 7. the layout flag always 0 | interleaved-rotary models rotate the wrong pairs | `test_config_features_match_the_numpy_backend` (mutant `s07`) |
| 8. the head dim as `d_rot` | a partial rotary model rotates everything | `test_config_features_match_the_numpy_backend` (mutant `s08`) |
| 9. $d^{-1/2}$ instead of $D^{-1/2}$ for the score scale | attention too flat | `test_tiny_hf_logits_golden` (mutant `s09`) |
| 10. a hard-coded $\epsilon$ | invisible on activations near 1, wrong when eps matters | `test_config_features_match_the_numpy_backend` (mutant `s10`) |
| 11. trusting ids in `embedding` | id $= V$ reads the row after the table | `test_rejects_bad_ids_and_empty_sequences` (mutant `s11`) |
| 12. packed positions $0 .. N - 1$ instead of per sequence | the second sequence of a varlen batch is rotated as if it continued the first | `test_varlen_batch_equals_per_sequence_calls` (mutant `s12`) |
| 13. a failed check that only prints | the backend serves with a broken kernel | `test_op_check_names_a_wrong_kernel` (mutant `s13`) |
| 14. one bound for both halves of a rotated pair | a correct RoPE kernel fails the check where $\lvert b c \rvert \gg \lvert b s \rvert$ (small angles) | `test_op_check_passes_on_every_tiny_model` (mutant `s14`) |
| 15. letting a `TlError` escape the check | `--check` crashes on the first stub | `test_op_check_reports_a_stubbed_kernel` (mutant `s15`) |
| 16. not advancing the cache length | the next chunk overwrites the last one | `test_cached_decode_is_bitwise_the_full_forward` (mutant `s16`) |
| 17. positions $s .. s + BT - 1$ across a batch | row 2 of a batch starts at position $T$ | `test_batch_rows_equal_single_calls` (mutant `s17`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L9.1` | `tl_matmul_f32` with `trans_b = 1` for every projection; its batch invariance makes the varlen and batch properties bitwise |
| Back | `L9.3` | `tl_flash_attn_fwd_f32` with `q_offset`, window, and sinks; its chunk invariance makes cached decode bitwise |
| Back | `L9.6` | RMSNorm, RoPE, SiLU-mul, embedding, add, argmax |
| Back | `rt.01` | `load()`, `Lib.declare`, `f32_ptr`, `TlError` |
| Back | `M09.3` | `gamma`, `matmul_error_bound`, `bound_ratio`: the op check's budgets |
| Back | `L7.9` | the model, its `state_dict` names, `config.rope_spec()` |
| Forward | `L10.1` | the Rust runner's logits are compared with `--backend c` on the same checkpoint; when they differ, the kernels are already cleared |
| Forward | `L10.2` | continuous batching relies on the invariances proven here |

If you skip this module, `MS-L9` cannot run: your CLI has no `--backend c`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `CBackend` op dispatch | PyTorch's dispatcher, ggml's backend scheduler | per-op device choice, fallbacks to another backend for unsupported ops, graph capture to skip Python per op | `ggml/src/ggml-backend.c` (`ggml_backend_sched`); `c10/core/DispatchKey.h` |
| load-time op check | cuDNN's algorithm finder, TensorRT's tactic selection at engine build | they time candidate kernels on the model's shapes and keep the fastest; correctness is assumed, not checked | the cuDNN `cudnnFindConvolutionForwardAlgorithm` documentation; TensorRT `IAlgorithmSelector` |
| bias in the GEMM ($\beta = 1$) | cuBLASLt epilogues | bias, activation, and residual fused into the GEMM's write-out | cuBLASLt `CUBLASLT_EPILOGUE_BIAS` |
| `forward_varlen` | FlashAttention's `varlen` API, vLLM's packed batches | `cu_seqlens` offsets in one kernel launch instead of a loop over sequences | `flash_attn_varlen_func` in `flash_attn/flash_attn_interface.py` |
| bitwise invariance tests | "Defeating Nondeterminism in LLM Inference" (Thinking Machines, 2025) | batch-invariant kernels for every reduction so served outputs never depend on load | the batch-invariant ops library linked from that post |
