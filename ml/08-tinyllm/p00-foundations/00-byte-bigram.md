<!-- ss:module L0.0 -->
# Byte bigram: counts, logits through C, safetensors v0

## Overview

| | |
|---|---|
| **Module** | `L0.0` · build · Python · Pass 1 · 3 to 4 h |
| **You build** | `python/tinyllm/lm/bigram.py`: `BigramLM` (`fit_counts`, `logits`, `nll`, `sample`); `python/tinyllm/io/safetensors.py`: `save_safetensors`, `load_safetensors` (F32 only) |
| **Contract** | [`course/contracts/py/tinyllm/lm/bigram.pyi`](../../../course/contracts/py/tinyllm/lm/bigram.pyi) · [`course/contracts/py/tinyllm/io/safetensors.pyi`](../../../course/contracts/py/tinyllm/io/safetensors.pyi) · formats: [`safetensors.md`](../../../course/contracts/formats/safetensors.md), [`tokenizer.md`](../../../course/contracts/formats/tokenizer.md) · CLI verbs: [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md) |
| **Tests** | `course/tests/L0.0/` (what they check: section 4) |
| **Needs** | `M03.1` naive matmul in C (`tl_matmul_f32`) · `rt.01` the ctypes loader (`load`, `f32_ptr`) · reading: `lang.01` Python and numpy (or `--ref-deps`) |
| **Used by** | `L10.0` your Rust engine serves this checkpoint · later: `L0.5` retrains it with autograd, `L0.6` adds every dtype |
| **Milestone** | `MS-P1` (the tracer: every layer is yours and runs end to end) |
| **Optional depth** | Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft), ch. 3 "N-gram Language Models" |

## Key Takeaways

- A bigram language model is one table: row $i$ is the distribution of the next byte after byte $i$, and add-alpha smoothing makes every row a proper distribution with no zeros (`test_rows_sum_to_one`).
- The negative log-likelihood of the count model is a closed form in the counts, and it is the number every later model in this course must beat (`test_hand_example_nll`, `test_nll_matches_counts`).
- Multiplying a one-hot row by the table picks a row exactly, so your C matmul returns the model's logits bit for bit (`test_logits_are_weight_rows`).
- Sampling is temperature, softmax, one uniform draw, and an inverse CDF; a seed makes it repeatable (`test_sample_is_seeded`, `test_sample_frequencies_match_model`).
- Your safetensors file is byte-identical to the reference library's, so your Rust engine (and anyone else's reader) loads it (`test_matches_library_bytes`).

## How to work this chapter

```bash
ss start L0.0              # stubs bigram.py and safetensors.py into your repo
ss tests L0.0              # read the test catalog first: rung R0, you write no tests here
ss check L0.0              # exit code is the verdict
ss check L0.0 --ref-deps   # only if your M03.1 or rt.01 is not passing yet
ss diff  L0.0              # after passing: your code against the reference
```

You also write the first verbs of your own CLI, `python -m tinyllm` (it is yours: the course ships no CLI). [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md) fixes what `MS-P1` will run:

| Verb | What it does with this module |
|---|---|
| `train bigram --data F --out DIR [--alpha A]` | reads the bytes of `F` as ids, calls `fit_counts`, writes `DIR/model.safetensors` (`bigram.weight`, metadata `{"format": "tinyllm"}`) and `DIR/config.json`, prints `{"out", "tokens", "nll"}` as the last line |
| `generate --model DIR --prompt P [--max-tokens N] [--greedy \| --temperature T] [--seed S]` | `load_safetensors`, `BigramLM(weight)`, `sample(list(P.encode()), N, T, S)`, prints `{"ids", "text"}` |
| `logits --model DIR --prompt P [--prefix-ids ...]` | `logits(ids)[-1]` as a JSON list |

`config.json` for the tracer is exactly `{"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": 256, "tl_format": 1}`.

---

## 1. Why now

After `M03.1` and `rt.01` your system has a C matmul that Python can call through ctypes, and nothing to call it with. The Rust engine you write next (`L10.0`) needs a model directory to serve, the Go gateway (`gw.00`) needs that engine behind it, and the cluster (`dep.00`) needs both. Today `{tinyllm} train bigram` does not exist and there is no checkpoint on disk, so every layer above Python has nothing to load. This module builds the smallest real language model, routes its forward pass through your own C kernel, and writes it in the file format every later model uses. From here on, the tracer bullet has a payload.

## 2. Principles

This section is "just enough" math, defined from scratch. Each idea comes back properly later: functions, `log`, and `exp` in `M00.1`; probability and random variables in `M07.1`; maximum likelihood in `M07.2`; entropy, cross-entropy, and perplexity in `M11.1`; stable softmax in `M09.2`.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size: the number of distinct token ids (256 for bytes) | `int` |
| $x_1, \dots, x_T$ | the training text as token ids, $x_t \in \{0, \dots, V-1\}$ | `int64[T]` |
| $C_{ij}$ | count of positions $t$ with $x_t = i$ and $x_{t+1} = j$ | `float64[V, V]` |
| $R_i = \sum_j C_{ij}$ | how often $i$ appears as a context (followed by something) | `float64[V]` |
| $\alpha > 0$ | smoothing constant (add-one: $\alpha = 1$) | `float` |
| $P(j \mid i)$ | the model's probability that $j$ follows $i$ | `float64` in $(0, 1)$ |
| $W_{ij} = \ln P(j \mid i)$ | the weight table, stored as logits | `float32[V, V]` |
| $\mathrm{softmax}(z)_j = e^{z_j} / \sum_k e^{z_k}$ | turns any real row $z$ into probabilities | `float64[V]` |
| $\mathrm{NLL}$ | mean negative log-likelihood, nats per predicted token | `float` |
| $\tau \ge 0$ | sampling temperature | `float` |
| $F_j = \sum_{k \le j} p_k$ | cumulative distribution (CDF) of a row $p$ | `float64[V]` |
| $u$ | one uniform random number in $[0, 1)$ | `float` |

**Bytes are tokens.** Text is stored as UTF-8 bytes, and each byte is a number from 0 to 255. The tracer uses the byte itself as the token id (decision D32, [`tokenizer.md`](../../../course/contracts/formats/tokenizer.md)), so $V = 256$, there is no tokenizer file, and `"hi"` is the ids `[104, 105]`. A character such as `é` takes two bytes, so it is two tokens.

**A language model predicts the next token.** Given the text so far, it returns a probability for each of the $V$ possible next tokens. The **bigram** assumption is the bluntest one that works: only the previous token matters, $P(x_{t+1} \mid x_1, \dots, x_t) \approx P(x_{t+1} \mid x_t)$. So the whole model is a $V \times V$ table: row $i$ is the distribution of what follows $i$.

**Counts to probabilities.** Slide over the text and count each adjacent pair $(x_t, x_{t+1})$ into $C_{x_t, x_{t+1}}$. Row index = where you are, column index = what comes next. The natural estimate is the observed frequency, $P(j \mid i) = C_{ij} / R_i$. It is the maximum-likelihood estimate (`M07.2` proves it): no other table gives the training text a higher probability.

**Zeros break it, so smooth.** A pair that never occurs gets probability 0, and a byte that never occurs as a context has $R_i = 0$, which makes its whole row $0/0$. Add-alpha smoothing pretends every pair was seen $\alpha$ extra times:

$$P(j \mid i) = \frac{C_{ij} + \alpha}{R_i + \alpha V}.$$

Each row still sums to one, because $\sum_j (C_{ij} + \alpha) = R_i + \alpha V$: the denominator must add $\alpha$ once for each of the $V$ columns, not once. An unseen context gets the uniform row $\alpha / (\alpha V) = 1/V$.

**Store logs, not probabilities.** The natural logarithm $\ln$ turns products into sums, $\ln(ab) = \ln a + \ln b$, and $\exp$ undoes it, $e^{\ln p} = p$. The model stores $W_{ij} = \ln P(j \mid i)$, called **logits**. Every later model in the course also outputs logits, and turns them into probabilities with softmax. For a row of log-probabilities, softmax gives the probabilities back exactly: $e^{\ln p_j} / \sum_k e^{\ln p_k} = p_j / 1$. Softmax also ignores a constant added to every entry, $\mathrm{softmax}(z + c) = \mathrm{softmax}(z)$, which is why you compute it as $e^{z_j - \max z}$: the largest exponent becomes $e^0 = 1$ and nothing overflows.

**Negative log-likelihood measures fit.** Under the bigram assumption, the probability of the whole text (after its first token) is the product $\prod_{t=1}^{T-1} P(x_{t+1} \mid x_t)$. That number underflows for any real text, so take its log and average:

$$\mathrm{NLL} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \ln P(x_{t+1} \mid x_t).$$

There are $T - 1$ predictions, because the first token has no context. The unit is **nats per token** (natural log). Lower is better; a model that knows nothing (uniform over 256 bytes) scores $\ln 256 \approx 5.545$. For the count model the sum regroups by pair: $\mathrm{NLL} = -\frac{1}{T-1} \sum_{i,j} C_{ij} \ln P(j \mid i)$, a closed form your autograd bigram in `L0.5` must reach by gradient descent.

**The one-hot matmul.** A one-hot vector $e_i$ has a 1 at position $i$ and 0 elsewhere. Multiplying it by the table picks row $i$: $(e_i W)_j = \sum_k [k = i] W_{kj} = W_{ij}$. Stack the one-hot rows of $T$ ids into a $T \times V$ matrix $E$, and $E W$ is the $T \times V$ matrix of logits, one row per position. Every other term in each sum is an exact zero, so the product equals the rows bit for bit. It costs $T \cdot V \cdot V$ multiply-adds where a row lookup would cost nothing; you pay it on purpose, because `logits = activations @ weight` is the shape of every model to come, and it sends the forward pass through your C kernel (`tl_matmul_f32`) via the ctypes loader. `L10.0` does the same in Rust.

**Temperature.** To sample, divide the logits by $\tau$ before softmax: $p = \mathrm{softmax}(z / \tau)$. With log-probabilities $z = \ln q$ this gives $p_j \propto q_j^{1/\tau}$. At $\tau = 1$ you sample the model as is; $\tau < 1$ sharpens it (at $\tau = 0.5$, $p \propto q^2$); $\tau > 1$ flattens it. $\tau = 0$ is defined as **greedy**: take the largest logit, and on a tie the lowest id (decision D11, so every implementation agrees).

**Sampling by inverse CDF.** To draw $j$ with probability $p_j$, build the running sum $F_j = p_0 + \dots + p_j$ (so $F_{V-1} = 1$), draw one uniform $u \in [0, 1)$, and return the smallest $j$ with $F_j > u$. The interval of $u$ values that lands on $j$ is $[F_{j-1}, F_j)$, whose length is exactly $p_j$. One draw per token, from a generator you created once with a seed, makes the same seed give the same text. The tracer may use `numpy.random.default_rng(seed)`; `L0.5` switches to your own PCG32 (`M06.3`) so Python and Rust agree.

**The file.** A safetensors file is an 8-byte little-endian length $N$, then $N$ bytes of JSON describing each tensor (dtype, shape, byte offsets), padded with spaces to a multiple of 8, then the raw tensor bytes back to back, little-endian and row-major. The full rules, including the order a canonical writer uses so its bytes match the reference library, are in [`safetensors.md`](../../../course/contracts/formats/safetensors.md). The tracer checkpoint holds one tensor, `bigram.weight`, F32 `[256, 256]`.

## 3. Worked example by hand

Take a three-letter alphabet so the table fits on paper: `a = 0`, `b = 1`, `c = 2`, so $V = 3$. The text is `abbacab`, ids `[0, 1, 1, 0, 2, 0, 1]`, $T = 7$.

**Pairs.** Six adjacent pairs: `ab`, `bb`, `ba`, `ac`, `ca`, `ab`.

**Counts** (row = current, column = next):

| | a | b | c | $R_i$ |
|---|---|---|---|---|
| a | 0 | 2 | 1 | 3 |
| b | 1 | 1 | 0 | 2 |
| c | 1 | 0 | 0 | 1 |

**Add one** ($\alpha = 1$, so each row total grows by $\alpha V = 3$) and divide:

| | a | b | c |
|---|---|---|---|
| a | 1/6 = 0.1667 | 3/6 = 0.5 | 2/6 = 0.3333 |
| b | 2/5 = 0.4 | 2/5 = 0.4 | 1/5 = 0.2 |
| c | 2/4 = 0.5 | 1/4 = 0.25 | 1/4 = 0.25 |

Every row sums to 1. The weight table is the natural log of each entry, for example $W_{ab} = \ln 0.5 = -0.6931$ and $W_{aa} = \ln(1/6) = -1.7918$.

**NLL.** The six predictions and their costs $-\ln P$:

| $t$ | context $\to$ next | $P$ | $-\ln P$ |
|---|---|---|---|
| 1 | a $\to$ b | 3/6 | $\ln 2 = 0.6931$ |
| 2 | b $\to$ b | 2/5 | $\ln 2.5 = 0.9163$ |
| 3 | b $\to$ a | 2/5 | $\ln 2.5 = 0.9163$ |
| 4 | a $\to$ c | 2/6 | $\ln 3 = 1.0986$ |
| 5 | c $\to$ a | 2/4 | $\ln 2 = 0.6931$ |
| 6 | a $\to$ b | 3/6 | $\ln 2 = 0.6931$ |

Sum: $3 \ln 2 + 2 \ln 2.5 + \ln 3 = 2.0794 + 1.8326 + 1.0986 = 5.0106$. Divided by 6: $\mathrm{NLL} = 0.8351$ nats per token, better than the uniform model's $\ln 3 = 1.0986$.

**Logits by one-hot matmul.** For ids `[0, 2]`, $E = \begin{bmatrix}1&0&0\\0&0&1\end{bmatrix}$ and $E W$ is rows `a` and `c` of $W$, unchanged.

**Greedy.** From `a` the largest entry is `b` (0.5). From `b`, `a` and `b` tie at 0.4, and ties go to the lowest id, so `a`. Then `b`, then `a`: greedy from `[a]` for 4 tokens is `[1, 0, 1, 0]`.

**Temperature 0.5 on row `a`.** $q^2 = (1/36, 9/36, 4/36)$, which renormalizes to $(1/14, 9/14, 4/14)$: `b` goes from 50% to 64%.

**Inverse CDF on row `a`.** $F = (1/6, 4/6, 1) = (0.1667, 0.6667, 1)$. A draw $u = 0.10$ gives `a` (the first $F_j > 0.10$), $u = 0.50$ gives `b`, $u = 0.90$ gives `c`.

**The file.** The format page works out a 120-byte file by hand: one tensor `w = [[1, 2], [3, 4]]` with metadata `{"format": "tinyllm"}`. Its header is 93 bytes of JSON, padded with 3 spaces to $N = 96$; the file is $8 + 96 + 16 = 120$ bytes, ending in `00 00 80 3f 00 00 00 40 00 00 40 40 00 00 80 40` (1.0, 2.0, 3.0, 4.0 as little-endian float32).

These numbers are the first cases in section 4: `test_hand_example_probabilities`, `test_hand_example_nll`, `test_greedy_ties_go_to_lowest_id`, and `test_worked_example_bytes`.

## 4. The interface

```python
# python/tinyllm/lm/bigram.py
class BigramLM:
    weight: NDArray                                   # float32 [V, V], log P(j | i)
    def __init__(self, weight: ArrayLike | None = None) -> None
    @property
    def vocab_size(self) -> int
    def fit_counts(self, ids: ArrayLike, vocab_size: int, alpha: float = 1.0) -> None
    def logits(self, ids: ArrayLike) -> NDArray       # [T, V] = onehot(ids) @ W via tl_matmul_f32
    def nll(self, ids: ArrayLike) -> float
    def sample(self, prefix: list[int], n: int, temperature: float, seed: int) -> list[int]

# python/tinyllm/io/safetensors.py (v0: F32 only)
def save_safetensors(path: str, tensors: dict[str, NDArray], meta: dict[str, str]) -> None
def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]
```

The contracts carry the exact rules: which errors are `ValueError`, that `sample` returns only the new ids, that an empty `meta` omits `__metadata__`. Call the C matmul through `rt.01`: `load().tl_matmul_f32(f32_ptr(A), f32_ptr(B), f32_ptr(C), M, N, K, K, N, N, 1.0, 0.0, 0, None)`, with $M = T$, $N = K = V$. The loader raises `TlError` on a bad status, so you do not check it yourself.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_probabilities` | unit | the section 3 table, from `weight` and from `logits` | you and the test agree on the definition |
| `test_hand_example_nll` | unit | $\mathrm{NLL} = 0.8351$ on `abbacab` | the number `train bigram` prints |
| `test_counts_direction` | boundary | text `ab`: $P(b \mid a) = 2/3$, row `b` uniform | a transposed table predicts the past |
| `test_rows_sum_to_one` | property | every row of a 256-byte model sums to 1, unseen bytes included | `L10.0` samples from any row |
| `test_nll_matches_counts` | differential | NLL equals the closed form recomputed by loops | the target `L0.5` must reach |
| `test_nll_normalizes_logits` | unit | rows `[0, 0]` and `[5, 5]` give $\ln 2$ | `L0.5`'s trained logits are not normalized |
| `test_alpha_must_be_positive` | boundary | $\alpha \le 0$ is a `ValueError` | no `-inf` or `nan` in a checkpoint |
| `test_rejects_out_of_range_ids` | boundary | ids `-1` and `V` are a `ValueError` | numpy would wrap `-1` silently |
| `test_logits_are_weight_rows` | differential | `logits(ids) == weight[ids]` bit for bit, float32 `[T, V]` | the forward pass `L10.0` reproduces |
| `test_logits_go_through_libtinyllm` | boundary | with `TINYLLM_LIB` pointing nowhere, `logits` fails | proves your C kernel is on the path |
| `test_greedy_ties_go_to_lowest_id` | unit | greedy from `a` is `[1, 0, 1, 0]` | the tie rule every engine shares (D11) |
| `test_temperature_zero_is_greedy` | unit | $\tau = 0$ ignores the seed | `generate --greedy` |
| `test_sample_returns_only_new_ids` | unit | `n` new ids, no prefix; `n = 0` is `[]` | `generate` prints generated ids only |
| `test_sample_is_seeded` | property | same seed, same ids; seeds differ | milestones replay a run |
| `test_sample_frequencies_match_model` | statistical | 6000 transitions fit the table (chi-square, $p > 10^{-3}$) | the sampler draws the model, not something near it |
| `test_temperature_sharpens` | statistical | at $\tau = 0.5$ transitions fit $q^2$ renormalized | temperature divides |
| `test_sample_rejects_bad_args` | boundary | empty prefix, $\tau < 0$, $n < 0$ are `ValueError` | bytes have no start token |
| `test_worked_example_bytes` | unit | the 120-byte file of section 3 | header length, padding, byte order |
| `test_matches_library_bytes` | golden | six files byte-identical to the pinned `safetensors` library | any reader, including your engine, loads your checkpoint |
| `test_bytes_are_little_endian_row_major` | boundary | a transposed view and a `>f4` array write row-major little-endian bytes | arrays in memory are not always in file order |
| `test_save_rejects_non_f32` | boundary | float64, int32, float16, a `__metadata__` tensor, a non-string value | v0 writes F32 only |
| `test_load_reads_library_files` | golden | every library file loads to the exact inputs | `generate` loads the checkpoint |
| `test_roundtrip` | property | `load(save(x)) == x`, writable float32 | checkpoints survive a round trip |
| `test_load_rejects_bad_files` | boundary | gap, trailing bytes, wrong size, short file, duplicate key, F16 | a reader never trusts the header |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. no smoothing ($\alpha = 0$) | `-inf` logits for unseen pairs; a row of `nan` for a byte never seen as a context; `0 * -inf = nan` inside the matmul | `test_alpha_must_be_positive` (mutant `m01`) |
| 2. counting `C[next, cur]` | the model predicts the previous byte; the hand table comes out transposed | `test_counts_direction` (mutant `s02`) |
| 3. trusting ids | `-1` silently counts into the last row (numpy wraps negative indexes) | `test_rejects_out_of_range_ids` (mutant `s03`) |
| 4. reading logits as log-probabilities in `nll` | correct for the count model, wrong the day `L0.5` stores trained logits | `test_nll_normalizes_logits` (mutant `s10`) |
| 5. a numpy fallback in `logits` | everything passes while your C build is broken, until the Rust engine meets it | `test_logits_go_through_libtinyllm` (mutant `s04`) |
| 6. seeding a new generator at every step | still repeatable, but every step draws the same $u$, so the text follows one quantile of each row | `test_sample_frequencies_match_model` (mutant `s09`) |
| 7. writing the array's memory as is | a transposed view writes columns; a big-endian array writes swapped bytes | `test_bytes_are_little_endian_row_major` (mutants `s15`, `s16`) |
| 8. a reader that trusts the header | reads past a gap or ignores trailing bytes, so a corrupt checkpoint loads | `test_load_rejects_bad_files` (mutants `s20`, `s21`) |
| 9. smoothing denominator $R_i + \alpha$ | rows sum to more than 1 | `test_rows_sum_to_one` (mutant `s01`) |
| 10. multiplying by the temperature | $\tau = 0.5$ flattens instead of sharpening | `test_temperature_sharpens` (mutant `s05`) |
| 11. `np.argmax` on a reversed row, or `>=` in a manual loop | greedy ties go to the highest id, and engines disagree | `test_greedy_ties_go_to_lowest_id` (mutant `s07`) |
| 12. scoring the context token in `nll` | an NLL that does not depend on what follows | `test_hand_example_nll` (mutant `s06`) |
| 13. insertion order, zero padding, padding when already aligned, `{}` for empty metadata, `\u` escapes | bytes differ from the library's, so a strict reader or a hash check rejects the file | `test_matches_library_bytes` (mutants `s12`, `s13`, `s17`, `s18`), `test_worked_example_bytes` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | `tl_matmul_f32` computes `onehot(ids) @ weight` |
| Back | `rt.01` | `load()` finds `libtinyllm` and checks its ABI version; `f32_ptr` passes arrays; `TlError` reports a failed call |
| Back | `lang.01` | numpy arrays, dtypes, `np.add.at`, and broadcasting (the vectorized bigram count) |
| Forward | `L10.0` | your Rust engine reads `bigram.weight` from this file (`tl_arch = bigram`, `tl_tokenizer = bytes`) and computes the same one-hot logits through `tl_matmul_f32` |
| Forward | `L0.5` | takes over `bigram.py`: trains the same table with your autograd until it reaches the count model's NLL within $10^{-3}$, and samples with your PCG32 |
| Forward | `L0.6` | takes over `safetensors.py`: every dtype, atomic checkpoints, the token-stream reader |

If you skip this module, `ss check L10.0` stops with `L10.0 needs L0.0`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| add-alpha bigram | KenLM n-gram models | 5-gram contexts, modified Kneser-Ney smoothing, backoff, compact tries | KenLM `lm/builder/` |
| one-hot matmul for logits | embedding lookup | a row gather instead of a matmul: `ggml_get_rows` in llama.cpp, `torch.nn.Embedding` | `ggml/src/ggml.c` |
| `sample` | vLLM and SGLang samplers | top-k, top-p, min-p, penalties, batched on GPU with per-request seeds | vLLM `v1/sample/sampler.py` |
| `save_safetensors` | Hugging Face `safetensors` | every dtype, memory-mapped zero-copy loading, the Rust core your engine mirrors | `safetensors/src/tensor.rs` |
