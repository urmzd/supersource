<!-- ss:module M09.4 -->
# FP8 E4M3/E5M2, MXFP4/MXFP8 with E8M0 scales, and the C conversions

## Overview

| | |
|---|---|
| **Module** | `M09.4` · build · C and Python · Pass 6 · 4 to 6 h |
| **You build** | `python/tinyllm/num/lowp.py`: `fp8_max`, `f32_to_fp8_bits`, `fp8_bits_to_f32`, `fp8_scale`, `quantize_fp8`, `dequantize_fp8`, `f32_to_e2m1_bits`, `e2m1_bits_to_f32`, `e8m0_to_f32`, `e8m0_scale_code`, `mx_quantize`, `mx_dequantize` · `c/src/numerics/lowp.c`: `tl_f32_to_e4m3`, `tl_e4m3_to_f32`, `tl_f32_to_e5m2`, `tl_e5m2_to_f32`, `tl_f32_to_bf16`, `tl_bf16_to_f32`, `tl_f32_to_f16`, `tl_f16_to_f32` |
| **Contract** | [`course/contracts/py/tinyllm/num/lowp.pyi`](../../course/contracts/py/tinyllm/num/lowp.pyi) · C: [`tinyllm/numerics.h`](../../course/contracts/c/include/tinyllm/numerics.h) (the M09.4 section, with the edge rules) |
| **Tests** | `course/tests/M09.4/`: `test_lowp.py` (Python), `test_lowp.c` (C, ASan and UBSan), `test_lowp_c_vs_python.py` (C against Python through ctypes) · fixture `course/fixtures/M09.4/lowp_golden.npz` from ml_dtypes |
| **Needs** | `M09.1` (`f32_to_bf16_bits`, `round_to_fp16`: the Python twins of the C bf16 and f16 conversions) · reading: `S-M09a` rounding by hand, `M06.3` the first C and Python pair |
| **Used by** | `L8.5` fp8 and MX weight quantization · `L9.5` reads the f16 group scales of int4 weights · `L9.4` reads the f16 KV cache with `tl_f16_to_f32` · `craft.13` KV format v2 stores fp8 E4M3 |
| **Milestone** | `MS-P6` (Pass 6 gate: every math module of the pass checks green) |
| **Optional depth** | Micikevicius et al., "FP8 Formats for Deep Learning" (2022); Open Compute Project, *OCP 8-bit Floating Point Specification (OFP8)* v1.0 (2023) and *OCP Microscaling Formats (MX) Specification* v1.0 (2023); Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023) |

## Key Takeaways

- A small float format is three numbers: exponent bits, fraction bits $M$, and bias $B$. In the binade of $\lfloor\log_2\lvert v\rvert\rfloor$ (or the subnormal range) its values are multiples of one quantum $2^{q_e}$, so rounding is "divide by the quantum, round to the nearest even integer, multiply back", and the code is $((q_e + M + B - 1) \ll M) + n$ (`test_hand_example`, `test_encode_golden`).
- E4M3 trades range for precision (3 fraction bits, max 448, no infinity); E5M2 the opposite (2 fraction bits, max 57344, IEEE infinities). The course saturates finite overflow instead of producing NaN or infinity (`test_saturation_and_specials`).
- Ties go to the code with an even last bit; one float32 step either side of a tie must round away from it (`test_ties_go_to_even`, `fp8_ties_to_even`).
- MX formats share one power-of-two scale per block of 32: $X = 2^{\lfloor\log_2 a_{\max}\rfloor - e_{\max}}$ puts the block's largest value in the element format's top binade, so each element needs only 4 (FP4) or 8 (FP8) bits (`test_hand_example_mx`, `test_mx_golden`).
- The C bit-level conversion and the Python value-level one are independent implementations of the same rounding, and they agree on every code and every tested input (`test_c_fp8_encode_equals_python_and_golden`).

## How to work this chapter

```bash
ss start M09.4              # stubs lowp.py and lowp.c into your repo
ss tests M09.4              # read the test catalog first: rung R0, you write no tests here
ss check M09.4              # exit code is the verdict: Python, C (sanitized), and C vs Python
ss check M09.4 --ref-deps   # only if you skipped M09.1
ss diff  M09.4              # after passing: your code against the reference
```

---

## 1. Why now

The first time you serve SmolLM2 with a KV cache, its keys and values take 2 (K and V) x 30 layers x 3 kv heads x 64 x 4 bytes, 46 KB per token, so about 94 MB for one 2048-token sequence in float32 (`M05.1` counts it). Halving that to float16 (`L8.3`, `L9.4`) or quartering it to fp8 (`craft.13`) is the difference between serving four requests and sixteen, and quantizing weights to fp8 or MXFP4 (`L8.5`) cuts the bytes every decode step must read. All of that rests on one operation done identically in two languages: round a float32 to a format with a few bits, and read it back. If Python rounds 1.0625 down and C rounds it up, the Python-quantized checkpoint and the C kernel disagree in the last bit of every tie, and the parity tests of Part 9 fail with no obvious cause. This module writes both sides once, from the definition, and proves them against ml_dtypes and each other.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $E, M$ | exponent bits and fraction bits of a format | integers |
| $B$ | exponent bias | integer |
| $e_f$ | the stored exponent field, $0 \le e_f < 2^E$ | integer |
| $f$ | the stored fraction field, $0 \le f < 2^M$ | integer |
| $e_{\min} = 1 - B$ | exponent of the smallest normal | integer |
| $\lfloor\log_2 v\rfloor$ | the binade of $v > 0$: $2^{\lfloor\log_2 v\rfloor} \le v < 2^{\lfloor\log_2 v\rfloor + 1}$ | integer |
| $q_e = \max(\lfloor\log_2 v\rfloor, e_{\min}) - M$ | exponent of the quantum near $v$ | integer |
| $n = \mathrm{rne}(v / 2^{q_e})$ | the significand count: $v$ in quanta, rounded to nearest, ties to even | integer |
| $s$ | a per-tensor scale, $x \approx s \cdot \mathrm{fp8}(x / s)$ | float |
| $X = 2^{k}$ | an MX block scale, stored as E8M0 code $k + 127$ | power of two |
| $a_{\max}$ | the largest magnitude in a block | float |
| $e_{\max}$ | exponent of the element format's largest normal: 2 (FP4), 8 (E4M3) | integer |

### 2.1 The formats

| Format | $E$ | $M$ | $B$ | Max finite | Smallest subnormal | Specials |
|---|---|---|---|---|---|---|
| E4M3 (OCP E4M3FN) | 4 | 3 | 7 | $448 = 1.75 \cdot 2^8$ | $2^{-9}$ | `0x7F`, `0xFF` NaN; no infinity |
| E5M2 (OCP) | 5 | 2 | 15 | $57344 = 1.75 \cdot 2^{15}$ | $2^{-16}$ | `0x7C` $+\infty$, `0x7D` to `0x7F` NaN |
| FP4 E2M1 (OCP MX) | 2 | 1 | 1 | 6 | 0.5 | none: values 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| bf16 | 8 | 7 | 127 | $\approx 3.39 \cdot 10^{38}$ | $2^{-133}$ | IEEE |
| f16 (binary16) | 5 | 10 | 15 | 65504 | $2^{-24}$ | IEEE |
| E8M0 (MX scale) | 8 | 0 | 127 | $2^{127}$ | $2^{-127}$ | `0xFF` NaN; no zero, no sign |

A code is sign, exponent field, fraction field. Its value is $(-1)^{\text{sign}} (2^M + f)\, 2^{e_f - B - M}$ for $e_f \ge 1$ (the leading 1 is implicit) and $(-1)^{\text{sign}} f\, 2^{1 - B - M}$ for $e_f = 0$ (subnormal: no implicit 1, the smallest exponent). E4M3 gives up infinity to keep one more binade: only the all-ones code is NaN, so $1.75 \cdot 2^8$ is finite.

### 2.2 Decoding is exact

Every code's value is a small integer times a power of two, so it is exactly representable in float32 (`ldexpf` in C, `np.ldexp` in Python). Decoding all 256 fp8 codes and comparing bits with ml_dtypes is a complete test. Encoding a decoded value must give back its code: every finite code is a fixed point of encode after decode.

### 2.3 Rounding by quanta

Within the binade of $v$ the format's values are the multiples of $2^{\lfloor\log_2 v\rfloor - M}$; below $2^{e_{\min}}$ they are the multiples of $2^{e_{\min} - M}$ (subnormals continue the last normal binade's spacing). So with $q_e = \max(\lfloor\log_2 v\rfloor, e_{\min}) - M$:

$$n = \mathrm{rne}\!\left(\frac{v}{2^{q_e}}\right), \qquad \text{code} = ((q_e + M + B - 1) \ll M) + n.$$

Check the code formula: a normal value with $n = 2^M + f$ has exponent field $q_e + M + B$, and $((q_e + M + B - 1) \ll M) + 2^M + f = ((q_e + M + B) \ll M) + f$. In the subnormal range $q_e + M + B - 1 = 0$ and the code is $n$. When rounding carries ($n = 2^{M+1}$, the next binade's first value), the addition carries into the exponent field by itself. Ties to even: when $v / 2^{q_e}$ is exactly halfway, $n$ is the even neighbour, so the code ends in 0 (the shifted part is a multiple of $2^M$, even).

Python and C reach $n$ differently. In float64, $v / 2^{q_e}$ is an exact power-of-two scaling and `np.rint` rounds to nearest even, so `lowp.py` works on values. `lowp.c` works on bits: a float32 magnitude is $s \cdot 2^{e_2}$ with an integer $s < 2^{24}$, and $n$ is $s$ shifted right by $q_e - e_2$ bits, rounded up when the shifted-out part is more than half, or exactly half with $n$ odd. Two independent routes to one answer is what makes the differential test worth having.

### 2.4 Saturation and the specials

A finite value past the largest finite code saturates to it, with its sign: an outlier activation in a quantized model must come back as the largest representable value, not as NaN (E4M3 has no infinity, and the next code is NaN) or infinity (E5M2). The edge rules are fixed in `numerics.h`: E4M3 saturates infinities too; E5M2 keeps $\pm\infty$; NaN encodes as `0x7F`; bf16 NaN is `0x7FC0` and f16 NaN `0x7E00`; bf16 and f16 overflow to infinity (they are IEEE formats). The sign of zero is kept: $-0.0$ is `0x80`.

### 2.5 Scaled quantization

E4M3's dynamic range is about $2^{-9}$ to 448. A weight matrix with $\lvert w\rvert \le 0.05$ would use only the bottom of it, so per-tensor fp8 stores $\mathrm{fp8}(w / s)$ with $s = a_{\max} / 448$: the largest weight maps onto 448 and every normal value keeps 3 fraction bits, a relative error at most $2^{-4}$. The quotient $w / s$ is formed in float64 and rounded once. Rounding it to float32 first and then to fp8 is a double rounding: a value just above an fp8 tie can land exactly on the tie in float32 and then round the wrong way.

### 2.6 Microscaling: one scale per block

MX formats (OCP MX v1.0) share a power-of-two scale across a block of 32 consecutive values. For a block with largest magnitude $a_{\max}$,

$$X = 2^{\lfloor\log_2 a_{\max}\rfloor - e_{\max}}, \qquad \text{element}_i = \mathrm{round}_{\text{elem}}(x_i / X)$$

so $a_{\max} / X$ lies in $[2^{e_{\max}}, 2^{e_{\max}+1})$, the element format's top binade, and every element keeps the format's relative precision near the block's largest value. The scale is stored as E8M0: just the exponent, code $k + 127$, clamped to 0 to 254 (255 is NaN). Because the top binade of FP4 runs to 8 but its largest value is 6, values between 6 and 8 times $X$ saturate to $6X$: a known cost of MXFP4. Dividing by a power of two is exact, so the only rounding is the element's. An all-zero block takes the scale that $a_{\max} = 1$ would get.

### 2.7 bf16 and f16 in C

The C half also converts bf16 and f16, the KV cache formats of `rt.04` and `L9.4`. They use the same rounding routine with $(M, B) = (7, 127)$ and $(10, 15)$; their Python twins are `M09.1`'s `f32_to_bf16_bits` and `round_to_fp16`, and the differential test holds C to both.

## 3. Worked example by hand

**0.3 to E4M3.** $\lfloor\log_2 0.3\rfloor = -2$ ($0.25 \le 0.3 < 0.5$), above $e_{\min} = -6$, so $q_e = -2 - 3 = -5$: the quantum is $2^{-5} = 0.03125$. $0.3 / 0.03125 = 9.6$ rounds to $n = 10$, the value $10 \cdot 2^{-5} = 0.3125$, and the code is $((-5 + 3 + 7 - 1) \ll 3) + 10 = 32 + 10 = 42 =$ `0x2A`. Check by decoding: $e_f = 42 \gg 3 = 5$, $f = 2$, value $(8 + 2) \cdot 2^{5 - 7 - 3} = 0.3125$.

**0.3 to E5M2.** $q_e = -2 - 2 = -4$, $0.3 \cdot 16 = 4.8$ rounds to 5, value $5/16 = 0.3125$, code $((-4 + 2 + 15 - 1) \ll 2) + 5 = 48 + 5 = 53 =$ `0x35`.

**A tie.** $1.0625 = 1 + 2^{-4}$ in E4M3: $q_e = 0 - 3 = -3$, $1.0625 \cdot 8 = 8.5$, exactly halfway between 8 and 9; the even one is 8, so the value is 1.0 (`0x38`). $1.1875 = 9.5$ quanta rounds to the even 10: 1.25 (`0x3A`).

**An MX block** of four FP4 elements: $x = [0.25, -1.4, 3.1, 11.0]$. $a_{\max} = 11$, $\lfloor\log_2 11\rfloor = 3$, FP4's $e_{\max} = 2$, so $X = 2^1$ (E8M0 code 128). $x / X = [0.125, -0.7, 1.55, 5.5]$. In FP4 ($M = 1$, $B = 1$, $e_{\min} = 0$): $0.125$ has $q_e = -1$, $0.25$ quanta, rounds to 0; $-0.7$: $1.4$ quanta of $0.5$, rounds to 1, so $-0.5$ (code $1 + 8 = 9$); $1.55$: $q_e = -1$, $3.1$ rounds to 3, so 1.5 (code 3); $5.5$: $q_e = 1$, $2.75$ rounds to 3, so 6 (code 7). Dequantized: $[0, -1, 3, 12]$. The largest value came back as 12, above 11: within one quantum, as rounding allows.

These are the first cases in section 4: `test_hand_example`, `test_hand_example_mx`, and the C `hand_example`.

## 4. The interface

```python
# python/tinyllm/num/lowp.py
def f32_to_fp8_bits(x, fmt: Literal["e4m3", "e5m2"]) -> NDArray        # uint8, RNE, saturating
def fp8_bits_to_f32(codes, fmt) -> NDArray                              # float32, exact
def fp8_scale(amax: float, fmt) -> float                                # amax / fp8_max(fmt)
def quantize_fp8(x, fmt, scale: float) -> NDArray                       # codes of x / scale, one rounding
def dequantize_fp8(q, fmt, scale: float) -> NDArray
def f32_to_e2m1_bits(x) -> NDArray; def e2m1_bits_to_f32(codes) -> NDArray
def e8m0_to_f32(codes) -> NDArray; def e8m0_scale_code(amax: float, elem) -> int
def mx_quantize(x, block: int = 32, elem="fp4_e2m1") -> tuple[NDArray, NDArray]   # codes, e8m0 scales
def mx_dequantize(codes, scales_e8m0, block: int, elem) -> NDArray
```

```c
/* c/src/numerics/lowp.c, declared in tinyllm/numerics.h */
uint8_t  tl_f32_to_e4m3(float x);  float tl_e4m3_to_f32(uint8_t b);
uint8_t  tl_f32_to_e5m2(float x);  float tl_e5m2_to_f32(uint8_t b);
uint16_t tl_f32_to_bf16(float x);  float tl_bf16_to_f32(uint16_t b);
uint16_t tl_f32_to_f16(float x);   float tl_f16_to_f32(uint16_t b);
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3: 0.3 in both fp8 formats, the two ties | you and the test agree on the definitions |
| `test_hand_example_mx` | unit | section 3's MX block: codes, scale 128, dequantized values | the MX rule by hand |
| `test_decode_every_code_golden` | golden | all fp8, fp4, and e8m0 codes bitwise against ml_dtypes | the tables `L8.5` and `craft.13` read |
| `test_encode_golden` | golden | ties, near-ties, subnormals, random bits against ml_dtypes | round to nearest even everywhere |
| `test_saturation_and_specials` | boundary | saturation at 448 and 57344, infinities, NaN | outliers never become NaN |
| `test_tiny_values_and_signed_zero` | boundary | $2^{-10}$ is a tie to 0, sign kept, subnormals | small activations |
| `test_encode_inverts_decode` | property | every finite code is a fixed point | wrong bias or lost implicit bit |
| `test_rounding_is_nearest_and_monotone` | property | 20000 values: nearest code, never decreasing | the definition of rounding |
| `test_ties_go_to_even` | boundary | every midpoint gets an even code | bit parity with C |
| `test_scaled_quantization_roundtrip` | property | relative error at most $2^{-4}$, the max comes back | `L8.5`'s fp8 weights |
| `test_quantize_rounds_once` | boundary | $1.0625 + 2^{-30}$ goes up, not to the even tie | double rounding |
| `test_fp4_values_and_ties` | unit | the eight FP4 values, its ties, saturation at 6 | MXFP4 elements |
| `test_e8m0_scale_code` | unit | floor of the log, clamping, all-zero blocks | the shared scale |
| `test_mx_golden` | golden | 32-element blocks over 45 binades against ml_dtypes | MXFP4 and MXFP8 weights |
| `test_mx_block_law` | property | each element is the nearest FP4 value times $X$ | blocks never share or leak scales |
| `test_mx_shapes_and_errors` | boundary | block tiling, all-zero blocks, bad input | caller bugs raise |
| `test_c_fp8_decode_equals_python_on_every_code` | differential | C decode equals Python and ml_dtypes on 256 codes | one table in two languages |
| `test_c_fp8_encode_equals_python_and_golden` | differential | C encode equals Python on fixture, special, and 20000 random inputs | the KV v2 writer and reader agree |
| `test_c_bf16_f16_equal_m091_and_numpy` | differential | C bf16 and f16 against `M09.1` and numpy, every code | the f16 KV cache of `L9.4` |
| `hand_example` | unit | section 3 in C | the bit-level route reaches the same codes |
| `fp8_saturation_and_specials` | boundary | the edge rules of `numerics.h` in C | |
| `fp8_tiny_and_signed_zero` | boundary | ties at the bottom, float32 subnormal inputs | shifts of 24 bits and more |
| `fp8_every_code_is_a_fixed_point` | property | decode then encode, all 256 codes, both formats | |
| `fp8_ties_to_even` | boundary | every midpoint and its float32 neighbours | the sticky bits count |
| `bf16_hand_values` | unit | ties, overflow to infinity, NaN, subnormals | bf16 weights (`L7.9`) |
| `bf16_every_code_round_trips` | property | all 65536 codes | |
| `f16_hand_values` | unit | 65504, the tie at 65520, subnormal ties | f16 KV cache |
| `f16_every_code_round_trips` | property | all 65536 codes | |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. rounding half away from zero (`floor(x + 0.5)`, `rem >= half`) | every tie one code high; C and torch disagree | `test_ties_go_to_even` (mutant `s01`), `fp8_ties_to_even` (mutant `s21`) |
| 2. no subnormal floor on the quantum, or subnormals decoded with $2^{-B}$ | tiny values get precision the format does not have | `test_tiny_values_and_signed_zero` (mutants `s02`, `s05`), `fp8_tiny_and_signed_zero` (mutants `s20`, `s24`, `s28`) |
| 3. the exponent field off by one, or a lost implicit 1 | whole binades off by a factor of 2 | `test_encode_inverts_decode` (mutants `s03`, `s04`), `fp8_every_code_is_a_fixed_point` (mutant `s19`) |
| 4. no saturation: finite overflow to NaN or infinity | an outlier activation turns the layer into NaN | `test_saturation_and_specials` (mutants `s07`, `s08`), `fp8_saturation_and_specials` (mutants `s22`, `s23`) |
| 5. the MX scale from the ceiling of $\log_2$, or one scale per row | elements lose a binade, or blocks share a scale | `test_e8m0_scale_code` (mutant `s12`), `test_mx_golden` (mutant `s18`) |
| 6. the per-tensor scale inverted, or dequantize dividing | weights off by $s^2$ | `test_scaled_quantization_roundtrip` (mutants `s15`, `s16`) |
| 7. rounding $x / s$ to float32 before fp8 | a value just past a tie rounds back down | `test_quantize_rounds_once` (mutant `s17`) |
| 8. bf16 by truncating the low 16 bits | a bias toward zero that adds up over a model | `bf16_hand_values` (mutant `s25`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.1` | the bf16 and f16 Python conversions the C half is compared with; the anatomy of a float |
| Back | `S-M09a` | rounding to nearest even by hand |
| Forward | `L8.5` | `quantize_fp8` and `mx_quantize` for fp8 and MXFP4 weights; the error budget of `M09.3` |
| Forward | `L9.5` | the fused int4 matmul decodes its f16 group scales |
| Forward | `L9.4` | paged attention reads the f16 KV cache through `tl_f16_to_f32` |
| Forward | `craft.13` | KV format v2 stores E4M3 with per-(layer, head) scales |

If you skip this module, `ss check L8.5` stops with `L8.5 needs M09.4`; `--ref-deps` substitutes the reference.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `f32_to_fp8_bits` | ml_dtypes `float8_e4m3fn` | all fp8 variants (`e4m3fnuz`, `e4m3b11fnuz`) as numpy dtypes, non-saturating | `ml_dtypes/_src/float8.h` |
| `tl_f32_to_e4m3` | NVIDIA `cvt.rn.satfinite.e4m3x2.f32` | two conversions per instruction on Hopper, saturating like yours | PTX ISA, `cvt` |
| `mx_quantize` | Microsoft microxcaling | MX emulation in PyTorch with stochastic rounding and FP6 elements | `mx/mx_ops.py` |
| `quantize_fp8` | Transformer Engine fp8 recipes | delayed scaling: the scale comes from an amax history of past steps | `transformer_engine/common/recipe` |
