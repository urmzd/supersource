# contracts/py/tinyllm/backend/c.pyi (L9.7): the Python C backend, with a load-time op check
# chapter: ml/08-tinyllm/p09-kernels/10-python-c-backend.md
#
# L7.9's Llama forward, one op at a time, in libtinyllm. The model's
# parameters stay the ones LlamaForCausalLM loaded (state_dict, HF names);
# every op of the forward is one call through rt.01's ctypes loader:
#
#   op          C function (header)                          module
#   embedding   tl_embedding_f32 (tinyllm/elementwise.h)     L9.6
#   rmsnorm     tl_rmsnorm_f32 (tinyllm/elementwise.h)       L9.6
#   matmul      tl_matmul_f32 (tinyllm/matmul.h), trans_b=1  L9.1 (M03.1 v0)
#   rope        tl_rope_f32 (tinyllm/elementwise.h)          L9.6
#   attention   tl_flash_attn_fwd_f32 (tinyllm/attention.h)  L9.3
#   silu_mul    tl_silu_mul_f32 (tinyllm/elementwise.h)      L9.6
#   add         tl_add_f32 (tinyllm/elementwise.h)           L9.6
#   argmax      tl_argmax_f32 (tinyllm/elementwise.h)        L9.6
#
# A Linear with a bias is ONE matmul call: the output is first filled with
# the bias rows and tl_matmul_f32 runs with beta = 1 (C = x W^T + 1 * C).
# Without a bias, beta = 0, so the output buffer is never read.
#
# Shapes: activations are [N, d] float32 rows (N = B * T tokens, or the
# packed tokens of forward_varlen); q is [N, H, D] for rope and
# [B, H, T, D] for attention; k and v [B, Hkv, T, D]. Attention runs causal
# with q_offset = the cache length, window = config.sliding_window (None is
# 0, no window), sink_logits = the layer's learned sinks (None without), and
# scale = head_dim^-0.5. RoPE takes config.rope_spec(): inv_freq,
# attention_scaling, layout ("half" is 0, "interleaved" 1), rotary_dim.
#
# Supported configs: GQA, MHA, and MQA attention; q/k/v and MLP biases;
# sliding windows; learned sinks; tied or untied heads; every rope_scaling
# kind; partial rotary; both rope layouts. Anything else (attention "mla",
# num_experts > 0, hidden_act other than "silu", sink_tokens > 0) is a
# ValueError naming the feature, raised before any C call.
#
# The op check (`{tinyllm} ... --backend c --check`): before the first
# forward, run every op of OPS once on the loaded model's shapes, with
# deterministic inputs (no RNG: a fixed formula of the index and `seed`)
# and, where an op has weights, the model's own layer 0 and head weights;
# compare each result with the same op in numpy float64 and divide the
# error by its M09.3 bound (tinyllm.num.tolerance.bound_ratio):
#   embedding, argmax   exact: bound 0 (ratio 0 when equal, inf otherwise)
#   add                 u |a + b|                     (one rounding)
#   matmul              matmul_error_bound(x, W^T, "f32") for every weight
#                       of layer 0 and the head (gamma_K |x| |W|^T)
#   rmsnorm             gamma(d + 8) |y|
#   silu_mul            gamma(12) |y|
#   rope                gamma(4) (|a cos| + |b sin|) for a' = a cos - b sin,
#                       gamma(4) (|a sin| + |b cos|) for b' = a sin + b cos
#                       (cos and sin, in float64, of the angle
#                       elementwise.h defines: pos * inv_freq formed in
#                       double and rounded once to float; times
#                       attention_scaling), 0 for the pass-through
#                       dimensions; positions reach max_position_embeddings - 1
#   attention           (2 E + gamma(3 Tk + 16)) (P |V|) per output, with
#                       E = gamma(D + 2) scale max_j |q| . |k_j|
#                           + u (max_j s_j - min_j s_j)
#                       over the visible scores s of the row
# u = unit_roundoff("f32") = 2^-24. ratio <= 1 is ok. A C function that
# raises TlError (a stubbed unit) is a failed op with ratio inf.
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM

OPS: tuple[str, ...]  # ("embedding", "rmsnorm", "matmul", "rope", "attention", "silu_mul", "add", "argmax")

def signatures() -> dict[str, tuple[Any, list[Any]]]:
    """(restype, argtypes) of the kernels after v0 that the backend calls, in
    the form Lib.declare takes (rt.01): tl_rmsnorm_f32, tl_rope_f32,
    tl_silu_mul_f32, tl_embedding_f32, tl_add_f32 (restype None: void),
    tl_argmax_f32 (c_int32), and tl_flash_attn_fwd_f32 (STATUS), with the
    headers' widths: int64_t is c_int64, int is c_int, int32_t * is
    POINTER(c_int32), float * is POINTER(c_float), tl_arena * and tl_pool *
    are c_void_p. Every op below declares them on its Lib once."""

# One function per op: numpy in, a new float32 array out (rope too: it
# copies x first), the work done by one C call. `lib` is a Lib from
# tinyllm.ffi.libtinyllm (None: load()). Inputs are converted to
# C-contiguous float32 (int32 for ids and positions).

def embedding(table: ArrayLike, ids: ArrayLike, lib: Any = None) -> NDArray:
    """table [V, d], ids any shape -> [ids.size, d]. ValueError for an id
    outside [0, V), checked before the C call (the kernel trusts its ids)."""

def rmsnorm(x: ArrayLike, w: ArrayLike, eps: float, lib: Any = None) -> NDArray:
    """x [..., d], w [d] -> the shape of x."""

def linear(x: ArrayLike, W: ArrayLike, b: Optional[ArrayLike] = None, lib: Any = None) -> NDArray:
    """x [N, K] @ W[M, K]^T (+ b[M]) -> [N, M], one tl_matmul_f32 call with
    trans_b = 1 (the weight is used as stored) and beta = 1 over a bias-filled
    output when b is given, else beta = 0. ValueError for mismatched K."""

def rope(
    x: ArrayLike,
    positions: ArrayLike,
    inv_freq: ArrayLike,
    attention_scaling: float = 1.0,
    layout: str = "half",
    rotary_dim: Optional[int] = None,
    lib: Any = None,
) -> NDArray:
    """x [T, H, D], positions int [T] -> the rotated copy (rotary_dim None is
    D; ValueError unless it is even, <= D, and len(inv_freq) ==
    rotary_dim / 2, or for a layout other than "half" or "interleaved")."""

def attention(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    scale: float,
    q_offset: int = 0,
    causal: bool = True,
    window: Optional[int] = None,
    sinks: Optional[ArrayLike] = None,
    lib: Any = None,
) -> NDArray:
    """q [B, H, Tq, D] (or [H, Tq, D]), k, v [B, Hkv, Tk, D] (or [Hkv, Tk, D])
    -> o, the shape of q. Query i is at absolute position q_offset + i."""

def silu_mul(gate: ArrayLike, up: ArrayLike, lib: Any = None) -> NDArray:
    """silu(gate) * up, elementwise, same shape."""

def add(a: ArrayLike, b: ArrayLike, lib: Any = None) -> NDArray:
    """a + b, elementwise, same shape."""

def argmax(x: ArrayLike, lib: Any = None) -> int:
    """The index of the largest entry of 1-D x; ties go to the lowest index;
    NaN entries are skipped; -1 when every entry is NaN."""

@dataclass
class OpCheck:
    op: str  # one of OPS
    calls: int  # C calls compared (matmul: one per weight)
    ratio: float  # the largest bound_ratio over those calls (0 is exact)
    ok: bool  # ratio <= 1
    detail: str  # the worst call ("q_proj [7, 32] x [32, 32]"), or the TlError text

class OpCheckError(RuntimeError):
    """A loaded backend's op check failed. str() names every failed op with
    its ratio."""

    report: list[OpCheck]

    def __init__(self, report: list[OpCheck]) -> None: ...

def check_ops(model: LlamaForCausalLM, lib: Any = None, seed: int = 0, T: int = 7) -> list[OpCheck]:
    """The op check above, one OpCheck per op in OPS order, T tokens of
    activations. Never raises for a wrong kernel: CBackend(check=True) turns
    a failed row into OpCheckError."""

class CCache:
    """The backend's KV cache: per layer, float32 keys and values
    [B, Hkv, capacity, D] that grow by doubling. Made by
    CBackend.new_cache()."""

    def seq_len(self, layer: int = 0) -> int:
        """Tokens cached for that layer (0 before the first forward)."""

class CBackend:
    config: LlamaConfig
    report: Optional[list[OpCheck]]  # check_ops(model) when check=True, else None

    def __init__(self, model: LlamaForCausalLM, lib: Any = None, check: bool = False, seed: int = 0) -> None:
        """Take the model's parameters (state_dict, float32, C-contiguous)
        and declare the kernels on lib (None: load()). ValueError for an
        unsupported config (above). With check, run check_ops(model, lib,
        seed) and raise OpCheckError when any op fails."""

    def new_cache(self) -> CCache: ...
    def forward(self, ids: ArrayLike, positions: Optional[ArrayLike] = None, cache: Optional[CCache] = None) -> NDArray:
        """LlamaForCausalLM.forward through the C ops: ids int [B, T] (or [T])
        -> float32 logits [B, T, vocab] (or [T, vocab]). positions [T] or
        [B, T] default to s .. s + T - 1, s = cache.seq_len(0) (0 without a
        cache); they feed rope only. The cache appends this chunk's keys and
        values for every layer and attention reads all of them (q_offset =
        s). ValueError for ids that are not 1-D or 2-D integers, an id outside
        [0, vocab), or a cache made for another batch size.

        With L9.1's batch-invariant matmul and L9.3's chunk-invariant
        attention, row t of the logits is bitwise the same whether the
        sequence is run whole, in chunks through a cache, or as one row of a
        batch."""

    def forward_varlen(self, seqs: Sequence[Sequence[int]]) -> list[NDArray]:
        """Several sequences of different lengths in one pass, without a
        cache: their tokens are packed into one [N, d] activation matrix, so
        every projection, norm, and MLP is one call over all N rows; rope
        uses each token's position in its own sequence (0 .. T_i - 1), and
        attention runs once per sequence on its own rows. Returns one
        float32 [T_i, vocab] logits array per sequence, bitwise equal to
        forward(seqs[i]) (batch invariance). ValueError for an empty
        sequence."""
