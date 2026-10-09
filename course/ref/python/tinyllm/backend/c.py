"""tinyllm.backend.c (L9.7): the Llama forward, one op at a time, in libtinyllm.

L7.9 built the forward out of numpy ops on autograd Tensors. This backend
keeps the same parameters (state_dict, Hugging Face names) and sends every
op to a C kernel through rt.01's ctypes loader: embedding, RMSNorm, the
projections, RoPE, attention, SiLU-mul, the residual adds, and greedy
argmax. Before the first forward it can run a load-time op check: each op
once on the model's own shapes, against numpy float64, divided by M09.3's
error bound, so a kernel that is wrong for THIS model is reported by name
before it produces a single token.

Contract: contracts/py/tinyllm/backend/c.pyi.
"""

from __future__ import annotations

import ctypes
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.ffi.libtinyllm import TlError, f32_ptr, load
from tinyllm.ffi.libtinyllm import signatures as v0_signatures
from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM
from tinyllm.num.tolerance import bound_ratio, gamma, matmul_error_bound, unit_roundoff

OPS = (
    "embedding",
    "rmsnorm",
    "matmul",
    "rope",
    "attention",
    "silu_mul",
    "add",
    "argmax",
)

_DECLARED: "weakref.WeakSet[Any]" = weakref.WeakSet()


def signatures() -> dict[str, tuple[Any, list[Any]]]:
    # SOLUTION-BEGIN L9.7
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64 = ctypes.c_int64
    vp = ctypes.c_void_p
    return {
        "tl_rmsnorm_f32": (None, [f32p, f32p, f32p, i64, i64, ctypes.c_float]),
        "tl_rope_f32": (
            None,
            [f32p, i32p, i64, i64, i64, i64, f32p, ctypes.c_float, ctypes.c_int],
        ),
        "tl_silu_mul_f32": (None, [f32p, f32p, f32p, i64]),
        "tl_embedding_f32": (None, [f32p, i32p, f32p, i64, i64]),
        "tl_add_f32": (None, [f32p, f32p, f32p, i64]),
        "tl_argmax_f32": (ctypes.c_int32, [f32p, i64]),
        "tl_flash_attn_fwd_f32": (
            # rt.01's STATUS restype marker, read off the loader's v0 table
            # (tl_matmul_f32 returns tl_status), so the call raises TlError.
            v0_signatures()["tl_matmul_f32"][0],
            [
                f32p,
                f32p,
                f32p,
                f32p,
                f32p,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                ctypes.c_float,
                i64,
                ctypes.c_int,
                i64,
                f32p,
                i64,
                i64,
                vp,
                vp,
            ],
        ),
    }
    # SOLUTION-END


def _ready(lib: Any) -> Any:
    """The Lib to call, with every kernel after v0 declared on it once."""
    # SOLUTION-BEGIN L9.7
    lib = load() if lib is None else lib
    if lib not in _DECLARED:
        for name, (restype, argtypes) in signatures().items():
            lib.declare(name, restype, argtypes)
        _DECLARED.add(lib)
    return lib
    # SOLUTION-END


def _f32(a: ArrayLike) -> NDArray:
    # SOLUTION-BEGIN L9.7
    return np.ascontiguousarray(a, dtype=np.float32)
    # SOLUTION-END


def _i32(a: NDArray) -> Any:
    # SOLUTION-BEGIN L9.7
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    # SOLUTION-END


# -- the ops --------------------------------------------------------------------------


def embedding(table: ArrayLike, ids: ArrayLike, lib: Any = None) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    t = _f32(table)
    i = np.ascontiguousarray(np.asarray(ids).reshape(-1), dtype=np.int64)
    V, d = t.shape
    # The kernel trusts its ids (elementwise.h): an id past the table would
    # read someone else's memory, so the check happens here, in Python.
    if i.size and (i.min() < 0 or i.max() >= V):
        raise ValueError(f"ids must be in [0, {V}), got min {i.min()} max {i.max()}")
    i32 = i.astype(np.int32)
    out = np.empty((i32.size, d), np.float32)
    lib.tl_embedding_f32(f32_ptr(t), _i32(i32), f32_ptr(out), i32.size, d)
    return out
    # SOLUTION-END


def rmsnorm(x: ArrayLike, w: ArrayLike, eps: float, lib: Any = None) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    a, g = _f32(x), _f32(w)
    d = a.shape[-1]
    if g.shape != (d,):
        raise ValueError(f"w must be [{d}], got {g.shape}")
    y = np.empty_like(a)
    lib.tl_rmsnorm_f32(
        f32_ptr(a), f32_ptr(g), f32_ptr(y), a.size // d if d else 0, d, eps
    )
    return y
    # SOLUTION-END


def linear(
    x: ArrayLike, W: ArrayLike, b: Optional[ArrayLike] = None, lib: Any = None
) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    a, w = _f32(x), _f32(W)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
        raise ValueError(f"need x [N, K] and W [M, K], got {a.shape} and {w.shape}")
    N, K = a.shape
    M = w.shape[0]
    if b is None:
        out, beta = np.empty((N, M), np.float32), 0.0  # beta 0: never read
    else:
        # The bias rows ride in C: C = x W^T + 1 * C, one call. A fresh
        # buffer: ascontiguousarray(broadcast_to(b, (1, M))) is b itself, and
        # the kernel would write the product into the model's bias.
        out, beta = np.empty((N, M), np.float32), 1.0
        out[:] = _f32(b)
    lib.tl_matmul_f32(
        f32_ptr(a), f32_ptr(w), f32_ptr(out), N, M, K, K, K, M, 1.0, beta, 1, None
    )
    return out
    # SOLUTION-END


def rope(
    x: ArrayLike,
    positions: ArrayLike,
    inv_freq: ArrayLike,
    attention_scaling: float = 1.0,
    layout: str = "half",
    rotary_dim: Optional[int] = None,
    lib: Any = None,
) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    if layout not in ("half", "interleaved"):
        raise ValueError(f"layout must be 'half' or 'interleaved', got {layout!r}")
    y = np.array(x, dtype=np.float32, order="C", copy=True)  # the kernel works in place
    if y.ndim != 3:
        raise ValueError(f"x must be [T, H, D], got {y.shape}")
    T, H, D = y.shape
    r = D if rotary_dim is None else int(rotary_dim)
    f = _f32(inv_freq).reshape(-1)
    if r % 2 or not 0 < r <= D or f.size != r // 2:
        raise ValueError(
            f"rotary_dim {r} must be even, <= {D}, and match {f.size} frequencies"
        )
    pos = np.ascontiguousarray(np.asarray(positions).reshape(-1), dtype=np.int32)
    if pos.size != T:
        raise ValueError(f"positions must be [{T}], got {pos.size}")
    lib.tl_rope_f32(
        f32_ptr(y),
        _i32(pos),
        T,
        H,
        D,
        r,
        f32_ptr(f),
        attention_scaling,
        0 if layout == "half" else 1,
    )
    return y
    # SOLUTION-END


# fmt: off
def attention(
    q: ArrayLike, k: ArrayLike, v: ArrayLike, scale: float, q_offset: int = 0, causal: bool = True,
    window: Optional[int] = None, sinks: Optional[ArrayLike] = None, lib: Any = None,
) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    qq, kk, vv = _f32(q), _f32(k), _f32(v)
    single = qq.ndim == 3
    if single:
        qq, kk, vv = qq[None], kk[None], vv[None]
    if qq.ndim != 4 or kk.shape != vv.shape or kk.ndim != 4:
        raise ValueError(
            f"need q [B, H, Tq, D] and k, v [B, Hkv, Tk, D], got {qq.shape}, {kk.shape}, {vv.shape}"
        )
    B, H, Tq, D = qq.shape
    Hkv, Tk = kk.shape[1], kk.shape[2]
    o = np.empty_like(qq)
    s = None if sinks is None else _f32(sinks)
    lib.tl_flash_attn_fwd_f32(
        f32_ptr(qq),
        f32_ptr(kk),
        f32_ptr(vv),
        f32_ptr(o),
        None,
        B,
        H,
        Hkv,
        Tq,
        Tk,
        D,
        scale,
        int(q_offset),
        1 if causal else 0,
        0 if window is None else int(window),
        f32_ptr(s),
        0,
        0,
        None,
        None,
    )
    return o[0] if single else o
    # SOLUTION-END


# fmt: on


def silu_mul(gate: ArrayLike, up: ArrayLike, lib: Any = None) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    g, u = _f32(gate), _f32(up)
    if g.shape != u.shape:
        raise ValueError(f"gate {g.shape} and up {u.shape} differ")
    y = np.empty_like(g)
    lib.tl_silu_mul_f32(f32_ptr(g), f32_ptr(u), f32_ptr(y), g.size)
    return y
    # SOLUTION-END


def add(a: ArrayLike, b: ArrayLike, lib: Any = None) -> NDArray:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    x, z = _f32(a), _f32(b)
    if x.shape != z.shape:
        raise ValueError(f"a {x.shape} and b {z.shape} differ")
    y = np.empty_like(x)
    lib.tl_add_f32(f32_ptr(x), f32_ptr(z), f32_ptr(y), x.size)
    return y
    # SOLUTION-END


def argmax(x: ArrayLike, lib: Any = None) -> int:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    a = _f32(x).reshape(-1)
    return int(lib.tl_argmax_f32(f32_ptr(a), a.size))
    # SOLUTION-END


# -- the load-time op check ------------------------------------------------------------


@dataclass
class OpCheck:
    op: str
    calls: int
    ratio: float
    ok: bool
    detail: str


class OpCheckError(RuntimeError):
    def __init__(self, report: list[OpCheck]) -> None:
        # SOLUTION-BEGIN L9.7
        self.report = report
        bad = [f"{r.op} (ratio {r.ratio:.3g}: {r.detail})" for r in report if not r.ok]
        super().__init__("the C op check failed: " + "; ".join(bad))
        # SOLUTION-END


def _probe(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> NDArray:
    """Deterministic inputs in [-scale, scale): a Weyl sequence of the flat
    index (no RNG, so the check is the same on every machine)."""
    # SOLUTION-BEGIN L9.7
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    frac = np.modf(i * 0.6180339887498949 + 0.1234 * (seed + 1))[0]
    return ((2.0 * frac - 1.0) * scale).reshape(shape).astype(np.float32)
    # SOLUTION-END


def _attn_f64(q, k, v, scale, q_offset, window, sinks):
    """float64 reference attention and its bound terms, for one batch row.
    q [H, Tq, D], k, v [Hkv, Tk, D]."""
    # SOLUTION-BEGIN L9.7
    H, Tq, D = q.shape
    Hkv, Tk, _ = k.shape
    rep = H // Hkv
    u = unit_roundoff("f32")
    o = np.zeros((H, Tq, D))
    bound = np.zeros((H, Tq, D))
    for h in range(H):
        kh, vh = k[h // rep].astype(np.float64), v[h // rep].astype(np.float64)
        for i in range(Tq):
            p = q_offset + i
            j = np.arange(Tk)
            vis = j <= p
            if window:
                vis &= j > p - window
            qi = q[h, i].astype(np.float64)
            s = (kh @ qi) * scale
            sv = s[vis]
            extra = [] if sinks is None else [float(sinks[h])]
            allv = np.concatenate([sv, extra])
            if allv.size == 0:
                continue
            m = allv.max()
            e = np.exp(allv - m)
            P = e / e.sum()
            pv = P[: sv.size]
            o[h, i] = pv @ vh[vis]
            dots = np.abs(kh[vis]) @ np.abs(qi) * scale if sv.size else np.zeros(1)
            E = gamma(D + 2, "f32") * float(dots.max()) + u * float(
                allv.max() - allv.min()
            )
            g = gamma(3 * Tk + 16, "f32")
            bound[h, i] = (2.0 * E + g) * (pv @ np.abs(vh[vis]))
    return o, bound
    # SOLUTION-END


def check_ops(
    model: LlamaForCausalLM, lib: Any = None, seed: int = 0, T: int = 7
) -> list[OpCheck]:
    # SOLUTION-BEGIN L9.7
    lib = _ready(lib)
    cfg = model.config
    w = {k: _f32(v) for k, v in model.state_dict().items()}
    d, H, Hkv, D = (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    V = cfg.vocab_size
    u = unit_roundoff("f32")
    spec = cfg.rope_spec()
    out: list[OpCheck] = []

    def run(op: str, fn) -> None:
        try:
            calls, ratio, detail = fn()
        except TlError as e:  # a stubbed or failing kernel
            out.append(OpCheck(op, 0, float("inf"), False, str(e)))
            return
        out.append(OpCheck(op, calls, float(ratio), bool(ratio <= 1.0), detail))

    def c_embedding():
        ids = np.array([(i * 7919 + seed) % V for i in range(T)])
        ids[0], ids[-1] = 0, V - 1
        table = w["model.embed_tokens.weight"]
        got = embedding(table, ids, lib)
        return (
            1,
            bound_ratio(got, table[ids].astype(np.float64), np.zeros(got.shape)),
            f"{T} ids of [{V}, {d}]",
        )

    def c_rmsnorm():
        x = _probe((T, d), seed, 2.0)
        g = w["model.layers.0.input_layernorm.weight"]
        x64 = x.astype(np.float64)
        y = (
            x64
            / np.sqrt((x64 * x64).mean(axis=-1, keepdims=True) + cfg.rms_norm_eps)
            * g
        )
        got = rmsnorm(x, g, cfg.rms_norm_eps, lib)
        return 1, bound_ratio(got, y, gamma(d + 8, "f32") * np.abs(y)), f"[{T}, {d}]"

    def c_matmul():
        names = [
            k
            for k in w
            if k.startswith("model.layers.0.") and k.endswith("_proj.weight")
        ]
        names.append(
            "lm_head.weight" if "lm_head.weight" in w else "model.embed_tokens.weight"
        )
        worst, which = 0.0, ""
        for n in names:
            W = w[n]
            x = _probe((T, W.shape[1]), seed + len(n))
            got = linear(x, W, None, lib)
            r = bound_ratio(
                got,
                x.astype(np.float64) @ W.astype(np.float64).T,
                matmul_error_bound(x, W.T, "f32"),
            )
            if r >= worst:
                worst, which = (
                    r,
                    f"{n.split('.')[-2]} [{T}, {W.shape[1]}] x [{W.shape[1]}, {W.shape[0]}]",
                )
        return len(names), worst, which

    def c_rope():
        x = _probe((T, H, D), seed, 1.5)
        top = cfg.max_position_embeddings - 1
        pos = np.array(
            [0, 1, top, top // 2, 3, top - 1, 17][:T] + [5] * max(0, T - 7)
        ) % (top + 1)
        got = rope(
            x,
            pos,
            spec.inv_freq,
            spec.attention_scaling,
            spec.layout,
            spec.rotary_dim,
            lib,
        )
        r = spec.rotary_dim
        h = r // 2
        # The angle as elementwise.h defines it: pos * inv_freq formed in
        # double and rounded once to float (what HF's float32 angle is too).
        ang = (
            pos[:, None].astype(np.float64)
            * _f32(spec.inv_freq).astype(np.float64)[None, :]
        )
        ang = ang.astype(np.float32).astype(np.float64)
        c = (np.cos(ang) * spec.attention_scaling)[:, None, :]
        s = (np.sin(ang) * spec.attention_scaling)[:, None, :]
        x64 = x.astype(np.float64)
        if spec.layout == "half":
            ia, ib = np.arange(h), np.arange(h) + h
        else:
            ia, ib = 2 * np.arange(h), 2 * np.arange(h) + 1
        a, b = x64[..., ia], x64[..., ib]
        want = x64.copy()
        want[..., ia], want[..., ib] = a * c - b * s, a * s + b * c
        bnd = np.zeros_like(want)
        # Each output is a two-term dot product with rounded cos and sin: its
        # own |terms| (a c - b s and a s + b c have different ones).
        g4 = gamma(4, "f32")
        bnd[..., ia] = g4 * (np.abs(a * c) + np.abs(b * s))
        bnd[..., ib] = g4 * (np.abs(a * s) + np.abs(b * c))
        return 1, bound_ratio(got, want, bnd), f"[{T}, {H}, {D}], positions up to {top}"

    def c_attention():
        Tk = T + 5
        q = _probe((H, T, D), seed, 1.0)
        k = _probe((Hkv, Tk, D), seed + 1, 1.0)
        vv = _probe((Hkv, Tk, D), seed + 2, 1.0)
        sinks = w.get("model.layers.0.self_attn.sinks")
        scale = D**-0.5
        got = attention(q, k, vv, scale, Tk - T, True, cfg.sliding_window, sinks, lib)
        want, bnd = _attn_f64(q, k, vv, scale, Tk - T, cfg.sliding_window, sinks)
        return (
            1,
            bound_ratio(got, want, bnd),
            f"q [{H}, {T}, {D}], k [{Hkv}, {Tk}, {D}]",
        )

    def c_silu_mul():
        g = _probe((T, cfg.intermediate_size), seed, 8.0)
        up = _probe((T, cfg.intermediate_size), seed + 3, 2.0)
        g64 = g.astype(np.float64)
        y = g64 / (1.0 + np.exp(-g64)) * up.astype(np.float64)
        got = silu_mul(g, up, lib)
        return (
            1,
            bound_ratio(got, y, gamma(12, "f32") * np.abs(y)),
            f"[{T}, {cfg.intermediate_size}]",
        )

    def c_add():
        a, b = _probe((T, d), seed, 4.0), _probe((T, d), seed + 4, 4.0)
        y = a.astype(np.float64) + b.astype(np.float64)
        return 1, bound_ratio(add(a, b, lib), y, u * np.abs(y)), f"[{T}, {d}]"

    def c_argmax():
        x = _probe((V,), seed, 1.0)
        x[V // 3] = x[V // 2] = 2.0  # a tie: the lower index wins
        x[0] = np.nan  # skipped
        got = argmax(x, lib)
        want = int(np.nanargmax(x))
        return (
            1,
            0.0 if got == want else float("inf"),
            f"[{V}] with a tie at {want} and a NaN",
        )

    run("embedding", c_embedding)
    run("rmsnorm", c_rmsnorm)
    run("matmul", c_matmul)
    run("rope", c_rope)
    run("attention", c_attention)
    run("silu_mul", c_silu_mul)
    run("add", c_add)
    run("argmax", c_argmax)
    return out
    # SOLUTION-END


# -- the backend ------------------------------------------------------------------------------


class CCache:
    def __init__(self, n_layers: int) -> None:
        # SOLUTION-BEGIN L9.7
        self._k: list[Optional[NDArray]] = [None] * n_layers
        self._v: list[Optional[NDArray]] = [None] * n_layers
        self._n = [0] * n_layers
        # SOLUTION-END

    def seq_len(self, layer: int = 0) -> int:
        # SOLUTION-BEGIN L9.7
        return self._n[layer]
        # SOLUTION-END

    def _append(self, layer: int, k: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        """Append [B, Hkv, T, D] keys and values; return every cached one,
        C-contiguous (the attention kernel reads [B, Hkv, Tk, D] packed)."""
        # SOLUTION-BEGIN L9.7
        B, Hkv, T, D = k.shape
        n = self._n[layer]
        buf = self._k[layer]
        if buf is not None and buf.shape[0] != B:
            raise ValueError(f"this cache holds batch {buf.shape[0]}, got {B}")
        if buf is None or n + T > buf.shape[2]:
            cap = max(16, 2 * (n + T))
            nk, nv = (
                np.zeros((B, Hkv, cap, D), np.float32),
                np.zeros((B, Hkv, cap, D), np.float32),
            )
            if buf is not None:
                nk[:, :, :n], nv[:, :, :n] = buf[:, :, :n], self._v[layer][:, :, :n]
            self._k[layer], self._v[layer] = nk, nv
        self._k[layer][:, :, n : n + T] = k
        self._v[layer][:, :, n : n + T] = v
        self._n[layer] = n + T
        return (
            np.ascontiguousarray(self._k[layer][:, :, : n + T]),
            np.ascontiguousarray(self._v[layer][:, :, : n + T]),
        )
        # SOLUTION-END


def _unsupported(cfg: LlamaConfig) -> list[str]:
    # SOLUTION-BEGIN L9.7
    bad = []
    if cfg.attention != "gqa":
        bad.append(
            f"attention {cfg.attention!r} (latent attention has q/k and v of different widths)"
        )
    if cfg.num_experts:
        bad.append(f"num_experts {cfg.num_experts} (no routed-expert op in libtinyllm)")
    if cfg.hidden_act != "silu":
        bad.append(f"hidden_act {cfg.hidden_act!r} (the C MLP op is SiLU-mul)")
    if cfg.sink_tokens:
        bad.append(
            f"sink_tokens {cfg.sink_tokens} (tl_flash_attn_fwd_f32 windows every key alike)"
        )
    return bad
    # SOLUTION-END


class CBackend:
    def __init__(
        self,
        model: LlamaForCausalLM,
        lib: Any = None,
        check: bool = False,
        seed: int = 0,
    ) -> None:
        # SOLUTION-BEGIN L9.7
        cfg = model.config
        bad = _unsupported(cfg)
        if bad:
            raise ValueError("the C backend does not support " + "; ".join(bad))
        self.config = cfg
        self.lib = _ready(lib)
        self.w = {k: _f32(v) for k, v in model.state_dict().items()}
        self.head = self.w.get("lm_head.weight", self.w["model.embed_tokens.weight"])
        spec = cfg.rope_spec()
        self.inv_freq = _f32(spec.inv_freq)
        self.mscale = float(spec.attention_scaling)
        self.layout = spec.layout
        self.rot = int(spec.rotary_dim)
        self.scale = cfg.head_dim**-0.5
        self.report = None
        if check:
            self.report = check_ops(model, self.lib, seed)
            if not all(r.ok for r in self.report):
                raise OpCheckError(self.report)
        # SOLUTION-END

    def new_cache(self) -> CCache:
        # SOLUTION-BEGIN L9.7
        return CCache(self.config.num_hidden_layers)
        # SOLUTION-END

    def _proj(self, x: NDArray, name: str) -> NDArray:
        # SOLUTION-BEGIN L9.7
        return linear(x, self.w[name + ".weight"], self.w.get(name + ".bias"), self.lib)
        # SOLUTION-END

    def _qkv(
        self, i: int, h: NDArray, pos: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Norm, projections, and rope for the N rows of h: q [N, H, D], k, v [N, Hkv, D]."""
        # SOLUTION-BEGIN L9.7
        c = self.config
        p = f"model.layers.{i}."
        x = rmsnorm(h, self.w[p + "input_layernorm.weight"], c.rms_norm_eps, self.lib)
        N, H, Hkv, D = (
            h.shape[0],
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )
        q = self._proj(x, p + "self_attn.q_proj").reshape(N, H, D)
        k = self._proj(x, p + "self_attn.k_proj").reshape(N, Hkv, D)
        v = self._proj(x, p + "self_attn.v_proj").reshape(N, Hkv, D)
        q = rope(q, pos, self.inv_freq, self.mscale, self.layout, self.rot, self.lib)
        k = rope(k, pos, self.inv_freq, self.mscale, self.layout, self.rot, self.lib)
        return q, k, v
        # SOLUTION-END

    def _attend(
        self, i: int, q: NDArray, k: NDArray, v: NDArray, q_offset: int
    ) -> NDArray:
        """q [B, H, T, D], k, v [B, Hkv, Tk, D] -> [B, H, T, D]."""
        # SOLUTION-BEGIN L9.7
        return attention(
            q,
            k,
            v,
            self.scale,
            q_offset,
            True,
            self.config.sliding_window,
            self.w.get(f"model.layers.{i}.self_attn.sinks"),
            self.lib,
        )
        # SOLUTION-END

    def _finish(self, i: int, h: NDArray, o: NDArray) -> NDArray:
        """o_proj, the residual, and the MLP half of layer i over N rows."""
        # SOLUTION-BEGIN L9.7
        p = f"model.layers.{i}."
        h = add(h, self._proj(o, p + "self_attn.o_proj"), self.lib)
        x = rmsnorm(
            h,
            self.w[p + "post_attention_layernorm.weight"],
            self.config.rms_norm_eps,
            self.lib,
        )
        a = silu_mul(
            self._proj(x, p + "mlp.gate_proj"),
            self._proj(x, p + "mlp.up_proj"),
            self.lib,
        )
        return add(h, self._proj(a, p + "mlp.down_proj"), self.lib)
        # SOLUTION-END

    def _logits(self, h: NDArray) -> NDArray:
        # SOLUTION-BEGIN L9.7
        x = rmsnorm(h, self.w["model.norm.weight"], self.config.rms_norm_eps, self.lib)
        return linear(x, self.head, None, self.lib)
        # SOLUTION-END

    def _ids(self, ids: ArrayLike) -> NDArray:
        # SOLUTION-BEGIN L9.7
        a = np.asarray(ids)
        if a.ndim not in (1, 2) or a.dtype.kind not in "iu":
            raise ValueError(
                f"ids must be integer [B, T] or [T], got {a.dtype} {a.shape}"
            )
        if a.size and (a.min() < 0 or a.max() >= self.config.vocab_size):
            raise ValueError(
                f"ids must be in [0, {self.config.vocab_size}), got min {a.min()} max {a.max()}"
            )
        return a
        # SOLUTION-END

    def forward(
        self,
        ids: ArrayLike,
        positions: Optional[ArrayLike] = None,
        cache: Optional[CCache] = None,
    ) -> NDArray:
        # SOLUTION-BEGIN L9.7
        a = self._ids(ids)
        single = a.ndim == 1
        a = a[None] if single else a
        B, T = a.shape
        c = self.config
        H, Hkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        s = cache.seq_len(0) if cache is not None else 0
        pos = np.arange(s, s + T) if positions is None else np.asarray(positions)
        pos = np.broadcast_to(pos, (B, T)).reshape(-1)  # one position per row of h
        h = embedding(self.w["model.embed_tokens.weight"], a, self.lib)  # [B T, d]
        for i in range(c.num_hidden_layers):
            q, k, v = self._qkv(i, h, pos)
            qh = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            kh = np.ascontiguousarray(k.reshape(B, T, Hkv, D).transpose(0, 2, 1, 3))
            vh = np.ascontiguousarray(v.reshape(B, T, Hkv, D).transpose(0, 2, 1, 3))
            if cache is not None:
                kh, vh = cache._append(i, kh, vh)
            o = self._attend(i, qh, kh, vh, s)
            h = self._finish(i, h, o.transpose(0, 2, 1, 3).reshape(B * T, H * D))
        logits = self._logits(h).reshape(B, T, c.vocab_size)
        return logits[0] if single else logits
        # SOLUTION-END

    def forward_varlen(self, seqs: Sequence[Sequence[int]]) -> list[NDArray]:
        # SOLUTION-BEGIN L9.7
        rows = [self._ids(np.asarray(sq, dtype=np.int64).reshape(-1)) for sq in seqs]
        if any(r.size == 0 for r in rows):
            raise ValueError("forward_varlen: every sequence needs at least one token")
        c = self.config
        H, Hkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        lens = [r.size for r in rows]
        offs = np.concatenate([[0], np.cumsum(lens)])
        pos = np.concatenate([np.arange(n) for n in lens])  # each sequence starts at 0
        h = embedding(
            self.w["model.embed_tokens.weight"], np.concatenate(rows), self.lib
        )
        for i in range(c.num_hidden_layers):
            q, k, v = self._qkv(i, h, pos)  # one call per op over all N rows
            o = np.empty((h.shape[0], H * D), np.float32)
            for j, n in enumerate(lens):
                a, b = offs[j], offs[j + 1]
                oj = self._attend(
                    i,
                    q[a:b].transpose(1, 0, 2)[None],
                    k[a:b].transpose(1, 0, 2)[None],
                    v[a:b].transpose(1, 0, 2)[None],
                    0,
                )
                o[a:b] = oj[0].transpose(1, 0, 2).reshape(n, H * D)
            h = self._finish(i, h, o)
        logits = self._logits(h)
        return [logits[offs[j] : offs[j + 1]] for j in range(len(lens))]
        # SOLUTION-END
