//! tl-sys (L10.1): the full hand-written binding to libtinyllm.
//!
//! Three layers, from the bottom:
//!
//! 1. Raw declarations: one `extern "C"` item per function of
//!    contracts/c/include/tinyllm.h that the engine calls, with the exact C
//!    types (`int64_t` is `i64`, `int` is `c_int`, `tl_status` and
//!    `tl_dtype` are `i32`, never Rust enums: c/ABI.md rule 5), and
//!    `#[repr(C)]` mirrors of the public structs whose sizes and offsets
//!    c/ABI.md fixes.
//! 2. Safe functions: each checks every slice length and dimension before a
//!    pointer crosses into C (C trusts what it is given), and turns a
//!    non-zero status into a [`TlError`] that carries `tl_last_error()`.
//! 3. RAII owners: [`KvPool`] owns one `tl_kv_pool`; its `Drop` calls
//!    `tl_kv_pool_destroy` exactly once, so a pool cannot leak or be freed
//!    twice from safe code.
//!
//! L10.0 wrote the v0 of this file (the ABI check, the error slot, and
//! `tl_matmul_f32`); every v0 item keeps its signature, so the tracer engine
//! still builds against this version.

use std::ffi::{c_void, CStr};
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

/// The ABI version this binding was written against (`TL_ABI_VERSION`).
pub const ABI_VERSION: u32 = 1;

/// `tl_status` values (`tinyllm/abi.h`). A status is an `i32`, never a Rust
/// enum: C may hand back a value this crate has never heard of.
pub const TL_OK: i32 = 0;
pub const TL_EINVAL: i32 = 1;
pub const TL_ENOMEM: i32 = 2;
pub const TL_ESHAPE: i32 = 3;
pub const TL_EDTYPE: i32 = 4;
pub const TL_EFULL: i32 = 5;
pub const TL_ENOTFOUND: i32 = 6;
pub const TL_EFORMAT: i32 = 7;
pub const TL_EBUSY: i32 = 8;
pub const TL_EUNSUPPORTED: i32 = 9;
pub const TL_EIO: i32 = 10;

/// `tl_dtype` values (`tinyllm/abi.h`).
pub const TL_F32: i32 = 0;
pub const TL_F16: i32 = 1;
pub const TL_BF16: i32 = 2;
pub const TL_F8_E4M3: i32 = 3;
pub const TL_F8_E5M2: i32 = 4;
pub const TL_I8: i32 = 5;
pub const TL_U8: i32 = 6;
pub const TL_I32: i32 = 7;
pub const TL_Q4_G: i32 = 8;

/// KV block format v1: f16 payloads (formats/kv-block.md, D13).
pub const TL_KV_FORMAT_V1: u32 = 1;

/// `tl_pool` (rt.03) is opaque to Rust. The engine passes a null pointer,
/// which every kernel reads as "serial".
#[repr(C)]
pub struct TlPool {
    _private: [u8; 0],
}

/// `tl_arena` (rt.02), opaque. A null scratch arena makes the attention
/// kernel use a private one.
#[repr(C)]
pub struct TlArena {
    _private: [u8; 0],
}

/// `tl_kv_pool` (rt.04), opaque. Owned through [`KvPool`].
#[repr(C)]
pub struct RawKvPool {
    _private: [u8; 0],
}

/// `tl_kv_cfg`: 28 bytes, no padding (c/ABI.md struct table).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvCfg {
    pub n_blocks: u32,
    pub block_tokens: u32,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: i32,
    pub format: u32,
}

/// `tl_kv_stats`: free + used + cached == n_blocks always.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvStats {
    pub free: u32,
    pub used: u32,
    pub cached: u32,
    pub evictions: u32,
}

/// `tl_allocator`: the hook every allocation inside the library goes
/// through (c/ABI.md rule 1). 24 bytes: two function pointers and `user`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Allocator {
    pub alloc: Option<unsafe extern "C" fn(user: *mut c_void, n: usize, align: usize) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(user: *mut c_void, p: *mut c_void)>,
    pub user: *mut c_void,
}

extern "C" {
    // abi.h (rt.01)
    pub fn tl_abi_version() -> u32;
    pub fn tl_status_str(s: i32) -> *const c_char;
    pub fn tl_last_error() -> *const c_char;
    pub fn tl_set_last_error(msg: *const c_char);
    pub fn tl_set_allocator(a: *const Allocator) -> i32;

    // matmul.h (M03.1, L9.1)
    pub fn tl_matmul_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i64,
        n: i64,
        k: i64,
        lda: i64,
        ldb: i64,
        ldc: i64,
        alpha: f32,
        beta: f32,
        trans_b: c_int,
        tp: *mut TlPool,
    ) -> i32;

    // softmax.h (L9.2)
    pub fn tl_softmax_f32(x: *const f32, y: *mut f32, rows: i64, cols: i64) -> i32;

    // attention.h (L9.3)
    pub fn tl_flash_attn_fwd_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
        lse: *mut f32,
        b: i64,
        h: i64,
        hkv: i64,
        tq: i64,
        tk: i64,
        d: i64,
        scale: f32,
        q_offset: i64,
        causal: c_int,
        window: i64,
        sink_logits: *const f32,
        br: i64,
        bc: i64,
        scratch: *mut TlArena,
        tp: *mut TlPool,
    ) -> i32;

    // qmatmul.h (L9.5)
    pub fn tl_matmul_q4_f32(
        x: *const f32,
        wq: *const u8,
        scales_f16: *const u16,
        y: *mut f32,
        m: i64,
        n: i64,
        k: i64,
        group: i64,
        tp: *mut TlPool,
    ) -> i32;

    // elementwise.h (L9.6)
    pub fn tl_rmsnorm_f32(x: *const f32, w: *const f32, y: *mut f32, rows: i64, d: i64, eps: f32);
    pub fn tl_rope_f32(
        x: *mut f32,
        pos: *const i32,
        t: i64,
        h: i64,
        d: i64,
        d_rot: i64,
        inv_freq: *const f32,
        attn_scaling: f32,
        layout: c_int,
    );
    pub fn tl_silu_mul_f32(gate: *const f32, up: *const f32, y: *mut f32, n: i64);
    pub fn tl_embedding_f32(table: *const f32, ids: *const i32, out: *mut f32, n: i64, d: i64);
    pub fn tl_add_f32(a: *const f32, b: *const f32, y: *mut f32, n: i64);
    pub fn tl_argmax_f32(x: *const f32, n: i64) -> i32;

    // kv_pool.h (rt.04)
    pub fn tl_kv_pool_create(cfg: *const KvCfg, out: *mut *mut RawKvPool) -> i32;
    pub fn tl_kv_pool_destroy(p: *mut RawKvPool);
    pub fn tl_kv_alloc(p: *mut RawKvPool, n: u32, ids: *mut u32) -> i32;
    pub fn tl_kv_ref(p: *mut RawKvPool, id: u32);
    pub fn tl_kv_unref(p: *mut RawKvPool, id: u32) -> i32;
    pub fn tl_kv_cow(p: *mut RawKvPool, id: u32, out: *mut u32) -> i32;
    pub fn tl_kv_set_fill(p: *mut RawKvPool, id: u32, n_tokens: u32) -> i32;
    pub fn tl_kv_fill(p: *const RawKvPool, id: u32) -> u32;
    pub fn tl_kv_block_hash(parent: u64, toks: *const u32, n: u32) -> u64;
    pub fn tl_kv_register(p: *mut RawKvPool, id: u32, hash: u64) -> i32;
    pub fn tl_kv_lookup(p: *mut RawKvPool, hash: u64, id: *mut u32) -> i32;
    pub fn tl_kv_block_ptr(p: *const RawKvPool, id: u32, layer: u32, is_v: c_int) -> *mut c_void;
    pub fn tl_kv_pool_cfg(p: *const RawKvPool, out: *mut KvCfg);
    pub fn tl_kv_block_bytes(p: *const RawKvPool) -> usize;
    pub fn tl_kv_export_bytes(p: *const RawKvPool, n: u32) -> usize;
    pub fn tl_kv_export(
        p: *const RawKvPool,
        ids: *const u32,
        n: u32,
        buf: *mut c_void,
        cap: usize,
        written: *mut usize,
    ) -> i32;
    pub fn tl_kv_import(p: *mut RawKvPool, buf: *const c_void, len: usize, ids_out: *mut u32) -> i32;
    pub fn tl_kv_stats_get(p: *const RawKvPool, out: *mut KvStats);
    pub fn tl_crc32c(data: *const c_void, n: usize, crc: u32) -> u32;
}

/// A failed call: the status, its name from `tl_status_str`, and the
/// message from the error slot (or from this crate, for checks done in Rust).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlError {
    pub status: i32,
    pub name: String,
    pub message: String,
}

impl fmt::Display for TlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SOLUTION-BEGIN L10.1
        write!(f, "{} ({}): {}", self.name, self.status, self.message)
        // SOLUTION-END
    }
}

impl std::error::Error for TlError {}

/// Copies a C string owned by libtinyllm into a Rust `String`. NULL gives "".
fn c_text(p: *const c_char) -> String {
    // SOLUTION-BEGIN L10.1
    if p.is_null() {
        return String::new();
    }
    // SAFETY: libtinyllm returns NUL-terminated strings that stay valid until
    // the next tl_ call on this thread; we copy before making another.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    // SOLUTION-END
}

/// The error for `status`, read from C right after the failing call.
pub fn last_error(status: i32) -> TlError {
    // SOLUTION-BEGIN L10.1
    // SAFETY: both functions take no pointers and return static or
    // thread-local strings.
    let message = c_text(unsafe { tl_last_error() });
    let name = c_text(unsafe { tl_status_str(status) });
    TlError { status, name, message }
    // SOLUTION-END
}

/// `Ok(())` for `TL_OK`, else the error slot's message.
fn check(status: i32) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    if status == TL_OK {
        Ok(())
    } else {
        Err(last_error(status))
    }
    // SOLUTION-END
}

/// `tl_abi_version()` of the linked library.
pub fn abi_version() -> u32 {
    // SOLUTION-BEGIN L10.1
    // SAFETY: no arguments, no shared state.
    unsafe { tl_abi_version() }
    // SOLUTION-END
}

/// Refuses a library built for another ABI version (c/ABI.md rule 12).
pub fn check_abi() -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let got = abi_version();
    if got == ABI_VERSION {
        Ok(())
    } else {
        Err(TlError {
            status: TL_EUNSUPPORTED,
            name: "TL_EUNSUPPORTED".to_string(),
            message: format!("libtinyllm has ABI version {got}; tl-sys was written for {ABI_VERSION}"),
        })
    }
    // SOLUTION-END
}

/// An error raised in Rust, before any C call.
fn rust_error(message: String) -> TlError {
    // SOLUTION-BEGIN L10.1
    TlError { status: TL_EINVAL, name: "TL_EINVAL".to_string(), message }
    // SOLUTION-END
}

/// `Err` naming `what` unless `have >= want`.
fn need(what: &str, have: usize, want: usize) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    if have < want {
        Err(rust_error(format!("tl_sys::{what}: has {have} elements, needs {want}")))
    } else {
        Ok(())
    }
    // SOLUTION-END
}

/// A dimension as `int64_t`.
fn dim(x: usize) -> Result<i64, TlError> {
    // SOLUTION-BEGIN L10.1
    i64::try_from(x).map_err(|_| rust_error(format!("tl_sys: dimension {x} does not fit in int64_t")))
    // SOLUTION-END
}

/// `a * b`, or an error on overflow.
fn mul(a: usize, b: usize) -> Result<usize, TlError> {
    // SOLUTION-BEGIN L10.1
    a.checked_mul(b).ok_or_else(|| rust_error(format!("tl_sys: {a} * {b} overflows")))
    // SOLUTION-END
}

/// The fewest elements a row-major matrix of `rows` rows needs when rows sit
/// `ld` elements apart and each row uses `cols` of them: (rows-1)*ld + cols.
/// None on overflow.
fn needed(rows: usize, cols: usize, ld: usize) -> Option<usize> {
    // SOLUTION-BEGIN L10.1
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    (rows - 1).checked_mul(ld)?.checked_add(cols)
    // SOLUTION-END
}

/// Installs `a` as the library's allocator hook; `None` restores the
/// default (malloc). The C side copies the struct.
///
/// # Safety
///
/// No object created through the old hook may be alive (c/ABI.md: change
/// the hook only while nothing it allocated is live), and the functions in
/// `a` must stay callable for as long as the hook is installed.
pub unsafe fn set_allocator(a: Option<&Allocator>) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let p = a.map_or(std::ptr::null(), |x| x as *const Allocator);
    check(tl_set_allocator(p))
    // SOLUTION-END
}

/// `C = alpha * A @ op(B) + beta * C` with explicit leading dimensions, as
/// `tl_matmul_f32` defines it. Panics never; an argument C rejects comes
/// back as its `TlError`. Slices too short for the dimensions are rejected
/// here, before C could read or write past their end.
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32_strided(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
    lda: usize, ldb: usize, ldc: usize,
    alpha: f32, beta: f32, trans_b: bool,
) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
    let checks = [
        ("a", a.len(), needed(m, k, lda)),
        ("b", b.len(), needed(b_rows, b_cols, ldb)),
        ("c", c.len(), needed(m, n, ldc)),
    ];
    for (name, have, want) in checks {
        match want {
            None => return Err(rust_error(format!("tl_sys::matmul_f32: {name} dimensions overflow"))),
            Some(w) if have < w => {
                return Err(rust_error(format!("tl_sys::matmul_f32: {name} has {have} elements, needs {w}")))
            }
            Some(_) => {}
        }
    }
    // SAFETY: every slice holds at least the elements the dimensions reach
    // (checked above), C does not keep the pointers after returning, and
    // `c` is a unique borrow so it cannot overlap `a` or `b`.
    let status = unsafe {
        tl_matmul_f32(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            dim(lda)?,
            dim(ldb)?,
            dim(ldc)?,
            alpha,
            beta,
            c_int::from(trans_b),
            std::ptr::null_mut(),
        )
    };
    check(status)
    // SOLUTION-END
}

/// `C = A @ op(B)` for tightly packed matrices: A is m x k, B is k x n
/// (or n x k with `trans_b`), C is m x n and is overwritten.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, trans_b: bool) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let ldb = if trans_b { k } else { n };
    matmul_f32_strided(a, b, c, m, n, k, k, ldb, n, 1.0, 0.0, trans_b)
    // SOLUTION-END
}

/// Row softmax of x [rows, cols] into y (L9.2).
pub fn softmax_f32(x: &[f32], y: &mut [f32], rows: usize, cols: usize) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let n = mul(rows, cols)?;
    need("softmax_f32: x", x.len(), n)?;
    need("softmax_f32: y", y.len(), n)?;
    // SAFETY: both slices hold rows * cols elements; y is a unique borrow.
    check(unsafe { tl_softmax_f32(x.as_ptr(), y.as_mut_ptr(), dim(rows)?, dim(cols)?) })
    // SOLUTION-END
}

/// `y[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * w` for x [rows, d] (L9.6).
pub fn rmsnorm_f32(x: &[f32], w: &[f32], y: &mut [f32], rows: usize, d: usize, eps: f32) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let n = mul(rows, d)?;
    need("rmsnorm_f32: x", x.len(), n)?;
    need("rmsnorm_f32: w", w.len(), d)?;
    need("rmsnorm_f32: y", y.len(), n)?;
    // SAFETY: sizes checked above; the kernel returns void and reads and
    // writes only those elements.
    unsafe { tl_rmsnorm_f32(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), dim(rows)?, dim(d)?, eps) };
    Ok(())
    // SOLUTION-END
}

/// Which elements of a head form a rotated pair (tinyllm/elementwise.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// HF Llama: (x[i], x[i + d_rot / 2]).
    Half,
    /// GPT-J: (x[2i], x[2i + 1]).
    Interleaved,
}

/// Rotary embedding in place on x [t, h, d]; token i is at position pos[i].
#[allow(clippy::too_many_arguments)]
pub fn rope_f32(
    x: &mut [f32], pos: &[i32], t: usize, h: usize, d: usize, d_rot: usize,
    inv_freq: &[f32], attn_scaling: f32, layout: RopeLayout,
) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    if d_rot % 2 != 0 || d_rot > d {
        return Err(rust_error(format!("tl_sys::rope_f32: d_rot {d_rot} must be even and at most d = {d}")));
    }
    need("rope_f32: x", x.len(), mul(mul(t, h)?, d)?)?;
    need("rope_f32: pos", pos.len(), t)?;
    need("rope_f32: inv_freq", inv_freq.len(), d_rot / 2)?;
    let lay = match layout {
        RopeLayout::Half => 0,
        RopeLayout::Interleaved => 1,
    };
    // SAFETY: sizes checked above; the kernel rotates in place.
    unsafe {
        tl_rope_f32(x.as_mut_ptr(), pos.as_ptr(), dim(t)?, dim(h)?, dim(d)?, dim(d_rot)?, inv_freq.as_ptr(), attn_scaling, lay)
    };
    Ok(())
    // SOLUTION-END
}

/// `y[i] = silu(gate[i]) * up[i]` over the whole of `y` (L9.6).
pub fn silu_mul_f32(gate: &[f32], up: &[f32], y: &mut [f32]) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let n = y.len();
    need("silu_mul_f32: gate", gate.len(), n)?;
    need("silu_mul_f32: up", up.len(), n)?;
    // SAFETY: gate and up hold at least n elements; y exactly n.
    unsafe { tl_silu_mul_f32(gate.as_ptr(), up.as_ptr(), y.as_mut_ptr(), dim(n)?) };
    Ok(())
    // SOLUTION-END
}

/// `out[i, :] = table[ids[i], :]` for a table of `rows` rows of width `d`.
/// Every id is checked against `rows` here: the C kernel trusts its caller.
pub fn embedding_f32(table: &[f32], rows: usize, ids: &[i32], out: &mut [f32], d: usize) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    need("embedding_f32: table", table.len(), mul(rows, d)?)?;
    need("embedding_f32: out", out.len(), mul(ids.len(), d)?)?;
    if let Some(&bad) = ids.iter().find(|&&i| i < 0 || i as usize >= rows) {
        return Err(rust_error(format!("tl_sys::embedding_f32: id {bad} outside [0, {rows})")));
    }
    // SAFETY: every id indexes a full row of table; out holds ids.len() rows.
    unsafe { tl_embedding_f32(table.as_ptr(), ids.as_ptr(), out.as_mut_ptr(), dim(ids.len())?, dim(d)?) };
    Ok(())
    // SOLUTION-END
}

/// `acc[i] += b[i]` through `tl_add_f32` with the output aliasing its first
/// input exactly, which elementwise.h allows.
pub fn add_assign_f32(acc: &mut [f32], b: &[f32]) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let n = acc.len();
    need("add_assign_f32: b", b.len(), n)?;
    let p = acc.as_mut_ptr();
    // SAFETY: a and y are the same pointer (exact aliasing is allowed); b is
    // a shared borrow that cannot overlap the unique borrow `acc`.
    unsafe { tl_add_f32(p, b.as_ptr(), p, dim(n)?) };
    Ok(())
    // SOLUTION-END
}

/// The index of the largest value, ties to the lowest index, NaN skipped;
/// `None` when x is empty or all NaN.
pub fn argmax_f32(x: &[f32]) -> Option<usize> {
    // SOLUTION-BEGIN L10.1
    let n = i64::try_from(x.len()).ok()?;
    // SAFETY: x holds n elements.
    let i = unsafe { tl_argmax_f32(x.as_ptr(), n) };
    if i < 0 {
        None
    } else {
        Some(i as usize)
    }
    // SOLUTION-END
}

/// The dimensions of one `tl_flash_attn_fwd_f32` call (tinyllm/attention.h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnShape {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub tq: usize,
    pub tk: usize,
    pub head_dim: usize,
    pub scale: f32,
    /// Absolute position of query 0.
    pub q_offset: usize,
    pub causal: bool,
    /// 0: no sliding window.
    pub window: usize,
}

/// FlashAttention forward (L9.3): q, o [B, H, Tq, D]; k, v [B, Hkv, Tk, D].
pub fn flash_attn_f32(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], s: &AttnShape) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    let qn = mul(mul(mul(s.batch, s.heads)?, s.tq)?, s.head_dim)?;
    let kn = mul(mul(mul(s.batch, s.kv_heads)?, s.tk)?, s.head_dim)?;
    need("flash_attn_f32: q", q.len(), qn)?;
    need("flash_attn_f32: o", o.len(), qn)?;
    need("flash_attn_f32: k", k.len(), kn)?;
    need("flash_attn_f32: v", v.len(), kn)?;
    // SAFETY: every tensor holds the elements its dims reach; lse, the sink
    // logits, the scratch arena, and the thread pool are all NULL, which the
    // header allows.
    let st = unsafe {
        tl_flash_attn_fwd_f32(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            o.as_mut_ptr(),
            std::ptr::null_mut(),
            dim(s.batch)?,
            dim(s.heads)?,
            dim(s.kv_heads)?,
            dim(s.tq)?,
            dim(s.tk)?,
            dim(s.head_dim)?,
            s.scale,
            dim(s.q_offset)?,
            c_int::from(s.causal),
            dim(s.window)?,
            std::ptr::null(),
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    check(st)
    // SOLUTION-END
}

/// `y = x @ W^T` for an int4 weight W [n, k] in the packed layout of
/// formats/safetensors.md (L9.5): wq [n, k / 2] bytes, scales [n, k / group]
/// f16 bit patterns.
#[allow(clippy::too_many_arguments)]
pub fn matmul_q4_f32(
    x: &[f32], wq: &[u8], scales: &[u16], y: &mut [f32],
    m: usize, n: usize, k: usize, group: usize,
) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    if k % 2 != 0 || group == 0 || group % 2 != 0 || k % group != 0 {
        return Err(TlError {
            status: TL_ESHAPE,
            name: "TL_ESHAPE".to_string(),
            message: format!("tl_sys::matmul_q4_f32: k = {k} and group = {group} must be even with k % group == 0"),
        });
    }
    need("matmul_q4_f32: x", x.len(), mul(m, k)?)?;
    need("matmul_q4_f32: wq", wq.len(), mul(n, k / 2)?)?;
    need("matmul_q4_f32: scales", scales.len(), mul(n, k / group)?)?;
    need("matmul_q4_f32: y", y.len(), mul(m, n)?)?;
    // SAFETY: sizes checked above; y is a unique borrow.
    let st = unsafe {
        tl_matmul_q4_f32(
            x.as_ptr(),
            wq.as_ptr(),
            scales.as_ptr(),
            y.as_mut_ptr(),
            dim(m)?,
            dim(n)?,
            dim(k)?,
            dim(group)?,
            std::ptr::null_mut(),
        )
    };
    check(st)
    // SOLUTION-END
}

/// CRC-32C of `data`, continuing from `crc` (0 for a fresh checksum).
pub fn crc32c(data: &[u8], crc: u32) -> u32 {
    // SOLUTION-BEGIN L10.1
    // SAFETY: reads data.len() bytes.
    unsafe { tl_crc32c(data.as_ptr().cast(), data.len(), crc) }
    // SOLUTION-END
}

/// The chained FNV-1a 64 block hash of formats/kv-block.md: `parent` (0 for
/// a sequence's first block) then the block's token ids. Never 0.
pub fn kv_block_hash(parent: u64, toks: &[u32]) -> u64 {
    // SOLUTION-BEGIN L10.1
    let n = u32::try_from(toks.len()).expect("a block holds fewer than 2^32 tokens");
    // SAFETY: reads toks.len() ids; pure function.
    unsafe { tl_kv_block_hash(parent, toks.as_ptr(), n) }
    // SOLUTION-END
}

/// One `tl_kv_pool`, destroyed exactly once when dropped (rt.04).
///
/// The pool is not thread-safe (kv_pool.h): `KvPool` is `Send` but not
/// `Sync`, so sharing one between threads needs a `Mutex`.
pub struct KvPool {
    raw: NonNull<RawKvPool>,
    cfg: KvCfg,
}

// SAFETY: the pool has no thread affinity; only concurrent use is unsafe,
// which `&mut self` on every mutating method (and !Sync) rules out.
unsafe impl Send for KvPool {}

impl KvPool {
    /// `tl_kv_pool_create`. Format v1 pools hold f16 payloads.
    pub fn new(cfg: KvCfg) -> Result<KvPool, TlError> {
        // SOLUTION-BEGIN L10.1
        let mut out: *mut RawKvPool = std::ptr::null_mut();
        // SAFETY: cfg is a valid tl_kv_cfg; out receives the new pool.
        check(unsafe { tl_kv_pool_create(&cfg, &mut out) })?;
        let raw = NonNull::new(out).ok_or_else(|| rust_error("tl_kv_pool_create returned TL_OK and NULL".to_string()))?;
        Ok(KvPool { raw, cfg })
        // SOLUTION-END
    }

    /// The configuration the pool was created with.
    pub fn cfg(&self) -> KvCfg {
        // SOLUTION-BEGIN L10.1
        self.cfg
        // SOLUTION-END
    }

    /// Takes `n` blocks (refcount 1 each), all or nothing; `TL_EFULL` when
    /// free + cached blocks are fewer than `n`.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, TlError> {
        // SOLUTION-BEGIN L10.1
        let mut ids = vec![0u32; n];
        if n == 0 {
            return Ok(ids);
        }
        let n32 = u32::try_from(n).map_err(|_| rust_error(format!("tl_sys::KvPool::alloc: {n} blocks")))?;
        // SAFETY: ids has room for n ids.
        check(unsafe { tl_kv_alloc(self.raw.as_ptr(), n32, ids.as_mut_ptr()) })?;
        Ok(ids)
        // SOLUTION-END
    }

    /// `tl_kv_ref`: one more reference to a used block. The C function
    /// returns void and reports a bad id only through the error slot, so the
    /// slot is cleared first and read after.
    pub fn retain(&mut self, id: u32) -> Result<(), TlError> {
        // SOLUTION-BEGIN L10.1
        // SAFETY: a NULL message clears the slot; the pool pointer is live.
        unsafe {
            tl_set_last_error(std::ptr::null());
            tl_kv_ref(self.raw.as_ptr(), id);
        }
        let msg = c_text(unsafe { tl_last_error() });
        if msg.is_empty() {
            Ok(())
        } else {
            Err(TlError { status: TL_EINVAL, name: "TL_EINVAL".to_string(), message: msg })
        }
        // SOLUTION-END
    }

    /// `tl_kv_unref`: drops one reference; at zero the block becomes cached
    /// (registered) or free.
    pub fn release(&mut self, id: u32) -> Result<(), TlError> {
        // SOLUTION-BEGIN L10.1
        // SAFETY: the pool pointer is live; C range-checks the id.
        check(unsafe { tl_kv_unref(self.raw.as_ptr(), id) })
        // SOLUTION-END
    }

    /// Copy on write: the same id when the caller holds the only reference,
    /// else a private copy (and one reference fewer on `id`).
    pub fn cow(&mut self, id: u32) -> Result<u32, TlError> {
        // SOLUTION-BEGIN L10.1
        let mut out = 0u32;
        // SAFETY: out receives one id.
        check(unsafe { tl_kv_cow(self.raw.as_ptr(), id, &mut out) })?;
        Ok(out)
        // SOLUTION-END
    }

    /// Records how many positions of the block hold data.
    pub fn set_fill(&mut self, id: u32, n_tokens: u32) -> Result<(), TlError> {
        // SOLUTION-BEGIN L10.1
        // SAFETY: the pool pointer is live; C checks id and n_tokens.
        check(unsafe { tl_kv_set_fill(self.raw.as_ptr(), id, n_tokens) })
        // SOLUTION-END
    }

    /// The fill of a block.
    pub fn fill(&self, id: u32) -> u32 {
        // SOLUTION-BEGIN L10.1
        // SAFETY: read-only call on a live pool.
        unsafe { tl_kv_fill(self.raw.as_ptr(), id) }
        // SOLUTION-END
    }

    /// Registers a used, full block under `hash`. `Ok(false)` when another
    /// block already holds that hash (`TL_EBUSY`): this one stays private.
    pub fn register(&mut self, id: u32, hash: u64) -> Result<bool, TlError> {
        // SOLUTION-BEGIN L10.1
        // SAFETY: the pool pointer is live.
        match unsafe { tl_kv_register(self.raw.as_ptr(), id, hash) } {
            TL_OK => Ok(true),
            TL_EBUSY => Ok(false),
            st => Err(last_error(st)),
        }
        // SOLUTION-END
    }

    /// A registered block by hash, with a reference taken for the caller
    /// (a cached block becomes used); `None` on a miss.
    pub fn lookup(&mut self, hash: u64) -> Option<u32> {
        // SOLUTION-BEGIN L10.1
        let mut id = 0u32;
        // SAFETY: id receives one id on a hit.
        match unsafe { tl_kv_lookup(self.raw.as_ptr(), hash, &mut id) } {
            TL_OK => Some(id),
            _ => None,
        }
        // SOLUTION-END
    }

    /// Payload bytes of one block.
    pub fn block_bytes(&self) -> usize {
        // SOLUTION-BEGIN L10.1
        // SAFETY: read-only call on a live pool.
        unsafe { tl_kv_block_bytes(self.raw.as_ptr()) }
        // SOLUTION-END
    }

    /// Elements in one K or V slab: n_kv_heads * block_tokens * head_dim.
    pub fn slab_len(&self) -> usize {
        // SOLUTION-BEGIN L10.1
        self.cfg.n_kv_heads as usize * self.cfg.block_tokens as usize * self.cfg.head_dim as usize
        // SOLUTION-END
    }

    /// The pointer to one slab, checked: f16 pools only, id and layer in range.
    fn slab_ptr(&self, id: u32, layer: u32, is_v: bool) -> Result<*mut u16, TlError> {
        // SOLUTION-BEGIN L10.1
        if self.cfg.dtype != TL_F16 {
            return Err(TlError {
                status: TL_EDTYPE,
                name: "TL_EDTYPE".to_string(),
                message: format!("tl_sys::KvPool: slabs are f16 in format v1; this pool has dtype {}", self.cfg.dtype),
            });
        }
        if id >= self.cfg.n_blocks || layer >= self.cfg.n_layers {
            return Err(rust_error(format!(
                "tl_sys::KvPool: block {id} layer {layer} outside {} blocks x {} layers",
                self.cfg.n_blocks, self.cfg.n_layers
            )));
        }
        // SAFETY: id and layer are in range (checked above).
        let p = unsafe { tl_kv_block_ptr(self.raw.as_ptr(), id, layer, c_int::from(is_v)) };
        if p.is_null() {
            return Err(last_error(TL_EINVAL));
        }
        Ok(p.cast::<u16>())
        // SOLUTION-END
    }

    /// One layer's K (`is_v == false`) or V slab of a block, as f16 bit
    /// patterns laid out [n_kv_heads][block_tokens][head_dim].
    pub fn slab(&self, id: u32, layer: u32, is_v: bool) -> Result<&[u16], TlError> {
        // SOLUTION-BEGIN L10.1
        let p = self.slab_ptr(id, layer, is_v)?;
        // SAFETY: the slab holds slab_len() f16 values and lives as long as
        // the pool; the shared borrow of self keeps writers out.
        Ok(unsafe { std::slice::from_raw_parts(p, self.slab_len()) })
        // SOLUTION-END
    }

    /// The same slab, writable. Write only into blocks you hold a
    /// reference to (kv_pool.h).
    pub fn slab_mut(&mut self, id: u32, layer: u32, is_v: bool) -> Result<&mut [u16], TlError> {
        // SOLUTION-BEGIN L10.1
        let p = self.slab_ptr(id, layer, is_v)?;
        // SAFETY: as in `slab`; the unique borrow of self makes this the only
        // live view of any slab.
        Ok(unsafe { std::slice::from_raw_parts_mut(p, self.slab_len()) })
        // SOLUTION-END
    }

    /// free, used, cached, evictions.
    pub fn stats(&self) -> KvStats {
        // SOLUTION-BEGIN L10.1
        let mut s = KvStats::default();
        // SAFETY: s receives the counters.
        unsafe { tl_kv_stats_get(self.raw.as_ptr(), &mut s) };
        s
        // SOLUTION-END
    }

    /// The export envelope of `ids` (formats/kv-block.md).
    pub fn export(&self, ids: &[u32]) -> Result<Vec<u8>, TlError> {
        // SOLUTION-BEGIN L10.1
        let n = u32::try_from(ids.len()).map_err(|_| rust_error("tl_sys::KvPool::export: too many blocks".to_string()))?;
        // SAFETY: read-only size query.
        let cap = unsafe { tl_kv_export_bytes(self.raw.as_ptr(), n) };
        let mut buf = vec![0u8; cap];
        let mut written = 0usize;
        // SAFETY: buf holds cap bytes; ids holds n ids.
        check(unsafe { tl_kv_export(self.raw.as_ptr(), ids.as_ptr(), n, buf.as_mut_ptr().cast(), cap, &mut written) })?;
        buf.truncate(written);
        Ok(buf)
        // SOLUTION-END
    }

    /// Reads an envelope into newly allocated blocks; returns their ids.
    pub fn import(&mut self, buf: &[u8]) -> Result<Vec<u32>, TlError> {
        // SOLUTION-BEGIN L10.1
        if buf.len() < 12 {
            return Err(TlError { status: TL_EFORMAT, name: "TL_EFORMAT".to_string(), message: "tl_sys::KvPool::import: envelope shorter than its header".to_string() });
        }
        let n = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        let mut ids = vec![0u32; n.max(1)];
        // SAFETY: ids has room for the n_blocks the header announces.
        check(unsafe { tl_kv_import(self.raw.as_ptr(), buf.as_ptr().cast(), buf.len(), ids.as_mut_ptr()) })?;
        ids.truncate(n);
        Ok(ids)
        // SOLUTION-END
    }
}

impl Drop for KvPool {
    fn drop(&mut self) {
        // SOLUTION-BEGIN L10.1
        // SAFETY: raw came from tl_kv_pool_create and is destroyed only here;
        // a moved-from KvPool is never dropped, so this runs once per pool.
        unsafe { tl_kv_pool_destroy(self.raw.as_ptr()) }
        // SOLUTION-END
    }
}
