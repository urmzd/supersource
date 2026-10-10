//! tl-sys kernels (L10.1): the C kernels of a Llama forward, the status
//! and dtype codes, and the allocator hook, with safe wrappers.
//!
//! Every wrapper checks every slice length (and every embedding id against
//! its table) before a pointer reaches C, which trusts the dimensions it is
//! given, and turns a non-zero `tl_status` into a [`TlError`] that carries
//! `tl_last_error()` (`crate::last_error`). Declarations use the exact C
//! types: `int64_t` is `i64`, `int` is `c_int`, `tl_status` and `tl_dtype`
//! are `i32` constants, never Rust enums (c/ABI.md rule 5).

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

use crate::{last_error, TlError, TlPool, TL_EINVAL, TL_OK};

/// The `tl_status` values v0 did not need (`tinyllm/abi.h`); `TL_OK`,
/// `TL_EINVAL`, and `TL_EUNSUPPORTED` are in the crate root.
pub const TL_ENOMEM: i32 = 2;
pub const TL_ESHAPE: i32 = 3;
pub const TL_EDTYPE: i32 = 4;
pub const TL_EFULL: i32 = 5;
pub const TL_ENOTFOUND: i32 = 6;
pub const TL_EFORMAT: i32 = 7;
pub const TL_EBUSY: i32 = 8;
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


/// `tl_arena` (rt.02), opaque. A null scratch arena makes the attention
/// kernel use a private one.
#[repr(C)]
pub struct TlArena {
    _private: [u8; 0],
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
    // abi.h (rt.01): the allocator hook and the error-slot setter
    pub fn tl_set_last_error(msg: *const c_char);
    pub fn tl_set_allocator(a: *const Allocator) -> i32;
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
}

/// `Ok(())` for `TL_OK`, else the error slot's message.
pub(crate) fn check(status: i32) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    if status == TL_OK {
        Ok(())
    } else {
        Err(last_error(status))
    }
    // SOLUTION-END
}

/// An error raised in Rust, before any C call (`TL_EINVAL`).
pub(crate) fn invalid(message: String) -> TlError {
    // SOLUTION-BEGIN L10.1
    TlError { status: TL_EINVAL, name: "TL_EINVAL".to_string(), message }
    // SOLUTION-END
}

/// `Err` naming `what` unless `have >= want`.
pub(crate) fn need(what: &str, have: usize, want: usize) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.1
    if have < want {
        Err(invalid(format!("tl_sys::{what}: has {have} elements, needs {want}")))
    } else {
        Ok(())
    }
    // SOLUTION-END
}

/// A dimension as `int64_t`.
pub(crate) fn dim(x: usize) -> Result<i64, TlError> {
    // SOLUTION-BEGIN L10.1
    i64::try_from(x).map_err(|_| invalid(format!("tl_sys: dimension {x} does not fit in int64_t")))
    // SOLUTION-END
}

/// `a * b`, or an error on overflow.
pub(crate) fn mul(a: usize, b: usize) -> Result<usize, TlError> {
    // SOLUTION-BEGIN L10.1
    a.checked_mul(b).ok_or_else(|| invalid(format!("tl_sys: {a} * {b} overflows")))
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
        return Err(invalid(format!("tl_sys::rope_f32: d_rot {d_rot} must be even and at most d = {d}")));
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
        return Err(invalid(format!("tl_sys::embedding_f32: id {bad} outside [0, {rows})")));
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

