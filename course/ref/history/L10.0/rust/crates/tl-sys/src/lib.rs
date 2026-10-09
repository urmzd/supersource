//! tl-sys v0 (L10.0): the minimal hand-written binding to libtinyllm.
//!
//! Only what the tracer engine calls: the ABI version, the error slot, and
//! `tl_matmul_f32` (contracts/c/include/tinyllm/{abi,matmul}.h). The raw
//! `extern "C"` block mirrors the C prototypes exactly; the safe functions
//! below check every slice length before handing a pointer to C, and turn a
//! non-zero `tl_status` into a `TlError` carrying `tl_last_error()`.
//!
//! L10.1 takes this file over (`upgrades`) with the full binding and RAII
//! wrappers; the functions here keep their signatures.

use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_char, c_int};

/// The ABI version this binding was written against (`TL_ABI_VERSION`).
pub const ABI_VERSION: u32 = 1;

/// `tl_status` values used here (`tinyllm/abi.h`). A status is an `i32`,
/// never a Rust enum: C may hand back a value this crate has never heard of.
pub const TL_OK: i32 = 0;
pub const TL_EINVAL: i32 = 1;
pub const TL_EUNSUPPORTED: i32 = 9;

/// `tl_pool` (rt.03) is opaque to Rust. v0 always passes a null pointer,
/// which `tl_matmul_f32` reads as "serial".
#[repr(C)]
pub struct TlPool {
    _private: [u8; 0],
}

extern "C" {
    pub fn tl_abi_version() -> u32;
    pub fn tl_status_str(s: i32) -> *const c_char;
    pub fn tl_last_error() -> *const c_char;
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
        // SOLUTION-BEGIN L10.0
        write!(f, "{} ({}): {}", self.name, self.status, self.message)
        // SOLUTION-END
    }
}

impl std::error::Error for TlError {}

/// Copies a C string owned by libtinyllm into a Rust `String`. NULL gives "".
fn c_text(p: *const c_char) -> String {
    // SOLUTION-BEGIN L10.0
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
    // SOLUTION-BEGIN L10.0
    // SAFETY: both functions take no pointers and return static or
    // thread-local strings.
    let message = c_text(unsafe { tl_last_error() });
    let name = c_text(unsafe { tl_status_str(status) });
    TlError { status, name, message }
    // SOLUTION-END
}

/// `tl_abi_version()` of the linked library.
pub fn abi_version() -> u32 {
    // SOLUTION-BEGIN L10.0
    // SAFETY: no arguments, no shared state.
    unsafe { tl_abi_version() }
    // SOLUTION-END
}

/// Refuses a library built for another ABI version (c/ABI.md rule 12).
pub fn check_abi() -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.0
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
    // SOLUTION-BEGIN L10.0
    TlError { status: TL_EINVAL, name: "TL_EINVAL".to_string(), message }
    // SOLUTION-END
}

/// The fewest elements a row-major matrix of `rows` rows needs when rows sit
/// `ld` elements apart and each row uses `cols` of them: (rows-1)*ld + cols.
/// None on overflow.
fn needed(rows: usize, cols: usize, ld: usize) -> Option<usize> {
    // SOLUTION-BEGIN L10.0
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    (rows - 1).checked_mul(ld)?.checked_add(cols)
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
    // SOLUTION-BEGIN L10.0
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
    let dims: Result<Vec<i64>, _> = [m, n, k, lda, ldb, ldc].iter().map(|&x| i64::try_from(x)).collect();
    let d = dims.map_err(|_| rust_error("tl_sys::matmul_f32: a dimension does not fit in int64_t".to_string()))?;
    // SAFETY: every slice holds at least the elements the dimensions reach
    // (checked above), C does not keep the pointers after returning, and
    // `c` is a unique borrow so it cannot overlap `a` or `b`.
    let status = unsafe {
        tl_matmul_f32(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            alpha,
            beta,
            c_int::from(trans_b),
            std::ptr::null_mut(),
        )
    };
    if status == TL_OK {
        Ok(())
    } else {
        Err(last_error(status))
    }
    // SOLUTION-END
}

/// `C = A @ op(B)` for tightly packed matrices: A is m x k, B is k x n
/// (or n x k with `trans_b`), C is m x n and is overwritten.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, trans_b: bool) -> Result<(), TlError> {
    // SOLUTION-BEGIN L10.0
    let ldb = if trans_b { k } else { n };
    matmul_f32_strided(a, b, c, m, n, k, k, ldb, n, 1.0, 0.0, trans_b)
    // SOLUTION-END
}
