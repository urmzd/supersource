//! tl-sys kv (L10.1): the KV block pool of rt.04 behind an RAII owner.
//!
//! [`KvPool`] owns one `tl_kv_pool`: `KvPool::new` is the only way to get
//! one and its `Drop` calls `tl_kv_pool_destroy` exactly once, so safe code
//! can neither leak a pool nor free it twice. The pool is not thread-safe
//! (kv_pool.h): `KvPool` is `Send`, not `Sync`.

use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

use crate::kernels::{check, invalid, tl_set_last_error, TL_EBUSY, TL_EDTYPE, TL_EFORMAT, TL_F16};
use crate::{last_error, tl_last_error, TlError, TL_EINVAL, TL_OK};

/// KV block format v1: f16 payloads (formats/kv-block.md, D13).
pub const TL_KV_FORMAT_V1: u32 = 1;

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

extern "C" {
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

/// Copies a C string owned by libtinyllm (NULL gives "").
fn c_text(p: *const c_char) -> String {
    // SOLUTION-BEGIN L10.1
    if p.is_null() {
        return String::new();
    }
    // SAFETY: libtinyllm returns NUL-terminated strings valid until the next
    // tl_ call on this thread; we copy at once.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
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
        let raw = NonNull::new(out).ok_or_else(|| invalid("tl_kv_pool_create returned TL_OK and NULL".to_string()))?;
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
        let n32 = u32::try_from(n).map_err(|_| invalid(format!("tl_sys::KvPool::alloc: {n} blocks")))?;
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
            return Err(invalid(format!(
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
        let n = u32::try_from(ids.len()).map_err(|_| invalid("tl_sys::KvPool::export: too many blocks".to_string()))?;
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
