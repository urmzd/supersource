//! The model graph in Rust over the C kernels (L10.1, D7).
//!
//! One call runs one engine step: a batch of sequences, each contributing a
//! run of new tokens at absolute positions `start..start + n` (a whole
//! prompt, a prefill chunk, or one decode token). Every token of every
//! sequence goes through the projections together, as one matrix of N rows,
//! so the weights are read once per step whatever the batch size; attention
//! is the only per-sequence part. Each kernel is batch- and chunk-invariant
//! (c/ABI.md rule 10), so a sequence's logits do not depend on what else is
//! in the batch or on how its prompt was split into chunks.
//!
//! K and V of every new token are written into the sequence's blocks of the
//! KV pool (rt.04) as f16, then read back for attention: the pool is the
//! only place a sequence's past lives between steps.

use tl_sys::{AttnShape, KvPool, RopeLayout};

use crate::model::{Linear, LlamaWeights, ModelConfig};
use crate::quant::{f16_to_f32, f32_to_f16};

/// One sequence's part of a step: `tokens` sit at positions
/// `start..start + tokens.len()`; `blocks` is its block table and covers at
/// least `start + tokens.len()` positions (block i holds positions
/// `i * block_tokens..(i + 1) * block_tokens`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardSeq<'a> {
    pub tokens: &'a [u32],
    pub start: usize,
    pub blocks: &'a [u32],
}

/// The sequences of one step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardBatch<'a> {
    pub seqs: Vec<ForwardSeq<'a>>,
}

/// Next-token logits: one row of `vocab` values per sequence of the batch,
/// for the position after its last token.
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    pub vocab: usize,
    pub data: Vec<f32>,
}

impl Logits {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        // SOLUTION-BEGIN L10.1
        if self.vocab == 0 {
            0
        } else {
            self.data.len() / self.vocab
        }
        // SOLUTION-END
    }

    /// Row `i`: the logits of sequence `i` of the batch.
    pub fn row(&self, i: usize) -> &[f32] {
        // SOLUTION-BEGIN L10.1
        &self.data[i * self.vocab..(i + 1) * self.vocab]
        // SOLUTION-END
    }
}

/// RoPE inverse frequencies, HF Llama's default: inv_freq[i] =
/// 1 / theta^(2i / d) for i < d / 2, computed in f32 as transformers does.
pub fn rope_inv_freq(theta: f64, d: usize) -> Vec<f32> {
    // SOLUTION-BEGIN L10.1
    let base = theta as f32;
    (0..d / 2).map(|i| 1.0f32 / base.powf((2 * i) as f32 / d as f32)).collect()
    // SOLUTION-END
}

fn err(e: tl_sys::TlError) -> String {
    // SOLUTION-BEGIN L10.1
    e.to_string()
    // SOLUTION-END
}

/// Writes K and V rows of one layer into the pool: row r of `k` and `v`
/// ([n, n_kv_heads * head_dim]) is position `start + r` of the sequence.
/// Then records each touched block's fill.
pub fn write_kv(pool: &mut KvPool, layer: u32, seq: &ForwardSeq, k: &[f32], v: &[f32]) -> Result<(), String> {
    // SOLUTION-BEGIN L10.1
    let cfg = pool.cfg();
    let (bt, hkv, hd) = (cfg.block_tokens as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
    let end = seq.start + seq.tokens.len();
    if seq.blocks.len() * bt < end {
        return Err(format!("forward: {} blocks cover {} positions, the step needs {end}", seq.blocks.len(), seq.blocks.len() * bt));
    }
    for (src, is_v) in [(k, false), (v, true)] {
        for r in 0..seq.tokens.len() {
            let p = seq.start + r;
            let (blk, slot) = (seq.blocks[p / bt], p % bt);
            let slab = pool.slab_mut(blk, layer, is_v).map_err(err)?;
            for h in 0..hkv {
                let dst = &mut slab[(h * bt + slot) * hd..(h * bt + slot + 1) * hd];
                let row = &src[r * hkv * hd + h * hd..r * hkv * hd + (h + 1) * hd];
                for (d, x) in dst.iter_mut().zip(row) {
                    *d = f32_to_f16(*x);
                }
            }
        }
    }
    if layer == 0 {
        for b in seq.start / bt..end.div_ceil(bt) {
            let fill = (end - b * bt).min(bt) as u32;
            if pool.fill(seq.blocks[b]) < fill {
                pool.set_fill(seq.blocks[b], fill).map_err(err)?;
            }
        }
    }
    Ok(())
    // SOLUTION-END
}

/// K or V of positions `0..tk` of one sequence and layer, widened to f32,
/// laid out [n_kv_heads, tk, head_dim] as the attention kernel wants.
pub fn gather_kv(pool: &KvPool, layer: u32, blocks: &[u32], tk: usize, is_v: bool) -> Result<Vec<f32>, String> {
    // SOLUTION-BEGIN L10.1
    let cfg = pool.cfg();
    let (bt, hkv, hd) = (cfg.block_tokens as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
    let mut out = vec![0.0f32; hkv * tk * hd];
    for b in 0..tk.div_ceil(bt) {
        let slab = pool.slab(blocks[b], layer, is_v).map_err(err)?;
        for slot in 0..bt.min(tk - b * bt) {
            let p = b * bt + slot;
            for h in 0..hkv {
                let src = &slab[(h * bt + slot) * hd..(h * bt + slot + 1) * hd];
                let dst = &mut out[(h * tk + p) * hd..(h * tk + p + 1) * hd];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = f16_to_f32(*s);
                }
            }
        }
    }
    Ok(out)
    // SOLUTION-END
}

/// `y = x @ W^T` into a new buffer of m rows.
fn project(l: &Linear, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    // SOLUTION-BEGIN L10.1
    let (out, _) = l.dims();
    let mut y = vec![0.0f32; m * out];
    l.forward(x, m, &mut y).map_err(err)?;
    Ok(y)
    // SOLUTION-END
}

/// The Llama forward of one step. Returns the logits of each sequence's
/// last token; with `hidden`, also fills it with the final-normed hidden
/// states of every token of the batch, [N, hidden_size] in batch order.
pub fn llama_forward(
    w: &LlamaWeights,
    cfg: &ModelConfig,
    inv_freq: &[f32],
    pool: &mut KvPool,
    batch: &ForwardBatch,
    hidden: Option<&mut Vec<f32>>,
) -> Result<Logits, String> {
    // SOLUTION-BEGIN L10.1
    let d = cfg.hidden_size;
    let (h, hkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let mut ids = Vec::new();
    let mut pos = Vec::new();
    let mut offs = Vec::with_capacity(batch.seqs.len());
    for s in &batch.seqs {
        if s.tokens.is_empty() {
            return Err("forward: a sequence with no new tokens".to_string());
        }
        if s.start + s.tokens.len() > cfg.max_position_embeddings {
            return Err(format!("forward: position {} is past the context of {}", s.start + s.tokens.len(), cfg.max_position_embeddings));
        }
        offs.push(ids.len());
        for (i, &t) in s.tokens.iter().enumerate() {
            ids.push(i32::try_from(t).map_err(|_| format!("forward: token id {t}"))?);
            pos.push(i32::try_from(s.start + i).map_err(|_| "forward: position overflow".to_string())?);
        }
    }
    let n = ids.len();
    let mut x = vec![0.0f32; n * d];
    tl_sys::embedding_f32(&w.embed, cfg.vocab_size, &ids, &mut x, d).map_err(err)?;
    let mut hbuf = vec![0.0f32; n * d];
    let scale = 1.0 / (hd as f32).sqrt();
    for (li, lw) in w.layers.iter().enumerate() {
        let layer = li as u32;
        // attention block
        tl_sys::rmsnorm_f32(&x, &lw.input_norm, &mut hbuf, n, d, cfg.rms_norm_eps).map_err(err)?;
        let mut q = project(&lw.q, &hbuf, n)?;
        let mut k = project(&lw.k, &hbuf, n)?;
        let v = project(&lw.v, &hbuf, n)?;
        tl_sys::rope_f32(&mut q, &pos, n, h, hd, hd, inv_freq, 1.0, RopeLayout::Half).map_err(err)?;
        tl_sys::rope_f32(&mut k, &pos, n, hkv, hd, hd, inv_freq, 1.0, RopeLayout::Half).map_err(err)?;
        let mut attn = vec![0.0f32; n * h * hd];
        for (s, &off) in batch.seqs.iter().zip(&offs) {
            let ns = s.tokens.len();
            write_kv(pool, layer, s, &k[off * hkv * hd..(off + ns) * hkv * hd], &v[off * hkv * hd..(off + ns) * hkv * hd])?;
            let tk = s.start + ns;
            let kc = gather_kv(pool, layer, s.blocks, tk, false)?;
            let vc = gather_kv(pool, layer, s.blocks, tk, true)?;
            // [ns, h, hd] -> [h, ns, hd]
            let mut qs = vec![0.0f32; h * ns * hd];
            for i in 0..ns {
                for hh in 0..h {
                    let src = &q[((off + i) * h + hh) * hd..((off + i) * h + hh + 1) * hd];
                    qs[(hh * ns + i) * hd..(hh * ns + i + 1) * hd].copy_from_slice(src);
                }
            }
            let mut o = vec![0.0f32; h * ns * hd];
            let shape = AttnShape { batch: 1, heads: h, kv_heads: hkv, tq: ns, tk, head_dim: hd, scale, q_offset: s.start, causal: true, window: 0 };
            tl_sys::flash_attn_f32(&qs, &kc, &vc, &mut o, &shape).map_err(err)?;
            for i in 0..ns {
                for hh in 0..h {
                    let dst = &mut attn[((off + i) * h + hh) * hd..((off + i) * h + hh + 1) * hd];
                    dst.copy_from_slice(&o[(hh * ns + i) * hd..(hh * ns + i + 1) * hd]);
                }
            }
        }
        let ao = project(&lw.o, &attn, n)?;
        tl_sys::add_assign_f32(&mut x, &ao).map_err(err)?;
        // MLP block
        tl_sys::rmsnorm_f32(&x, &lw.post_norm, &mut hbuf, n, d, cfg.rms_norm_eps).map_err(err)?;
        let g = project(&lw.gate, &hbuf, n)?;
        let u = project(&lw.up, &hbuf, n)?;
        let mut m = vec![0.0f32; g.len()];
        tl_sys::silu_mul_f32(&g, &u, &mut m).map_err(err)?;
        let dn = project(&lw.down, &m, n)?;
        tl_sys::add_assign_f32(&mut x, &dn).map_err(err)?;
    }
    if let Some(out) = hidden {
        out.resize(n * d, 0.0);
        tl_sys::rmsnorm_f32(&x, &w.norm, out, n, d, cfg.rms_norm_eps).map_err(err)?;
    }
    // final norm and LM head on each sequence's last token only
    let rows = batch.seqs.len();
    let mut last = vec![0.0f32; rows * d];
    for (r, (s, &off)) in batch.seqs.iter().zip(&offs).enumerate() {
        let i = off + s.tokens.len() - 1;
        last[r * d..(r + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
    }
    let mut normed = vec![0.0f32; rows * d];
    tl_sys::rmsnorm_f32(&last, &w.norm, &mut normed, rows, d, cfg.rms_norm_eps).map_err(err)?;
    let v = cfg.vocab_size;
    let mut data = vec![0.0f32; rows * v];
    match &w.lm_head {
        Some(head) => head.forward(&normed, rows, &mut data).map_err(err)?,
        None => tl_sys::matmul_f32(&normed, &w.embed, &mut data, rows, v, d, true).map_err(err)?,
    }
    Ok(Logits { vocab: v, data })
    // SOLUTION-END
}

/// The bigram forward: each sequence's logits are row `last token` of W,
/// computed as onehot(last) @ W through `tl_matmul_f32`, as in L10.0.
pub fn bigram_forward(w: &[f32], batch: &ForwardBatch) -> Result<Logits, String> {
    // SOLUTION-BEGIN L10.1
    let rows = batch.seqs.len();
    let mut onehot = vec![0.0f32; rows * 256];
    for (r, s) in batch.seqs.iter().enumerate() {
        let last = *s.tokens.last().ok_or("forward: a sequence with no new tokens")? as usize;
        if last >= 256 {
            return Err(format!("forward: token id {last} outside the byte vocabulary"));
        }
        onehot[r * 256 + last] = 1.0;
    }
    let mut data = vec![0.0f32; rows * 256];
    tl_sys::matmul_f32(&onehot, w, &mut data, rows, 256, 256, false).map_err(err)?;
    Ok(Logits { vocab: 256, data })
    // SOLUTION-END
}
