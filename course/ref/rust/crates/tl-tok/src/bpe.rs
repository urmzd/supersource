//! Byte-level BPE (L1.5), exact against your Python `BPETokenizer` (L1.2),
//! GPT-2's tiktoken ids, and Hugging Face `tokenizers` on SmolLM2.
//!
//! Encoding, in order:
//!
//! 1. **Added tokens** (`<|endoftext|>`, `<|im_start|>`, ...) that appear
//!    literally and are allowed are split out first, leftmost-longest; they
//!    encode to their own id and are never pre-tokenized or merged.
//! 2. The text between them is **pre-tokenized** ([`crate::pretok`]).
//! 3. Each pre-token's UTF-8 bytes become the ids of the 256 single-byte
//!    tokens. A byte with no token is dropped (no `unk`), as Hugging Face does.
//! 4. **Merges**: repeatedly merge the adjacent pair with the lowest merge
//!    rank, the leftmost on ties, until no adjacent pair has a rank.
//!
//! Step 4 naively rescans every pair after each merge: O(n^2) per
//! pre-token. Here every mergeable pair sits in a [`LazyHeap`] keyed by
//! (rank, position of its left symbol); a merge removes the two pairs it
//! destroys by handle and pushes the two it creates: O(n log n). Both give
//! the same ids because both always apply the (lowest rank, leftmost) pair.
//!
//! The vocabulary (`bytes -> id`) and the merge table
//! (`(left, right) -> (rank, merged id)`) are [`RobinHoodMap`]s (ds.05).

use std::cmp::Ordering;

use tl_ds::heap::{Handle, LazyHeap};
use tl_ds::robin::RobinHoodMap;

use crate::pretok::PreTokenizer;
use crate::{LoadError, SpecialSet, Tokenizer};

/// GPT-2's byte-to-character map (`bytes_to_unicode`, M05.2): the 188
/// printable bytes `!`..=`~`, `¡`..=`¬`, `®`..=`ÿ` map to the code point of
/// the same value; the other 68 bytes map to U+0100, U+0101, ... in byte
/// order. Token strings in `vocab.json` and `tokenizer.json` are written in
/// this alphabet (byte 0x20, the space, is `Ġ` = U+0120).
pub fn bytes_to_unicode() -> [char; 256] {
    // SOLUTION-BEGIN L1.5
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        out[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
    // SOLUTION-END
}

/// The inverse map, indexed by code point: `inv[c]` is the byte that `c`
/// stands for. The alphabet ends at U+0143 (255 + 68), so 324 entries.
pub fn unicode_to_bytes() -> Vec<Option<u8>> {
    // SOLUTION-BEGIN L1.5
    let mut inv = vec![None; 324];
    for (b, c) in bytes_to_unicode().iter().enumerate() {
        inv[*c as usize] = Some(b as u8);
    }
    inv
    // SOLUTION-END
}

/// The bytes a token string stands for, or `None` when a character is
/// outside the byte alphabet. `inv` is [`unicode_to_bytes`].
pub fn token_str_to_bytes(s: &str, inv: &[Option<u8>]) -> Option<Vec<u8>> {
    // SOLUTION-BEGIN L1.5
    s.chars().map(|c| inv.get(c as usize).copied().flatten()).collect()
    // SOLUTION-END
}

/// An added token: literal text with its own id, matched before
/// pre-tokenization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub special: bool,
}

/// What merging `(left, right)` gives: the merge's rank and the new id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Merge {
    pub rank: u32,
    pub id: u32,
}

const NONE: usize = usize::MAX;

/// A symbol of a pre-token while merging: a token id and its neighbours.
#[derive(Clone, Copy, Debug)]
struct Sym {
    id: u32,
    prev: usize,
    next: usize,
}

/// A byte-level BPE tokenizer.
#[derive(Clone)]
pub struct ByteBpe {
    vocab: RobinHoodMap<Vec<u8>, u32>,
    tokens: Vec<Vec<u8>>, // id -> bytes (an added token's UTF-8 content)
    merges: RobinHoodMap<(u32, u32), Merge>,
    byte_ids: [Option<u32>; 256],
    added: Vec<AddedToken>,
    pretok: PreTokenizer,
}

impl std::fmt::Debug for ByteBpe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SOLUTION-BEGIN L1.5
        f.debug_struct("ByteBpe")
            .field("vocab_size", &self.tokens.len())
            .field("merges", &self.merges.len())
            .field("added", &self.added.len())
            .field("pretok", &self.pretok)
            .finish()
        // SOLUTION-END
    }
}

impl ByteBpe {
    /// Builds a tokenizer from token bytes and ids, merges by rank (as
    /// pairs of token bytes; `merges[r]` has rank `r`), added tokens, and a
    /// pre-tokenizer. `Format` errors: a merge whose parts or result is not
    /// in the vocabulary, two tokens with one id, or ids that leave a gap.
    /// A repeated merge keeps its first (lowest) rank.
    pub fn new(
        vocab: Vec<(Vec<u8>, u32)>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        added: Vec<AddedToken>,
        pretok: PreTokenizer,
    ) -> Result<ByteBpe, LoadError> {
        // SOLUTION-BEGIN L1.5
        let size = vocab
            .iter()
            .map(|(_, id)| *id)
            .chain(added.iter().map(|a| a.id))
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut tokens: Vec<Option<Vec<u8>>> = vec![None; size];
        let mut map = RobinHoodMap::with_capacity(vocab.len());
        for (bytes, id) in vocab {
            if let Some(prev) = &tokens[id as usize] {
                if *prev != bytes {
                    return Err(LoadError::Format(format!("id {id} names two tokens")));
                }
            }
            tokens[id as usize] = Some(bytes.clone());
            map.insert(bytes, id);
        }
        for a in &added {
            match &tokens[a.id as usize] {
                Some(b) if *b != a.content.as_bytes() => {
                    return Err(LoadError::Format(format!(
                        "added token {:?} has id {}, which the vocabulary gives to another token",
                        a.content, a.id
                    )))
                }
                _ => tokens[a.id as usize] = Some(a.content.as_bytes().to_vec()),
            }
        }
        let tokens: Vec<Vec<u8>> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, t)| t.ok_or_else(|| LoadError::Format(format!("no token has id {i}: ids leave a gap"))))
            .collect::<Result<_, _>>()?;
        let mut table = RobinHoodMap::with_capacity(merges.len());
        for (rank, (l, r)) in merges.into_iter().enumerate() {
            let lid = map.get(&l).copied();
            let rid = map.get(&r).copied();
            let mut joined = l.clone();
            joined.extend_from_slice(&r);
            let id = map.get(&joined).copied();
            match (lid, rid, id) {
                (Some(a), Some(b), Some(id)) => {
                    table.entry((a, b)).or_insert(Merge { rank: rank as u32, id });
                }
                _ => {
                    return Err(LoadError::Format(format!(
                        "merge {rank} ({:?} {:?}) uses a token that is not in the vocabulary",
                        String::from_utf8_lossy(&l),
                        String::from_utf8_lossy(&r)
                    )))
                }
            }
        }
        let mut byte_ids = [None; 256];
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            *slot = map.get(&[b as u8][..]).copied();
        }
        Ok(ByteBpe { vocab: map, tokens, merges: table, byte_ids, added, pretok })
        // SOLUTION-END
    }

    /// The id of a token given by its bytes.
    pub fn token_to_id(&self, bytes: &[u8]) -> Option<u32> {
        // SOLUTION-BEGIN L1.5
        self.vocab.get(bytes).copied()
        // SOLUTION-END
    }

    /// The merge of the adjacent ids `(left, right)`, if any.
    pub fn merge(&self, left: u32, right: u32) -> Option<Merge> {
        // SOLUTION-BEGIN L1.5
        self.merges.get(&(left, right)).copied()
        // SOLUTION-END
    }

    pub fn merge_count(&self) -> usize {
        // SOLUTION-BEGIN L1.5
        self.merges.len()
        // SOLUTION-END
    }

    pub fn added_tokens(&self) -> &[AddedToken] {
        // SOLUTION-BEGIN L1.5
        &self.added
        // SOLUTION-END
    }

    pub fn pre_tokenizer(&self) -> &PreTokenizer {
        // SOLUTION-BEGIN L1.5
        &self.pretok
        // SOLUTION-END
    }

    /// Appends the ids of one pre-token (its UTF-8 bytes) to `out`, merging
    /// from a lazy heap.
    pub fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        // SOLUTION-BEGIN L1.5
        let mut syms: Vec<Sym> = Vec::with_capacity(piece.len());
        for &b in piece {
            if let Some(id) = self.byte_ids[b as usize] {
                let i = syms.len();
                syms.push(Sym { id, prev: if i == 0 { NONE } else { i - 1 }, next: i + 1 });
            }
        }
        let n = syms.len();
        if n == 0 {
            return;
        }
        syms[n - 1].next = NONE;
        if n > 1 {
            // (rank, left position): the lowest rank first, then the leftmost.
            let by_rank = |a: &(u32, usize), b: &(u32, usize)| -> Ordering { a.cmp(b) };
            let mut heap = LazyHeap::new(by_rank);
            let mut pair: Vec<Option<Handle>> = vec![None; n];
            for i in 0..n - 1 {
                if let Some(m) = self.merge(syms[i].id, syms[i + 1].id) {
                    pair[i] = Some(heap.push((m.rank, i)));
                }
            }
            while let Some((_, i)) = heap.pop() {
                pair[i] = None;
                let j = syms[i].next;
                let m = self.merge(syms[i].id, syms[j].id).expect("a live pair always merges");
                // The pairs (prev, i) and (j, next) disappear with this merge.
                let p = syms[i].prev;
                if p != NONE {
                    if let Some(h) = pair[p].take() {
                        heap.remove(h);
                    }
                }
                if let Some(h) = pair[j].take() {
                    heap.remove(h);
                }
                // i absorbs j.
                let nx = syms[j].next;
                syms[i].id = m.id;
                syms[i].next = nx;
                if nx != NONE {
                    syms[nx].prev = i;
                }
                // The pairs (prev, i) and (i, next) are new.
                if p != NONE {
                    if let Some(mm) = self.merge(syms[p].id, syms[i].id) {
                        pair[p] = Some(heap.push((mm.rank, p)));
                    }
                }
                if nx != NONE {
                    if let Some(mm) = self.merge(syms[i].id, syms[nx].id) {
                        pair[i] = Some(heap.push((mm.rank, i)));
                    }
                }
            }
        }
        let mut i = 0;
        while i != NONE {
            out.push(syms[i].id);
            i = syms[i].next;
        }
        // SOLUTION-END
    }

    /// Ids of `text` with no added-token matching: pre-tokenize, then merge.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        // SOLUTION-BEGIN L1.5
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        for piece in self.pretok.split(text) {
            self.encode_piece(piece.as_bytes(), &mut out);
        }
        out
        // SOLUTION-END
    }

    /// The allowed added token that matches leftmost-longest in `text`:
    /// `(byte start, byte length, id)`.
    fn next_added(&self, text: &str, allowed: &SpecialSet) -> Option<(usize, usize, u32)> {
        // SOLUTION-BEGIN L1.5
        let mut best: Option<(usize, usize, u32)> = None;
        for a in &self.added {
            if a.content.is_empty() || !allowed.allows(&a.content) {
                continue;
            }
            if let Some(start) = text.find(a.content.as_str()) {
                let len = a.content.len();
                let better = match best {
                    None => true,
                    Some((s, l, _)) => start < s || (start == s && len > l),
                };
                if better {
                    best = Some((start, len, a.id));
                }
            }
        }
        best
        // SOLUTION-END
    }

    /// `encode` of each text, in order, on `threads` OS threads (0 means the
    /// hardware concurrency). The result does not depend on `threads`.
    pub fn encode_batch(&self, texts: &[&str], threads: usize) -> Vec<Vec<u32>> {
        // SOLUTION-BEGIN L1.5
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads
        };
        let threads = threads.min(texts.len()).max(1);
        if threads == 1 {
            return texts.iter().map(|t| self.encode(t)).collect();
        }
        let chunk = texts.len().div_ceil(threads);
        std::thread::scope(|s| {
            let workers: Vec<_> = texts
                .chunks(chunk)
                .map(|part| s.spawn(move || part.iter().map(|t| self.encode(t)).collect::<Vec<_>>()))
                .collect();
            workers.into_iter().flat_map(|w| w.join().expect("an encode worker panicked")).collect()
        })
        // SOLUTION-END
    }
}

impl Tokenizer for ByteBpe {
    fn encode(&self, text: &str) -> Vec<u32> {
        // SOLUTION-BEGIN L1.5
        self.encode_with_special(text, &SpecialSet::all())
        // SOLUTION-END
    }

    fn encode_with_special(&self, text: &str, allowed: &SpecialSet) -> Vec<u32> {
        // SOLUTION-BEGIN L1.5
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        let mut rest = text;
        while let Some((start, len, id)) = self.next_added(rest, allowed) {
            for piece in self.pretok.split(&rest[..start]) {
                self.encode_piece(piece.as_bytes(), &mut out);
            }
            out.push(id);
            rest = &rest[start + len..];
        }
        for piece in self.pretok.split(rest) {
            self.encode_piece(piece.as_bytes(), &mut out);
        }
        out
        // SOLUTION-END
    }

    fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        // SOLUTION-BEGIN L1.5
        self.tokens.get(id as usize).map(|t| t.as_slice())
        // SOLUTION-END
    }

    fn vocab_size(&self) -> u32 {
        // SOLUTION-BEGIN L1.5
        self.tokens.len() as u32
        // SOLUTION-END
    }
}
