# Part 1: Tokenizers

From bytes to subwords. The tracer's byte tokenizer (256 ids, no training) gives way to the `Tokenizer` protocol and three trained families: byte-level BPE (GPT-2 compatible, loading Hugging Face files), WordPiece (BERT), and the Unigram LM (EM, Viterbi, subword sampling). Then BPE moves to Rust (`tl-tok`) with a streaming UTF-8 decoder and a PyO3 binding, proven byte-for-byte against your Python, and the metrics that decide a vocabulary size for the capstone.

**Course passes**: 3 (L1.1 to L1.6, milestone MS-L1, part of gate MS-P3)

**Before you start**: the just-in-time math `M05.2`, `M06.2`, `M07.1`, `M07.2`, `M11.2` and the solve parts `S-M06b`, `S-M07b`; for `L1.5`, the [Rust primer](../../../software-craftsmanship/12-language-and-tool-primers/04-rust.md) and the Rust structures `ds.05` and `ds.06` in [Systems Data Structures](../../../algorithms/16-systems-data-structures/).

## Key ideas

- **A tokenizer is a contract**: `encode(decode(ids)) == ids` and `decode(encode(text)) == text` for every string, including invalid UTF-8 split across a stream.
- **BPE** merges the most frequent adjacent pair until the vocabulary is full; encoding replays the merges by rank. A heap over pairs (`ds.06`) makes training fast.
- **Unigram** starts from a large vocabulary and prunes by likelihood loss; it can sample segmentations, which the capstone ablation (BPE vs Unigram) measures.
- **Compression is the first metric**: bytes per token and bits per byte on held-out text (`L1.6`).

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L1.1` | Tokenizer protocol, char tokenizer | build | 3 |
| `L1.2` | Byte-level BPE (GPT-2 compatible): hand-written pre-tokenizer, trainer, HF loader | build | 3 |
| `L1.3` | WordPiece (BERT basic tokenizer + greedy longest match) | build | 3 |
| `L1.4` | Unigram LM tokenizer (EM, Viterbi, subword sampling) | build | 3 |
| `L1.5` | Rust fast BPE (`tl-tok`) with a streaming UTF-8 decoder, the byte tokenizer, and PyO3 binding (`tl-py`) | build | 3 |
| `L1.6` | Tokenizer metrics | build | 3 |

## Chapters

<!-- ss:chapters -->
No chapters yet: they arrive with authoring batch B4 (course/DESIGN.md 9). `ss lint --fix-index` then fills this table from the registry.
<!-- /ss:chapters -->

## Going further

- Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) (Rust BPE merges and the pre-tokenizer pipeline) and OpenAI [tiktoken](https://github.com/openai/tiktoken).
- Sennrich, Haddow, and Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909); Kudo, [*Subword Regularization*](https://arxiv.org/abs/1804.10959).
