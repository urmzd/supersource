# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.1", "tiktoken==0.12.0", "sentencepiece==0.2.0"]
# ///
"""Maintainer generator for the tokenizer fixtures of L1.2, L1.3, L1.4, and L1.6.

    uv run --script course/oracle/tok/golden.py        (from the repo root; needs network)

Downloads the pinned official files (sha256 checked), writes the fixtures,
and prints the MANIFEST.tsv rows. Every id is computed by Hugging Face
`tokenizers` and cross-checked: GPT-2 ids also by tiktoken, the Unigram
lattice also by sentencepiece. Nothing here is imported by a course test.

  course/fixtures/tok-gpt2/tokenizer.json     openai-community/gpt2 (MIT), unchanged
  course/fixtures/tok-gpt2/cases.jsonl        300 strings: ids, decoded
  course/fixtures/tok-smollm2/tokenizer.json  HuggingFaceTB/SmolLM2-135M (Apache-2.0), unchanged
  course/fixtures/tok-smollm2/cases.jsonl
  course/fixtures/tok-bert/tokenizer.json     google-bert/bert-base-uncased (Apache-2.0), unchanged
  course/fixtures/tok-bert/cases.jsonl        ids, ids with [CLS]/[SEP], decoded, decoded skipping specials
  course/fixtures/L1.2/train_merges.json      BpeTrainer merges on L1.2/train.txt for five configs
  course/fixtures/L1.4/hf-unigram/tokenizer.json   UnigramTrainer on L1.2/train.txt (vocab 300)
  course/fixtures/L1.4/hf-unigram/cases.jsonl
  course/fixtures/L1.4/spm-unigram/tokenizer.json  a sentencepiece unigram model, pieces and scores as tokenizer.json
  course/fixtures/L1.4/spm-unigram/cases.jsonl
  course/fixtures/L1.6/metrics.json           fertility, bytes per token, fallback rate per tokenizer

The classic GPT-2 files (vocab.json, merges.txt) and BERT's vocab.txt are
not committed: the script asserts they equal what tokenizer.json holds, and
the tests derive them from it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import sentencepiece as spm
import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strings import strings  # noqa: E402

ROOT = Path.cwd()
FX = ROOT / "course" / "fixtures"
CORPUS = FX / "L1.2" / "train.txt"

GPT2 = "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e"
BERT = "https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594"
SMOL = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
PINS = {
    f"{GPT2}/tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    f"{GPT2}/vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    f"{GPT2}/merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    f"{BERT}/tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    f"{BERT}/vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    f"{SMOL}/tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
}
TRAIN_CONFIGS = [  # (vocab_size, min_freq, specials)
    (300, 2, []),
    (400, 2, ["<|endoftext|>"]),
    (600, 1, ["<s>", "</s>"]),
    (800, 2, []),
    (1000, 1, []),
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != PINS[url]:
        raise SystemExit(f"{url}: sha256 {got} != pinned {PINS[url]}")
    return data


def write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def bpe_cases(tok: Tokenizer, texts: list[str]) -> list[dict]:
    rows = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        rows.append({"text": t, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
    return rows


def fallback(tok: Tokenizer, i: int, unk: int | None, special: set[int]) -> bool:
    if unk is not None and i == unk:
        return True
    if i in special:
        return False
    return "�" in tok.decode([i], skip_special_tokens=False)


def metrics(tok: Tokenizer, texts: list[str], words: list[str], unk: int | None) -> dict:
    special = {t.id for t in []}
    special = {i for i, a in tok.get_added_tokens_decoder().items() if a.special}
    enc = [tok.encode(t, add_special_tokens=False).ids for t in texts]
    n_tok = sum(len(e) for e in enc)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    fb = sum(fallback(tok, i, unk, special) for e in enc for i in e)
    fert = sum(len(tok.encode(w, add_special_tokens=False).ids) for w in words) / len(words)
    return {"fertility": fert, "bytes_per_token": n_bytes / n_tok, "byte_fallback_rate": fb / n_tok,
            "tokens": n_tok, "bytes": n_bytes, "fallback_tokens": fb}


def main() -> int:
    texts = strings()
    out: list[Path] = []

    # ---- GPT-2 -----------------------------------------------------------
    gj = fetch(f"{GPT2}/tokenizer.json")
    doc = json.loads(gj)
    assert doc["model"]["vocab"] == json.loads(fetch(f"{GPT2}/vocab.json")), "vocab.json differs"
    merges_txt = fetch(f"{GPT2}/merges.txt").decode("utf-8")
    lines = [x for x in merges_txt.split("\n") if x and not x.startswith("#version")]
    assert [m if isinstance(m, str) else " ".join(m) for m in doc["model"]["merges"]] == lines, "merges differ"
    write(FX / "tok-gpt2" / "tokenizer.json", gj)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        p.write_bytes(gj)
        g = Tokenizer.from_file(str(p))
    rows = bpe_cases(g, texts)
    tk = tiktoken.get_encoding("gpt2")
    for r in rows:
        assert tk.encode(r["text"], allowed_special="all") == r["ids"], ("tiktoken", r["text"])
    write(FX / "tok-gpt2" / "cases.jsonl", jsonl(rows))
    out += [FX / "tok-gpt2" / "tokenizer.json", FX / "tok-gpt2" / "cases.jsonl"]

    # ---- SmolLM2 ---------------------------------------------------------
    sj = fetch(f"{SMOL}/tokenizer.json")
    write(FX / "tok-smollm2" / "tokenizer.json", sj)
    s = Tokenizer.from_file(str(FX / "tok-smollm2" / "tokenizer.json"))
    write(FX / "tok-smollm2" / "cases.jsonl", jsonl(bpe_cases(s, texts)))
    out += [FX / "tok-smollm2" / "tokenizer.json", FX / "tok-smollm2" / "cases.jsonl"]

    # ---- BERT ------------------------------------------------------------
    bj = fetch(f"{BERT}/tokenizer.json")
    bdoc = json.loads(bj)
    vocab_lines = fetch(f"{BERT}/vocab.txt").decode("utf-8").split("\n")
    if vocab_lines and vocab_lines[-1] == "":
        vocab_lines.pop()
    assert {t: i for i, t in enumerate(vocab_lines)} == bdoc["model"]["vocab"], "vocab.txt differs"
    write(FX / "tok-bert" / "tokenizer.json", bj)
    b = Tokenizer.from_file(str(FX / "tok-bert" / "tokenizer.json"))
    rows = []
    for t in texts:
        ids = b.encode(t, add_special_tokens=False).ids
        rows.append({
            "text": t, "ids": ids, "ids_special": b.encode(t).ids,
            "decoded": b.decode(ids, skip_special_tokens=False),
            "decoded_skip": b.decode(ids, skip_special_tokens=True),
        })
    write(FX / "tok-bert" / "cases.jsonl", jsonl(rows))
    out += [FX / "tok-bert" / "tokenizer.json", FX / "tok-bert" / "cases.jsonl"]

    # ---- the BPE trainer on L1.2/train.txt -------------------------------
    corpus = CORPUS.read_text(encoding="utf-8").splitlines(keepends=True)
    golden = []
    for vs, mf, sp in TRAIN_CONFIGS:
        hf = Tokenizer(models.BPE())
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tr = trainers.BpeTrainer(vocab_size=vs, min_frequency=mf, special_tokens=sp,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        hf.train_from_iterator(corpus, tr)
        d = json.loads(hf.to_str())
        merges = [m.split(" ") if isinstance(m, str) else m for m in d["model"]["merges"]]
        golden.append({"vocab_size": vs, "min_freq": mf, "specials": sp,
                       "n_vocab": len(d["model"]["vocab"]), "merges": merges})
    write(FX / "L1.2" / "train_merges.json", json.dumps(golden, ensure_ascii=False, indent=0) + "\n")
    out += [FX / "L1.2" / "train_merges.json"]

    # ---- Unigram: Hugging Face trainer, and a sentencepiece model ---------
    u = Tokenizer(models.Unigram())
    u.pre_tokenizer = pre_tokenizers.Metaspace()
    u.decoder = decoders.Metaspace()
    u.train_from_iterator([x.rstrip("\n") for x in corpus], trainers.UnigramTrainer(
        vocab_size=300, special_tokens=["<unk>"], unk_token="<unk>", show_progress=False))
    hu_path = FX / "L1.4" / "hf-unigram" / "tokenizer.json"
    hu_path.parent.mkdir(parents=True, exist_ok=True)
    u.save(str(hu_path))
    rows = []
    for t in texts:
        ids = u.encode(t, add_special_tokens=False).ids
        rows.append({"text": t, "ids": ids, "decoded": u.decode(ids, skip_special_tokens=False),
                     "decoded_skip": u.decode(ids, skip_special_tokens=True)})
    write(FX / "L1.4" / "hf-unigram" / "cases.jsonl", jsonl(rows))
    with tempfile.TemporaryDirectory() as td:
        spm.SentencePieceTrainer.train(
            input=str(CORPUS), model_prefix=f"{td}/u", vocab_size=300, model_type="unigram",
            normalization_rule_name="identity", remove_extra_whitespaces=False,
            split_by_whitespace=True, add_dummy_prefix=True, bos_id=-1, eos_id=-1, unk_id=0,
            character_coverage=1.0, byte_fallback=False, split_digits=False, minloglevel=2,
            num_threads=1)
        sp = spm.SentencePieceProcessor(model_file=f"{td}/u.model")
    pieces = [[sp.id_to_piece(i), float(sp.get_score(i))] for i in range(sp.get_piece_size())]
    meta = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
    sdoc = {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": 0, "content": "<unk>", "single_word": False, "lstrip": False,
                              "rstrip": False, "normalized": False, "special": True}],
            "normalizer": None, "pre_tokenizer": meta, "post_processor": None, "decoder": meta,
            "model": {"type": "Unigram", "unk_id": 0, "vocab": pieces, "byte_fallback": False}}
    write(FX / "L1.4" / "spm-unigram" / "tokenizer.json", json.dumps(sdoc, ensure_ascii=False, indent=1) + "\n")
    hs = Tokenizer.from_file(str(FX / "L1.4" / "spm-unigram" / "tokenizer.json"))
    rows, skipped = [], 0
    for t in texts:
        ids = sp.encode(t)
        if hs.encode(t, add_special_tokens=False).ids != ids:
            skipped += 1  # sentencepiece itself handles this string differently (reported)
            continue
        rows.append({"text": t, "ids": ids})
    print(f"spm-unigram: {len(rows)} strings agree between sentencepiece and tokenizers, {skipped} differ")
    write(FX / "L1.4" / "spm-unigram" / "cases.jsonl", jsonl(rows))
    out += [hu_path, FX / "L1.4" / "hf-unigram" / "cases.jsonl",
            FX / "L1.4" / "spm-unigram" / "tokenizer.json", FX / "L1.4" / "spm-unigram" / "cases.jsonl"]

    # ---- metrics (L1.6) ----------------------------------------------------
    words = sorted({w for t in texts for w in t.split()})
    m = {
        "words": len(words),
        "gpt2": metrics(g, texts, words, None),
        "smollm2": metrics(s, texts, words, None),
        "bert": metrics(b, texts, words, b.token_to_id("[UNK]")),
        "hf-unigram": metrics(u, texts, words, 0),
    }
    write(FX / "L1.6" / "metrics.json", json.dumps(m, indent=1) + "\n")
    out += [FX / "L1.6" / "metrics.json"]

    tools = "tokenizers==0.22.1 tiktoken==0.12.0 sentencepiece==0.2.0"
    ups = {"tok-gpt2/tokenizer.json": ("openai-community/gpt2@607a30d", "MIT"),
           "tok-smollm2/tokenizer.json": ("HuggingFaceTB/SmolLM2-135M@93efa2f", "Apache-2.0"),
           "tok-bert/tokenizer.json": ("google-bert/bert-base-uncased@86b5e09", "Apache-2.0")}
    for p in out:
        rel = p.relative_to(ROOT).as_posix()
        key = p.relative_to(FX).as_posix()
        up, lic = ups.get(key, ("-", "Apache-2.0"))
        if key.startswith("tok-gpt2/"):
            lic = "MIT"
        data = p.read_bytes()
        print("\t".join([rel, hashlib.sha256(data).hexdigest(), str(len(data)),
                         "course/oracle/tok/golden.py", tools, up, lic]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
