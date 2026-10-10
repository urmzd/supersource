# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.2.6", "tokenizers==0.23.3"]
# ///
"""Maintainer generator for the C1 reference artifacts at the smoke tier.

The capstone's full run (about 10.4M parameters on TinyStories, hours on a
laptop) and its `short` ablations are author-local (DESIGN 9, B11). What the
course commits is the SMOKE tier the milestone's --smoke mode runs: the same
pipeline and the same report, on the committed synthetic stories
(course/fixtures/MS-corpus/sources/stories.jsonl: the first 130 documents
train, the last 32 are the held-out set), with a model of about 0.8M
parameters trained for 200 steps. Every number in the report is measured
here; nothing is typed in.

    uv run --offline --script course/oracle/C1/smoke_capstone.py

Writes (paths in the reference learner repo):

    course/ref/entry/specs/c1/smoke.json       the train spec of the main run
    course/ref/docs/capstone/c1/report.json    tier, model, train, eval, ablations, scaling
    course/ref/docs/capstone/c1/zoo.json       formats/eval-results.schema.json, suite zoo
    course/ref/docs/capstone/c1/samples.jsonl  20 seeded samples of the main model

What runs: a byte-level BPE and a Unigram tokenizer (HF tokenizers, vocab
512) on the training text; the main Llama (GQA, dense SwiGLU, tied); three
ablations at the main config, each one 200-step run per arm with seed 0 and
paired per-document bits per byte on the held-out set with a seeded
bootstrap 95% CI (tokenizer: BPE vs Unigram; attention: GQA vs MLA at equal
KV bytes; MLP: dense vs a top-1 MoE of 4 experts at equal active
parameters); a power-law fit bpb = a * params^-alpha by least squares in
log-log space over three sizes; and the zoo baselines (an interpolated
Kneser-Ney 4-gram, an NPLM, an LSTM, a GPT, and the Llama) on the same
held-out documents.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SRC = Path("course/fixtures/MS-corpus/sources/stories.jsonl")
SPEC_OUT = Path("course/ref/entry/specs/c1")
DOC_OUT = Path("course/ref/docs/capstone/c1")
SEED = 0
VOCAB, CTX, STEPS, BATCH, LR = 512, 128, 200, 16, 3e-3
MAIN = dict(d=128, layers=4, heads=4, kv=2, ff=344)

torch.set_num_threads(1)
docs = [json.loads(line)["text"] for line in SRC.read_text().splitlines() if line.strip()]
train_docs, val_docs = docs[:130], docs[130:]
train_text = "\n".join(train_docs)


# -- tokenizers -------------------------------------------------------------
def make_tokenizer(kind: str) -> Tokenizer:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if kind == "bpe":
        tok = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=VOCAB, initial_alphabet=alphabet, show_progress=False)
    else:
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=VOCAB, initial_alphabet=alphabet, show_progress=False,
                                          special_tokens=[], max_piece_length=12)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(train_docs, trainer)
    return tok


# -- the models -------------------------------------------------------------
def rmsnorm(x, g, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g


def rope(x, theta=10000.0):  # x [B, H, T, hd], half layout
    T, hd = x.shape[-2], x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    return x * cos + torch.cat([-x[..., hd // 2 :], x[..., : hd // 2]], -1) * sin


class Llama(torch.nn.Module):
    """config: d, layers, heads, kv, ff, vocab, attention gqa|mla (rank, rope_dim, nope_dim, v_dim), experts, top_k."""

    def __init__(self, c: dict):
        super().__init__()
        self.c = c
        d, H, V = c["d"], c["heads"], c["vocab"]
        self.hd = d // H
        g = torch.Generator().manual_seed(c.get("init_seed", SEED))
        P = torch.nn.ParameterDict()

        def w(name, *shape, std=0.02):
            P[name] = torch.nn.Parameter(torch.randn(*shape, generator=g) * std)

        def one(name, n):
            P[name] = torch.nn.Parameter(torch.ones(n))

        w("embed", V, d)
        for i in range(c["layers"]):
            p = f"l{i}_"
            one(p + "g_attn", d)
            one(p + "g_mlp", d)
            if c.get("attention", "gqa") == "gqa":
                w(p + "wq", H * self.hd, d)
                w(p + "wk", c["kv"] * self.hd, d)
                w(p + "wv", c["kv"] * self.hd, d)
                w(p + "wo", d, H * self.hd, std=0.02 / math.sqrt(2 * c["layers"]))
            else:
                r, dr, dn, dv = c["rank"], c["rope_dim"], c["nope_dim"], c["v_dim"]
                w(p + "wq", H * (dn + dr), d)
                w(p + "w_kv_a", r + dr, d)
                one(p + "g_kv", r)
                w(p + "w_kv_b", H * (dn + dv), r)
                w(p + "wo", d, H * dv, std=0.02 / math.sqrt(2 * c["layers"]))
            E = c.get("experts", 0)
            if E:
                w(p + "router", E, d)
                for e in range(E):
                    w(p + f"e{e}_gate", c["ff"], d)
                    w(p + f"e{e}_up", c["ff"], d)
                    w(p + f"e{e}_down", d, c["ff"], std=0.02 / math.sqrt(2 * c["layers"]))
            else:
                w(p + "gate", c["ff"], d)
                w(p + "up", c["ff"], d)
                w(p + "down", d, c["ff"], std=0.02 / math.sqrt(2 * c["layers"]))
        one("g_final", d)
        self.P = P

    def attn(self, p, h):
        c, P = self.c, self.P
        B, T, d = h.shape
        H = c["heads"]
        mask = torch.full((T, T), float("-inf")).triu(1)
        if c.get("attention", "gqa") == "gqa":
            q = (h @ P[p + "wq"].T).view(B, T, H, self.hd).transpose(1, 2)
            k = (h @ P[p + "wk"].T).view(B, T, c["kv"], self.hd).transpose(1, 2)
            v = (h @ P[p + "wv"].T).view(B, T, c["kv"], self.hd).transpose(1, 2)
            q, k = rope(q), rope(k)
            rep = H // c["kv"]
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd) + mask, -1)
            return (att @ v).transpose(1, 2).reshape(B, T, H * self.hd) @ P[p + "wo"].T
        r, dr, dn, dv = c["rank"], c["rope_dim"], c["nope_dim"], c["v_dim"]
        q = (h @ P[p + "wq"].T).view(B, T, H, dn + dr).transpose(1, 2)
        q_nope, q_rope = q[..., :dn], rope(q[..., dn:])
        a = h @ P[p + "w_kv_a"].T
        ckv, k_rope = rmsnorm(a[..., :r], P[p + "g_kv"]), rope(a[..., r:].unsqueeze(1))  # k_rope shared by heads
        kv = (ckv @ P[p + "w_kv_b"].T).view(B, T, H, dn + dv).transpose(1, 2)
        k_nope, v = kv[..., :dn], kv[..., dn:]
        s = (q_nope @ k_nope.transpose(-1, -2) + q_rope @ k_rope.transpose(-1, -2)) / math.sqrt(dn + dr)
        att = torch.softmax(s + mask, -1)
        return (att @ v).transpose(1, 2).reshape(B, T, H * dv) @ P[p + "wo"].T

    def mlp(self, p, h):
        P = self.P

        def swiglu(q):
            return (torch.nn.functional.silu(h @ P[q + "gate"].T) * (h @ P[q + "up"].T)) @ P[q + "down"].T

        E = self.c.get("experts", 0)
        if not E:
            return swiglu(p), 0.0
        logits = h @ P[p + "router"].T  # [B, T, E]
        probs = torch.softmax(logits, -1)
        top = probs.argmax(-1)  # top-1
        out = torch.zeros_like(h)
        for e in range(E):
            sel = (top == e).unsqueeze(-1).float()
            out = out + sel * probs[..., e : e + 1] * swiglu(p + f"e{e}_")
        frac = torch.stack([(top == e).float().mean() for e in range(E)])
        aux = E * (frac * probs.mean((0, 1))).sum()  # Switch load balancing
        return out, aux

    def forward(self, ids):
        P = self.P
        x = P["embed"][ids]
        aux = 0.0
        for i in range(self.c["layers"]):
            p = f"l{i}_"
            x = x + self.attn(p, rmsnorm(x, P[p + "g_attn"]))
            m, a = self.mlp(p, rmsnorm(x, P[p + "g_mlp"]))
            x, aux = x + m, aux + a
        return rmsnorm(x, P["g_final"]) @ P["embed"].T, aux


class GPT(torch.nn.Module):
    def __init__(self, V, d=128, layers=4, heads=4):
        super().__init__()
        torch.manual_seed(SEED)
        self.wte, self.wpe = torch.nn.Embedding(V, d), torch.nn.Embedding(CTX, d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = torch.nn.LayerNorm(d)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.wte(ids) + self.wpe(torch.arange(T))
        x = self.blocks(x, mask=torch.nn.Transformer.generate_square_subsequent_mask(T), is_causal=True)
        return self.ln(x) @ self.wte.weight.T, 0.0


class LSTMLM(torch.nn.Module):
    def __init__(self, V, d=128, hidden=256):
        super().__init__()
        torch.manual_seed(SEED)
        self.emb, self.rnn, self.out = torch.nn.Embedding(V, d), torch.nn.LSTM(d, hidden, batch_first=True), torch.nn.Linear(hidden, V)

    def forward(self, ids):
        return self.out(self.rnn(self.emb(ids))[0]), 0.0


class NPLM(torch.nn.Module):
    def __init__(self, V, n=4, d=64, hidden=256):
        super().__init__()
        torch.manual_seed(SEED)
        self.n, self.emb = n, torch.nn.Embedding(V + 1, d)  # V: padding before the start
        self.h, self.out = torch.nn.Linear(n * d, hidden), torch.nn.Linear(hidden, V)
        self.V = V

    def forward(self, ids):
        B, T = ids.shape
        pad = torch.full((B, self.n - 1), self.V, dtype=ids.dtype)
        x = torch.cat([pad, ids], 1)
        ctx = torch.stack([x[:, i : i + T] for i in range(self.n)], -1)  # the last n tokens up to t
        e = self.emb(ctx).reshape(B, T, -1)
        return self.out(torch.tanh(self.h(e))), 0.0


def n_params(c: dict) -> tuple[int, int]:
    """(total, active) parameters of a Llama config, the formula the C1 check uses."""
    d, H, V, L = c["d"], c["heads"], c["vocab"], c["layers"]
    hd = d // H
    if c.get("attention", "gqa") == "gqa":
        attn = d * H * hd + 2 * d * c["kv"] * hd + H * hd * d
    else:
        r, dr, dn, dv = c["rank"], c["rope_dim"], c["nope_dim"], c["v_dim"]
        attn = d * H * (dn + dr) + d * (r + dr) + r + r * H * (dn + dv) + H * dv * d
    E, k = c.get("experts", 0), c.get("top_k", 1)
    mlp_total = (d * E + E * 3 * d * c["ff"]) if E else 3 * d * c["ff"]
    mlp_active = (d * E + k * 3 * d * c["ff"]) if E else mlp_total
    base = V * d + d + L * (attn + 2 * d)
    return base + L * mlp_total, base + L * mlp_active


def kv_bytes(c: dict) -> int:
    """KV cache bytes per token in bf16: K and V per layer (GQA), or c_kv plus the shared rope key (MLA)."""
    if c.get("attention", "gqa") == "gqa":
        return c["layers"] * 2 * c["kv"] * (c["d"] // c["heads"]) * 2
    return c["layers"] * (c["rank"] + c["rope_dim"]) * 2


# -- training and evaluation -------------------------------------------------
def stream(tok: Tokenizer) -> torch.Tensor:
    return torch.tensor(tok.encode("\n".join(train_docs)).ids)


def train(model: torch.nn.Module, data: torch.Tensor, steps: int = STEPS) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(SEED)
    warm, decay_from = 20, int(steps * 0.8)
    t0, last = time.time(), []
    for step in range(steps):  # WSD: linear warmup, stable, linear decay to 0.1
        lr = LR * min(1.0, (step + 1) / warm)
        if step >= decay_from:
            lr = LR * (1 - 0.9 * (step - decay_from) / max(1, steps - decay_from))
        for gr in opt.param_groups:
            gr["lr"] = lr
        ix = torch.randint(0, len(data) - CTX - 1, (BATCH,), generator=gen)
        x = torch.stack([data[i : i + CTX] for i in ix])
        y = torch.stack([data[i + 1 : i + CTX + 1] for i in ix])
        logits, aux = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        (loss + 0.01 * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last.append(loss.item())
    return {"steps": steps, "tokens": steps * BATCH * CTX, "final_train_loss": float(np.mean(last[-20:])),
            "wall_s": round(time.time() - t0, 1)}


def tok_bytes(tok: Tokenizer, i: int) -> int:
    return len(tok.decode([i]).encode())


@torch.no_grad()
def doc_bits(model, tok: Tokenizer) -> list[tuple[float, int, int]]:
    """(bits, bytes, tokens predicted) per held-out document, in chunks of CTX
    tokens; the first token of each chunk is not predicted and its bytes are
    not counted."""
    model.eval()
    out = []
    for doc in val_docs:
        ids = tok.encode(doc).ids
        bits = nbytes = n = 0
        for s in range(0, len(ids), CTX):
            chunk = ids[s : s + CTX]
            if len(chunk) < 2:
                continue
            lp = torch.log_softmax(model(torch.tensor([chunk]))[0][0], -1)
            bits += -sum(lp[t, chunk[t + 1]].item() for t in range(len(chunk) - 1)) / math.log(2)
            nbytes += sum(tok_bytes(tok, i) for i in chunk[1:])
            n += len(chunk) - 1
        out.append((bits, nbytes, n))
    model.train()
    return out


def pooled(rows) -> float:
    return sum(r[0] for r in rows) / sum(r[1] for r in rows)


def bootstrap_ci(values: np.ndarray, seed: int = SEED, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pooled_ci(rows, seed: int = SEED, n_boot: int = 2000) -> tuple[float, float]:
    bits = np.array([r[0] for r in rows])
    by = np.array([r[1] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    v = bits[idx].sum(1) / by[idx].sum(1)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def hf_config(c: dict) -> dict:
    out = {"tl_arch": "llama", "tl_tokenizer": "file", "vocab_size": c["vocab"], "hidden_size": c["d"],
           "intermediate_size": c["ff"], "num_hidden_layers": c["layers"], "num_attention_heads": c["heads"],
           "num_key_value_heads": c["kv"], "max_position_embeddings": CTX, "rms_norm_eps": 1e-05,
           "rope_theta": 10000.0, "tie_word_embeddings": True, "tl_format": 1}
    if c.get("attention", "gqa") == "mla":
        out.update({"tl_attention": "mla", "tl_mla_rank": c["rank"], "qk_rope_head_dim": c["rope_dim"],
                    "qk_nope_head_dim": c["nope_dim"], "v_head_dim": c["v_dim"]})
    if c.get("experts", 0):
        out.update({"tl_num_experts": c["experts"], "tl_top_k_experts": c.get("top_k", 1),
                    "moe_intermediate_size": c["ff"], "norm_topk_prob": False})
    return out


def run_llama(c: dict, tok: Tokenizer, data: torch.Tensor) -> tuple[Llama, dict, list]:
    m = Llama(c)
    total = sum(p.numel() for p in m.parameters())
    assert total == n_params(c)[0], (total, n_params(c))
    info = train(m, data)
    return m, info, doc_bits(m, tok)


def ablation(aid, question, a_name, a_cfg, a_rows, b_name, b_cfg, b_rows, extra) -> dict:
    da = np.array([r[0] / r[1] for r in a_rows])
    db = np.array([r[0] / r[1] for r in b_rows])
    diff = db - da
    lo, hi = bootstrap_ci(diff)
    winner = "b" if hi < 0 else "a" if lo > 0 else "tie"
    return {"id": aid, "question": question, "metric": "bpb", "unit": "held-out document", "n": int(len(diff)),
            "seed": SEED, "a": {"name": a_name, "config": a_cfg, "value": round(pooled(a_rows), 6)},
            "b": {"name": b_name, "config": b_cfg, "value": round(pooled(b_rows), 6)},
            "mean_diff": round(float(diff.mean()), 6), "ci95": [round(lo, 6), round(hi, 6)], "winner": winner, **extra}


def kn4_rows(tok: Tokenizer, n: int = 4, D: float = 0.75) -> list[tuple[float, int, int]]:
    """Interpolated Kneser-Ney (absolute discount D) over BPE ids of the
    training stream: raw counts at the order in use, continuation counts
    (distinct left contexts) below it, the uniform distribution at order 0."""
    data = tok.encode("\n".join(train_docs)).ids
    V = tok.get_vocab_size()
    raw = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]  # raw[k][ctx (k-1 ids)][w]
    for i in range(len(data)):
        for k in range(1, min(n, i + 1) + 1):
            raw[k][tuple(data[i - k + 1 : i])][data[i]] += 1
    left = [defaultdict(lambda: defaultdict(set)) for _ in range(n + 1)]
    for k in range(2, n + 1):
        for ctx, ws in raw[k].items():
            for w in ws:
                left[k - 1][ctx[1:]][w].add(ctx[0])
    cont = [{c: {w: len(v) for w, v in ws.items()} for c, ws in left[k].items()} for k in range(n + 1)]

    def prob(ctx: tuple, w: int, k: int, top: bool) -> float:
        if k == 0:
            return 1.0 / V
        c = ctx[len(ctx) - (k - 1) :] if k > 1 else ()
        table = raw[k].get(c) if top else cont[k].get(c)
        if not table:
            return prob(ctx, w, k - 1, False)
        total = sum(table.values())
        lower = D * len(table) / total * prob(ctx, w, k - 1, False)
        return max(table.get(w, 0) - D, 0) / total + lower

    out = []
    for doc in val_docs:
        ids = tok.encode(doc).ids
        bits = sum(-math.log2(prob(tuple(ids[max(0, t - n + 1) : t]), ids[t], min(n, t + 1), True)) for t in range(1, len(ids)))
        out.append((bits, sum(tok_bytes(tok, i) for i in ids[1:]), len(ids) - 1))
    return out


def zoo_row(model: str, arch: str | None, rows, params: int | None) -> dict:
    lo, hi = pooled_ci(rows)
    r = {"model": model, "task": "ts-val", "metric": "bpb", "higher_is_better": False,
         "value": round(pooled(rows), 6), "ci95": [round(lo, 6), round(hi, 6)],
         "n": int(sum(x[2] for x in rows)), "status": "ok"}
    if arch:
        r["tl_arch"] = arch
    if params is not None:
        r["params"] = int(params)
    return r


def main() -> None:
    t_start = time.time()
    bpe, uni = make_tokenizer("bpe"), make_tokenizer("unigram")
    data = stream(bpe)
    base = dict(MAIN, vocab=VOCAB)

    main_model, main_info, main_rows = run_llama(base, bpe, data)
    lo, hi = pooled_ci(main_rows)
    val_loss = sum(r[0] for r in main_rows) * math.log(2) / sum(r[2] for r in main_rows)

    # ablations (a = the main choice)
    _, _, uni_rows = run_llama(base, uni, stream(uni))
    mla = dict(base, attention="mla", rank=112, rope_dim=16, nope_dim=32, v_dim=32)
    assert kv_bytes(mla) == kv_bytes(base)
    _, _, mla_rows = run_llama(mla, bpe, data)
    moe = dict(base, experts=4, top_k=1)
    _, _, moe_rows = run_llama(moe, bpe, data)
    ablations = [
        ablation("tokenizer", "BPE or Unigram at vocab 512, same sample, same model", "bpe",
                 {"tokenizer": "bpe", "vocab_size": VOCAB}, main_rows, "unigram",
                 {"tokenizer": "unigram", "vocab_size": VOCAB}, uni_rows, {}),
        ablation("attention", "GQA or MLA at equal KV bytes per token", "gqa", hf_config(base), main_rows,
                 "mla", hf_config(mla), mla_rows, {"kv_bytes_per_token": {"a": kv_bytes(base), "b": kv_bytes(mla)}}),
        ablation("mlp", "dense SwiGLU or a top-1 MoE of 4 experts at equal active parameters", "dense",
                 hf_config(base), main_rows, "moe", hf_config(moe), moe_rows,
                 {"active_params": {"a": n_params(base)[1], "b": n_params(moe)[1]}}),
    ]

    # scaling: three sizes of the smoke family
    sizes = [dict(d=64, layers=2, heads=4, kv=2, ff=172, vocab=VOCAB), dict(d=96, layers=3, heads=4, kv=2, ff=256, vocab=VOCAB), base]
    points = []
    for c in sizes:
        rows = main_rows if c is base else run_llama(c, bpe, data)[2]
        points.append({"params": n_params(c)[0], "bpb": round(pooled(rows), 6)})
    X = np.c_[np.ones(3), np.log([p["params"] for p in points])]
    coef, *_ = np.linalg.lstsq(X, np.log([p["bpb"] for p in points]), rcond=None)
    scaling = {"law": "bpb = a * params^-alpha", "points": points,
               "fit": {"a": round(float(math.exp(coef[0])), 6), "alpha": round(float(-coef[1]), 6)}}

    # zoo baselines on the same held-out documents
    V = bpe.get_vocab_size()
    zoo = [zoo_row("kn-4", None, kn4_rows(bpe), None)]
    for name, arch, m in (("nplm", "nplm", NPLM(V)), ("lstm", "rnnlm", LSTMLM(V)), ("gpt", "gpt", GPT(V))):
        train(m, data)
        zoo.append(zoo_row(name, arch, doc_bits(m, bpe), sum(p.numel() for p in m.parameters())))
    zoo.append(zoo_row("c1-llama", "llama", main_rows, n_params(base)[0]))
    for model, task, metric, why in (("seq2seq", "dates", "em", "seq2seq EM"), ("classifier", "sst2-2k", "accuracy", "classification"),
                                     ("word2vec", "word-sim", "spearman", "word similarity")):
        zoo.append({"model": model, "task": task, "metric": metric, "value": None, "status": "skipped",
                    "reason": f"smoke tier: {why} rows come from the L6.7 zoo suite on the full capstone run"})

    # 20 seeded samples of the main model
    samples = []
    main_model.eval()
    with torch.no_grad():
        for s in range(20):
            g = torch.Generator().manual_seed(s)
            ids = bpe.encode("Once upon a time").ids
            n0 = len(ids)
            for _ in range(64):
                p = torch.softmax(main_model(torch.tensor([ids[-CTX:]]))[0][0, -1] / 0.8, -1)
                ids.append(int(torch.multinomial(p, 1, generator=g)))
            samples.append({"seed": s, "prompt": "Once upon a time", "temperature": 0.8, "tokens": 64,
                            "text": bpe.decode(ids[n0:])})

    spec = {"name": "c1-smoke", "model": hf_config(base), "tokenizer": {"id": "c1-bpe-512", "path": "tokenizers/c1-bpe-512/tokenizer.json"},
            "data": {"train": ["tokens/c1-bpe-512/stories/_MANIFEST.json"], "val": ["tokens/c1-bpe-512/stories-val.bin"]},
            "optimizer": {"name": "adamw", "lr": LR, "betas": [0.9, 0.95], "weight_decay": 0.1},
            "schedule": {"name": "wsd", "warmup_steps": 20, "decay_frac": 0.2, "min_lr_ratio": 0.1},
            "precision": {"dtype": "fp32"}, "batch": {"micro_batch": BATCH, "accum": 1, "seq_len": CTX},
            "steps": STEPS, "ckpt_every": 100, "seed": SEED}
    report = {
        "format": "tl.capstone-report.v1", "tier": "smoke", "spec": "specs/c1/smoke.json",
        "data": {"corpus": "course synthetic stories (MS-corpus fixture)", "train_docs": len(train_docs), "val_docs": len(val_docs)},
        "tokenizer": {"kind": "bpe", "vocab_size": VOCAB, "sample_bytes": len(train_text.encode())},
        "model": {"config": hf_config(base), "params": n_params(base)[0]},
        "train": {**main_info, "seed": SEED, "precision": "fp32", "micro_batch": BATCH, "accum": 1},
        "eval": {"val_bpb": round(pooled(main_rows), 6), "ci95": [round(lo, 6), round(hi, 6)], "n": len(main_rows),
                 "val_loss": round(val_loss, 6)},
        "ablations": ablations, "scaling": scaling, "long_context": None,
        "zoo": "docs/capstone/c1/zoo.json", "samples": "docs/capstone/c1/samples.jsonl", "release": "specs/c1/release.json",
        "generator": "course/oracle/C1/smoke_capstone.py",
        "versions": {"torch": torch.__version__.split("+")[0], "numpy": np.__version__},
    }
    SPEC_OUT.mkdir(parents=True, exist_ok=True)
    DOC_OUT.mkdir(parents=True, exist_ok=True)
    (SPEC_OUT / "smoke.json").write_text(json.dumps(spec, indent=1) + "\n")
    (DOC_OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (DOC_OUT / "zoo.json").write_text(json.dumps({"format": "tl.eval-results.v1", "suite": "zoo", "seed": SEED,
                                                  "created": "2026-10-09T00:00:00Z", "rows": zoo}, indent=1) + "\n")
    (DOC_OUT / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(json.dumps({"val_bpb": report["eval"], "ablations": [(a["id"], a["mean_diff"], a["ci95"], a["winner"]) for a in ablations],
                      "scaling": scaling["fit"], "zoo": [(r["model"], r.get("value")) for r in zoo],
                      "wall_s": round(time.time() - t_start)}, indent=1))


if __name__ == "__main__":
    main()
