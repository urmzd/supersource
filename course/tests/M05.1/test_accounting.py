"""Course tests for M05.1: counting parameters, FLOPs, KV bytes, and memory
(tinyllm/accounting.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/M05.1), and the chapter section it comes
from.

The worked example of the chapter (section 3): V = 10, d = 4, 2 layers,
2 query heads over 1 kv head of width 2, d_ff = 6, untied. Parameters
embed 40, attn 96, mlp 144, norm 20, lm_head 40, total 340. Forward FLOPs
per token at seq_len 3: 2 * 280 + 96 = 656; training 1968. KV cache 16
bytes per token in fp16.

The golden fixture (course/fixtures/M05.1/hf_param_counts.json) holds
Hugging Face transformers 5.19.0 `num_parameters()` for SmolLM2-135M, Llama 2
7B, Mixtral 8x7B, DeepSeek-V3, and five tiny configs, built on the meta
device, from course/oracle/M05.1/hf_param_counts.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest
from tinyllm.accounting import (
    ModelConfig,
    flops_per_token,
    kv_bytes_per_token,
    memory_plan,
    param_count,
)

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "M05.1", "hf_param_counts.json")
KEYS = ("embed", "attn", "mlp", "norm", "lm_head", "total")


def hand() -> ModelConfig:
    return ModelConfig(vocab=10, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_head=2, d_ff=6, tie_embeddings=False)


def smollm2() -> ModelConfig:
    return ModelConfig(vocab=49152, d_model=576, n_layers=30, n_heads=9, n_kv_heads=3, d_head=64, d_ff=1536,
                       tie_embeddings=True)


def from_hf(case: dict) -> ModelConfig:
    """The fixture's HF config fields mapped onto ModelConfig."""
    t, h = case["model_type"], case["config"]
    H = h["num_attention_heads"]
    if t == "deepseek_v3":
        return ModelConfig(
            vocab=h["vocab_size"], d_model=h["hidden_size"], n_layers=h["num_hidden_layers"], n_heads=H,
            n_kv_heads=h["num_key_value_heads"], d_head=h["qk_nope_head_dim"], d_ff=h["intermediate_size"],
            tie_embeddings=h["tie_word_embeddings"], attn="mla", kv_lora_rank=h["kv_lora_rank"],
            qk_rope_dim=h["qk_rope_head_dim"], n_experts=h["n_routed_experts"], top_k=h["num_experts_per_tok"],
            n_shared=h["n_shared_experts"], q_lora_rank=h["q_lora_rank"] or 0, v_head_dim=h["v_head_dim"],
            d_ff_expert=h["moe_intermediate_size"], n_dense_layers=h["first_k_dense_replace"])
    kv = h["num_key_value_heads"]
    return ModelConfig(
        vocab=h["vocab_size"], d_model=h["hidden_size"], n_layers=h["num_hidden_layers"], n_heads=H, n_kv_heads=kv,
        d_head=h.get("head_dim") or h["hidden_size"] // H, d_ff=h["intermediate_size"],
        tie_embeddings=h["tie_word_embeddings"], attn="mha" if kv == H else "gqa",
        n_experts=h.get("num_local_experts") or 0, top_k=h.get("num_experts_per_tok") or 0,
        qkv_bias=(t == "qwen2"))


def cases() -> dict:
    with open(FIX) as f:
        return {c["name"]: c for c in json.load(f)["cases"]}


# --- the worked example -------------------------------------------------------------


def test_hand_example_params():
    # WHY: the chapter's worked example, component by component: embedding
    #      10 x 4 = 40; attention per layer q 4x4 + k 2x4 + v 2x4 + o 4x4 =
    #      48, two layers 96; MLP 3 x 4 x 6 = 72 per layer, 144; norms
    #      2 x 4 per layer plus the final 4 = 20; lm_head 40; total 340.
    # KIND: unit
    # CATCHES: s02, s10, m01
    # CHAPTER: M05.1 section 3, Worked example by hand
    p = param_count(hand())
    assert {k: p[k] for k in KEYS} == {"embed": 40, "attn": 96, "mlp": 144, "norm": 20, "lm_head": 40, "total": 340}
    assert p["active"] == 340


def test_hand_example_flops_and_kv_bytes():
    # WHY: N = attn + mlp + lm_head = 96 + 144 + 40 = 280 matmul parameters,
    #      2 FLOPs each; the attention term is 2 layers x 2 heads x 3
    #      positions x (2 + 2) x 2 = 96; forward 656 and a training step three
    #      times that. One token adds a key and a value of width 2 for its one
    #      kv head in both layers: 2 x 2 x 1 x 2 x 2 bytes = 16 in fp16.
    # KIND: unit
    # CATCHES: s03, s04, s07
    # CHAPTER: M05.1 section 3, Worked example by hand
    c = hand()
    assert flops_per_token(c, 3, training=False) == 656
    assert flops_per_token(c, 3, training=True) == 1968
    assert kv_bytes_per_token(c, 2) == 16


def test_hand_example_memory_plan():
    # WHY: P = 340: fp32 AdamW keeps weights 1360, grads 1360, no master
    #      copy, two fp32 moments 2720; activations for batch 1, seq 3 are
    #      3 x (2 x A x 4 + 4 x 10) with A = 8 (norm in and out) + 4 + 2 + 2
    #      (q, k, v) + 2 x 3 (softmax rows) + 4 (attention output) + 8 (norm
    #      in and out) + 4 x 6 (gate, up, activation, product) = 58, so
    #      3 x 504 = 1512.
    # KIND: unit
    # CATCHES: s08, s12
    # CHAPTER: M05.1 section 3, Worked example by hand
    m = memory_plan(hand(), 1, 3, 4, "adamw")
    assert m == {"weights": 1360, "grads": 1360, "master": 0, "optimizer": 2720, "activations": 1512, "total": 6952}


# --- against Hugging Face ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["smollm2-135m", "llama-2-7b", "mixtral-8x7b", "deepseek-v3", "tiny-qwen2-bias", "tiny-llama-mha-headdim",
     "tiny-mixtral-mqa-tied", "tiny-deepseek-mla-moe", "tiny-deepseek-qlora-tied"],
)
def test_golden_hf_param_counts(name):
    # WHY: `{tinyllm} info` must report the parameter count Hugging Face
    #      reports (MS-L7 step 4), component by component, for dense GQA,
    #      MHA with a head_dim that is not d / H, qkv biases, tied and untied
    #      heads, mixture of experts with shared experts and dense first
    #      layers, and MLA with and without query compression.
    # KIND: golden
    # CATCHES: s01, s02, s10, s13, m02, m03
    # CHAPTER: M05.1 section 4, The interface
    c = cases()[name]
    p = param_count(from_hf(c))
    assert {k: p[k] for k in KEYS} == c["hf"]
    if "safetensors_total" in c:
        assert p["total"] == c["safetensors_total"]


def test_counts_are_python_ints():
    # WHY: 671,026,404,352 must come back as that integer: a float rounds the
    #      last digits away (float32 keeps 7), and a numpy int32 overflows.
    # KIND: boundary
    # CATCHES: s14
    # CHAPTER: M05.1 section 5, Pitfalls
    c = from_hf(cases()["deepseek-v3"])
    p = param_count(c)
    assert all(type(v) is int for v in p.values())
    assert p["total"] == 671_026_404_352
    assert type(flops_per_token(c, 4096, True)) is int
    assert type(kv_bytes_per_token(c, 2)) is int
    assert all(type(v) is int for v in memory_plan(c, 1, 16, 2, "adamw").values())


# --- properties ---------------------------------------------------------------------------


def test_tying_saves_exactly_one_matrix_of_params_but_no_flops():
    # WHY: a tied model stores the V x d matrix once (lm_head 0), so the total
    #      drops by exactly V d; but the lm_head is still a matrix product per
    #      token, so the FLOPs do not change.
    # KIND: property
    # CATCHES: s01, s06
    # CHAPTER: M05.1 section 2.2, Embeddings and the head
    untied = replace(smollm2(), tie_embeddings=False)
    a, b = param_count(smollm2()), param_count(untied)
    assert b["total"] - a["total"] == 49152 * 576
    assert a["lm_head"] == 0 and b["lm_head"] == 49152 * 576
    assert flops_per_token(smollm2(), 512, False) == flops_per_token(untied, 512, False)


def test_kv_bytes_scale_with_kv_heads_not_query_heads():
    # WHY: the cache stores keys and values per KV head: MQA (1 kv head)
    #      caches 1/H of MHA, GQA Hkv/H, and the number of query heads does
    #      not enter. SmolLM2-135M in bf16: 30 x 2 x 3 x 64 x 2 = 23,040
    #      bytes per token.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: M05.1 section 2.4, KV bytes
    s = smollm2()
    assert kv_bytes_per_token(s, 2) == 23040
    mha = replace(s, n_kv_heads=9, attn="mha")
    mqa = replace(s, n_kv_heads=1)
    assert kv_bytes_per_token(mha, 2) == 3 * kv_bytes_per_token(s, 2) == 9 * kv_bytes_per_token(mqa, 2)
    assert kv_bytes_per_token(replace(s, n_heads=27, d_ff=1536), 2) == 23040
    assert kv_bytes_per_token(s, 4) == 2 * kv_bytes_per_token(s, 2)


def test_mla_caches_the_latent():
    # WHY: MLA caches one latent of width kv_lora_rank plus one shared rope
    #      key per layer, whatever the head count: DeepSeek-V3 in bf16 is
    #      61 x (512 + 64) x 2 = 70,272 bytes per token, where an MHA cache
    #      of its 128 heads of width 128 + 64 would be 61 x 2 x 128 x 192 x 2.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: M05.1 section 2.4, KV bytes
    ds = from_hf(cases()["deepseek-v3"])
    assert kv_bytes_per_token(ds, 2) == 70272
    assert kv_bytes_per_token(replace(ds, n_heads=1, n_kv_heads=1), 2) == 70272


def test_moe_active_parameters():
    # WHY: a token visits top_k of E experts, so the parameters it touches
    #      (and its FLOPs) are far fewer than the total: Mixtral 8x7B touches
    #      about 12.9B of 46.7B, DeepSeek-V3 about 37.6B of 671B. Dense models
    #      touch everything.
    # KIND: property
    # CATCHES: s05
    # CHAPTER: M05.1 section 2.3, Mixture of experts
    mx = param_count(from_hf(cases()["mixtral-8x7b"]))
    ds = param_count(from_hf(cases()["deepseek-v3"]))
    assert mx["active"] == mx["total"] - 32 * 6 * 3 * 4096 * 14336 == 12_879_925_248
    assert ds["active"] == ds["total"] - 58 * 248 * 3 * 7168 * 2048 == 37_552_282_624
    assert param_count(smollm2())["active"] == param_count(smollm2())["total"]


def test_smollm2_flops():
    # WHY: N = 134,479,872 matmul parameters (the tied head counts as a
    #      product) and an attention term of 2 x 30 x 9 x 2048 x 128 at a
    #      2048-token context: 410,517,504 FLOPs per token forward. At this
    #      size the attention term is half of 2N: "6N" alone undercounts.
    # KIND: unit
    # CATCHES: s04, s06, s07
    # CHAPTER: M05.1 section 2.5, FLOPs per token
    assert flops_per_token(smollm2(), 2048, False) == 410_517_504
    assert flops_per_token(smollm2(), 2048, True) == 3 * 410_517_504


def test_flops_attention_term_is_linear_in_context():
    # WHY: one more context position costs 2 L H (d_qk + d_v) FLOPs per token
    #      in the forward (QK^T and AV each read it once), for MHA, GQA, and
    #      MLA (d_qk = dh + dr, d_v = v_head_dim).
    # KIND: property
    # CATCHES: s07
    # CHAPTER: M05.1 section 2.5, FLOPs per token
    for c, step in ((smollm2(), 2 * 30 * 9 * 128), (from_hf(cases()["tiny-deepseek-mla-moe"]), 2 * 3 * 4 * (12 + 6))):
        f = [flops_per_token(c, t, False) for t in (1, 2, 3, 100)]
        assert f[1] - f[0] == f[2] - f[1] == step
        assert f[3] - f[0] == 99 * step


def test_flops_count_only_routed_experts():
    # WHY: a MoE layer's forward costs the router plus top_k experts plus the
    #      shared ones, not all E experts: adding experts at fixed top_k adds
    #      parameters but no FLOPs.
    # KIND: property
    # CATCHES: s15
    # CHAPTER: M05.1 section 2.3, Mixture of experts
    c = from_hf(cases()["tiny-mixtral-mqa-tied"])
    more = replace(c, n_experts=8)
    assert param_count(more)["total"] > param_count(c)["total"]
    router = 2 * (8 - 4) * c.d_model * c.n_layers
    assert flops_per_token(more, 64, False) - flops_per_token(c, 64, False) == router


def test_memory_plan_16_bytes_per_param():
    # WHY: AdamW costs 16 bytes per parameter before activations whether you
    #      train in fp32 (4 + 4 + 8) or bf16 (2 + 2 + a 4-byte master + 8),
    #      the ZeRO paper's 16 Psi; SGD with momentum costs 12 in fp32.
    # KIND: property
    # CATCHES: s08, s11
    # CHAPTER: M05.1 section 2.6, Training memory
    P = param_count(smollm2())["total"]
    for db in (2, 4):
        m = memory_plan(smollm2(), 1, 8, db, "adamw")
        assert m["weights"] + m["grads"] + m["master"] + m["optimizer"] == 16 * P
    s = memory_plan(smollm2(), 1, 8, 4, "sgd")
    assert s["weights"] + s["grads"] + s["master"] + s["optimizer"] == 12 * P
    assert s["total"] == sum(v for k, v in s.items() if k != "total")


def test_activations_linear_in_batch_quadratic_in_context():
    # WHY: activations grow with batch x seq, and the stored softmax rows
    #      add H x seq values per token per layer: doubling the context more
    #      than doubles them, by exactly the extra probability rows.
    # KIND: property
    # CATCHES: s12
    # CHAPTER: M05.1 section 2.6, Training memory
    s = smollm2()
    a = lambda b, t: memory_plan(s, b, t, 2, "adamw")["activations"]  # noqa: E731
    assert a(4, 256) == 4 * a(1, 256)
    extra = a(1, 512) - 2 * a(1, 256)
    assert extra == 512 * 30 * 9 * 256 * 2


def test_validation():
    # WHY: an inconsistent config gives silently wrong counts: heads that do
    #      not divide into groups, MHA with fewer kv heads, MLA without a
    #      latent, a top_k larger than the experts. Fail loudly instead.
    # KIND: boundary
    # CATCHES: m04, m05
    # CHAPTER: M05.1 section 4, The interface
    s = smollm2()
    bad = [replace(s, n_kv_heads=2), replace(s, attn="mha"), replace(s, attn="mla"), replace(s, d_model=0),
           replace(s, n_experts=4, top_k=5), replace(s, n_experts=4, top_k=0), replace(s, top_k=2),
           replace(s, n_dense_layers=31), replace(s, attn="sparse")]
    for c in bad:
        with pytest.raises(ValueError):
            param_count(c)
    with pytest.raises(ValueError):
        flops_per_token(s, 0, False)
    with pytest.raises(ValueError):
        kv_bytes_per_token(s, 0)
    with pytest.raises(ValueError):
        memory_plan(s, 1, 8, 3, "adamw")
    with pytest.raises(ValueError):
        memory_plan(s, 1, 8, 2, "lion")
    with pytest.raises(ValueError):
        memory_plan(s, 0, 8, 2, "sgd")
