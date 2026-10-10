"""The near-tie rule of tokens-equal (DESIGN 5.7): an oracle file may carry
its own top-2 margin per step, so a divergence at an oracle near tie is
reported `near-tie` without rerunning the learner's logits; verify rejects an
oracle file with a near tie in a step that has no near_tie rule."""

import json


def _setup(ss, margins):
    for name in ("greedy_ab.json", "greedy_hello.json"):
        p = ss.course / "fixtures/MS-M90" / name
        doc = json.loads(p.read_text())
        doc["margins"] = margins
        p.write_text(json.dumps(doc))
    man = ss.course / "fixtures/MANIFEST.tsv"
    import hashlib

    lines = []
    for line in man.read_text().splitlines():
        c = line.split("\t")
        if c[0].startswith("course/fixtures/MS-M90/"):
            data = (ss.site / c[0]).read_bytes()
            c[1], c[2] = hashlib.sha256(data).hexdigest(), str(len(data))
        lines.append("\t".join(c))
    man.write_text("\n".join(lines) + "\n")
    ss.commit_site("oracle margins")
    ss.init()
    ss.install_system()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss("check", "M90.1", rc=0)
    cli = ss.learner / "python/tinyllm/__main__.py"
    cli.write_text(
        cli.read_text()
        .replace(
            "ids = bigram.generate(a.prompt, a.max_tokens",
            "ids = bigram.generate(a.prompt + 'x', a.max_tokens",
        )
        .replace(
            'print(json.dumps({"logits"',
            'raise SystemExit("logits must not run"); print(json.dumps({"logits"',
        )
    )


def test_oracle_margin_below_the_limit_is_a_near_tie(ss):
    _setup(ss, [5e-4] + [0.5] * 7)
    out = ss("milestone", "MS-M90", "--step", "greedy", rc=0).out
    assert (
        "near-tie" in out and "the oracle's top-2 margin there is 0.0005 < 0.001" in out
    )


def test_oracle_margin_above_the_limit_is_a_real_divergence(ss):
    _setup(ss, [0.5] * 8)
    out = ss("milestone", "MS-M90", "--step", "greedy", rc=1).out
    assert (
        "the oracle's top-2 margin there is 0.5 >= 0.001, so this is a real divergence"
        in out
    )


def test_verify_rejects_unfiltered_oracle_without_near_tie(ss):
    ms = ss.course / "milestones/MS-M90.toml"
    ms.write_text(ms.read_text().replace(", near_tie = { margin = 1e-3 }", ""))
    p = ss.course / "fixtures/MS-M90/greedy_ab.json"
    doc = json.loads(p.read_text())
    doc["margins"] = [2e-4] * 8
    p.write_text(json.dumps(doc))
    out = ss("verify", "course", "--global").out
    assert "holds a near tie (top-2 margin 0.0002 < 1e-3)" in out
