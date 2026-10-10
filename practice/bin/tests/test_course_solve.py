"""The solve kind end to end through ss (DESIGN 5.5): start writes the answer
template, check grades with SymPy in a subprocess, proofs are self-graded
from y/n answers (piped here), --ci skips the rubric, verify check 7 proves
the key."""


def run_ss(ss, *args, stdin: str = "", rc=None) -> str:
    return ss(*args, input=stdin, rc=rc).out


ANSWERS = """
[q1]
answer = "2x cos(x^2)"
[q2]
answer = "(2, oo) U (-oo, -2)"
[q3a]
answer = "2/4"
[q3.b]
answer = "yes"
[q4]
answer = "[-5, 0]"
[q5]
proof = "S-M90a/q5.md"
"""

PROOF = """# S-M90a q5

Let a = 2j and b = 2k for integers j and k. Then a + b = 2j + 2k = 2(j + k),
and j + k is an integer, so a + b is even.
"""


def test_solve_start_check_and_self_graded_proof(ss):
    ss.add_extras("solve")
    ss.init()
    out = ss("start", "S-M90a", rc=0).out
    assert "wrote     solve/S-M90a.toml" in out and "solve/S-M90a/q5.md" in out
    tmpl = (ss.learner / "solve" / "S-M90a.toml").read_text()
    assert "[q3.a]" in tmpl and 'proof  = "S-M90a/q5.md"' in tmpl
    # Nothing answered: every question is missing; exit 1.
    out = run_ss(ss, "check", "S-M90a", rc=1)
    assert "missing" in out and "FAIL S-M90a" in out

    (ss.learner / "solve" / "S-M90a.toml").write_text(
        ANSWERS.replace('"yes"', '"no"').replace("2x cos(x^2)", "2*x*cos(x)")
    )
    (ss.learner / "solve" / "S-M90a" / "q5.md").write_text(PROOF)
    out = run_ss(ss, "check", "S-M90a", stdin="y\nn\n", rc=1)
    assert "FAIL" in out and "differs at x=" in out
    assert "cos(x**2)" not in out and "x^2" not in out  # feedback never shows the key
    assert "q3.b" in out and "rubric: no to Writes the sum" in out

    (ss.learner / "solve" / "S-M90a.toml").write_text(ANSWERS)
    out = run_ss(ss, "check", "S-M90a", rc=1)
    assert "your rubric said no" in out  # the recorded self-grade of this proof
    out = run_ss(ss, "check", "S-M90a", "--regrade", stdin="y\ny\n", rc=0)
    assert "PASS S-M90a (self-graded proofs)" in out
    v = ss.verdicts("S-M90a")[-1]
    assert v["result"] == "pass" and v["self"] is True
    assert v["questions"] == {
        "q1": "pass",
        "q2": "pass",
        "q3.a": "pass",
        "q3.b": "pass",
        "q4": "pass",
        "q5": "self",
    }
    # The proof is unchanged: its self-grade is reused without asking.
    out = run_ss(ss, "check", "S-M90a", rc=0)
    assert "self-graded earlier" in out
    assert "self      S-M90a" in ss("status", rc=0).out

    # An edited proof needs a new self-grade; --ci never asks.
    (ss.learner / "solve" / "S-M90a" / "q5.md").write_text(PROOF + "\nQED.\n")
    out = run_ss(ss, "check", "S-M90a", rc=1)
    assert "self (skipped)" in out
    out = run_ss(ss, "check", "--all", "--ci", rc=0)
    assert "S-M90a" in out and "self" in out and "skipped in --ci" in out


def test_verify_check_7_proves_the_key(ss):
    ss.add_extras("solve")
    out = ss("verify", "course", "S-M90a").out
    assert (
        "ok     7 solve key parses; every expect passes and every reject canary fails"
        in out
    )
    key = ss.course / "solve" / "S-M90a" / "key.toml"
    key.write_text(key.read_text().replace('answer  = "0.5"', 'answer  = "1/2"'))
    out = ss("verify", "course", "S-M90a").out
    assert "FAIL   7 solve key" in out and "reject canary '1/2' passes" in out
