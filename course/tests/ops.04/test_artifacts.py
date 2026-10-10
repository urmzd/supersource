"""Why this exists: a migration runbook must provide executable safety gates."""

from pathlib import Path


RUNBOOK = Path("docs/runbooks/KvV2Migration.md")


def test_runbook_has_operational_gates_and_recovery():
    # KIND: unit
    # WHY: a narrative without a canary gate, rollback, and validation is unsafe to
    # page from; the runbook must also carry the v1/v2 hash and scale evidence.
    assert RUNBOOK.is_file(), "create docs/runbooks/KvV2Migration.md"
    text = RUNBOOK.read_text(encoding="utf-8").lower()
    assert len(text.split()) >= 180, "record enough detail for an on-call handoff"
    for evidence in (
        "owner",
        "precondition",
        "canary",
        "rollback",
        "verification",
        "v1",
        "v2",
        "scale",
        "hash",
    ):
        assert evidence in text, f"runbook is missing migration evidence: {evidence}"
