"""Why this exists: the API migration artifact must support a safe cohort rollout."""

from pathlib import Path


RUNBOOK = Path("docs/runbooks/ApiV2Migration.md")


def test_runbook_has_cohort_gates_and_rollback():
    # KIND: unit
    # WHY: a healthy service can still break old clients, so both versions need
    # explicit evidence: cohort gates on api_version and token, rollback, sunset.
    assert RUNBOOK.is_file(), "create docs/runbooks/ApiV2Migration.md"
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
        "sunset",
        "api_version",
        "token",
    ):
        assert evidence in text, f"runbook is missing migration evidence: {evidence}"
