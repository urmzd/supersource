"""R9 artifact checks. Each test states why it is part of the release gate."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SS_REPO", "."))


def test_agent_mutants_have_separate_ci_coverage():
    """Each semantic agent regression needs a scorer that can observe it."""
    suite = ROOT / "go/tests/agent-evals/suite.json"
    data = json.loads(suite.read_text())
    names = {s["name"] for s in data["scorers"]}
    assert {"citation_resolution", "tool_success", "state_change"} <= names


def test_eval_run_is_reproducible():
    """A fixed suite must include deterministic seeds and a held-out split."""
    config = json.loads((ROOT / "go/tests/agent-evals/suite.json").read_text())
    assert config["seed"] is not None
    assert config["split"]["heldout"]
