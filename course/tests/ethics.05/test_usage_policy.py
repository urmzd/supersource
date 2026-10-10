"""Artifact checks for ethics.05. Each test says why the check exists."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("SS_REPO", "."))
COURSE = Path(os.environ.get("SS_COURSE_TREE", Path(__file__).resolve().parents[2]))


def test_policy_schema_and_order():
    """The ordered config must conform to policy.v1 before deployment."""
    policy = yaml.safe_load((ROOT / "deploy/policy.v1.yaml").read_text())
    schema = json.loads((COURSE / "contracts/formats/policy.v1.schema.json").read_text())
    assert policy["version"] == 1 and policy["rules"]
    assert all(rule["id"] and rule["reason"] for rule in policy["rules"])
    assert [rule["id"] for rule in policy["rules"]] == ["restricted-model", "free-token-cap", "unsafe-prompt"]
    assert schema["properties"]["rules"]["type"] == "array"


def test_red_team_prompts_are_denied():
    """Known disallowed use cases must be explicit classifier-backed rules."""
    policy = yaml.safe_load((ROOT / "deploy/policy.v1.yaml").read_text())
    rule = next(r for r in policy["rules"] if r["id"] == "unsafe-prompt")
    assert rule["action"] == "deny" and rule["classifier"]["class"] == "unsafe"
    assert 0 < rule["classifier"]["threshold"] <= 1


def test_no_pii_in_audit_examples():
    """Policy documentation must specify privacy-safe audit fields."""
    doc = (ROOT / "docs/USAGE_POLICY.md").read_text().lower()
    assert "must not place prompt text" in doc
    assert "request id" in doc and "policy revision" in doc
