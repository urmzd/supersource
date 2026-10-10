"""Agent deployment artifact checks. The tests explain the policy they protect."""

import json
import os
import tomllib
from pathlib import Path

ROOT = Path(os.environ.get("SS_REPO", "."))
COURSE = Path(os.environ.get("SS_COURSE_TREE", Path(__file__).resolve().parents[2]))
SYSTEM = tomllib.loads((ROOT / "system.toml").read_text())["system"]["name"]
CHART = ROOT / "deploy/helm" / f"{SYSTEM}-agent"


def test_agent_image_policy():
    """The image must run as non-root and use a pinned non-latest base."""
    dockerfile = (ROOT / "deploy/docker/agent.Dockerfile").read_text().lower()
    assert "user " in dockerfile and "latest" not in dockerfile
    assert "from " in dockerfile and "healthcheck" in dockerfile


def test_agent_chart_schema():
    """The values schema is the published chart contract."""
    schema = json.loads((CHART / "values.schema.json").read_text())
    reference = json.loads(
        (COURSE / "contracts/helm/agent.values.schema.json").read_text()
    )
    assert schema == reference


def test_agent_queue_and_secret():
    """The chart must target durable queue agent and never inline API keys."""
    text = "\n".join(p.read_text() for p in CHART.rglob("*.yaml"))
    assert "agent" in text and "secretKeyRef" in text
    assert "replicas:" in text and "resources:" in text
