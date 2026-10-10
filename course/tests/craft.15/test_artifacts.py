"""Reference evidence for the dependency-upgrade review practice."""

from pathlib import Path


# WHY: a usable upgrade review needs reproducible graph, security, performance, and rollback evidence.
# KIND: unit
def test_artifact_paths_are_declared():
    root = Path(__file__).resolve().parents[3]
    artifact = (root / "course/ref/docs/maintenance/dependency-upgrades.md").read_text()
    for required in ("Resolved graph", "govulncheck", "p95", "Rollback", "Owner"):
        assert required.lower() in artifact.lower()
    assert "semver range as proof" in artifact
