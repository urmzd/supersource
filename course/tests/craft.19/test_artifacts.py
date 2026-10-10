"""Reference evidence for the secrets and authorization review practice."""

from pathlib import Path


# WHY: authorization must be tested across principal/resource boundaries and secrets need a concrete lifecycle.
# KIND: unit
def test_artifact_paths_are_declared():
    root = Path(__file__).resolve().parents[3]
    artifact = (root / "course/ref/docs/security-review.md").read_text().lower()
    for required in (
        "authorization matrix",
        "constant-time",
        "cache window",
        "cross-tenant",
        "rotation",
        "owner",
    ):
        assert required in artifact
    assert "presented secret" in artifact and "current etag" in artifact
