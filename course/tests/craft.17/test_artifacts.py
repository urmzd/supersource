"""Reference evidence for the STRIDE threat-model practice."""

from pathlib import Path


# WHY: the model must cover all STRIDE categories and name controls, residual risk, and owners.
# KIND: unit
def test_artifact_paths_are_declared():
    root = Path(__file__).resolve().parents[3]
    artifact = (root / "course/ref/docs/THREAT_MODEL.md").read_text().lower()
    for threat in (
        "spoofing",
        "tampering",
        "repudiation",
        "information disclosure",
        "denial of service",
        "elevation of privilege",
    ):
        assert threat in artifact
    assert (
        "trust boundaries" in artifact
        and "residual risk" in artifact
        and "owner" in artifact
    )
