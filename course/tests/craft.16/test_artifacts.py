"""Reference evidence for the performance-regression bisect practice."""

from pathlib import Path


# WHY: a bisect without fixed workload, thresholds, and raw endpoint evidence can blame noise.
# KIND: unit
def test_artifact_paths_are_declared():
    root = Path(__file__).resolve().parents[3]
    artifact = (root / "course/ref/docs/maintenance/perf-bisect.md").read_text()
    for required in (
        "Workload",
        "warm-up",
        "median",
        "git bisect run",
        "Rollback",
        "Owner",
    ):
        assert required.lower() in artifact.lower()
    assert "15%" in artifact and "3 ms" in artifact
