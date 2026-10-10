"""Why this exists: ensure the craft.10 learner evidence is present before the check can pass."""

from pathlib import Path


def test_artifact_paths_are_declared():
    # KIND: unit
    # WHY: course verification needs annotated tests for the declared learner evidence.
    card = Path("course/modules/craft.10.toml").read_text()
    assert "artifacts =" in card
