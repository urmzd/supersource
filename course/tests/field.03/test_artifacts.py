"""Why this exists: ensure the field.03 learner evidence is present before the check can pass."""

from pathlib import Path


def test_artifact_paths_are_declared():
    # KIND: unit
    # WHY: course verification needs annotated tests for the declared engagement evidence.
    card = Path("course/modules/field.03.toml").read_text()
    assert "artifacts =" in card
