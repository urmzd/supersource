"""Why this exists: ensure the craft.11 learner evidence is present before the check can pass."""

from pathlib import Path


def test_artifact_paths_are_declared():
    # KIND: unit
    # WHY: course verification needs annotated tests for the declared learner evidence.
    card = Path("course/modules/craft.11.toml").read_text()
    for artifact in (
        '"CHANGELOG.md"',
        '".github/workflows/release.yml"',
        '"docs/releases/v1.0.0.md"',
    ):
        assert artifact in card
