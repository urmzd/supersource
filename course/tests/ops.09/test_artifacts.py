"""Why this exists: ensure the ops.09 learner evidence is present before the check can pass."""
from pathlib import Path

def test_artifact_paths_are_declared():
    # KIND: unit
    # WHY: require the drill to declare the evidence it asks responders to create.
    card = Path("course/modules/ops.09.toml").read_text()
    assert 'artifacts =' in card
