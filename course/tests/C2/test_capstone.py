"""Optional post-training capstone checks: provenance and supported claims."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SS_REPO", "."))


def test_run_manifest_and_model_card():
    """A post-trained artifact must name its base, data, objective, and seed."""
    run = json.loads((ROOT / "artifacts/c2/run.json").read_text())
    card = (ROOT / "docs/capstone/c2/MODEL_CARD.md").read_text().lower()
    assert run["base_checkpoint"] and run["dataset_revision"] and run["objective"] and run["seed"] is not None
    assert "limitations" in card and "evaluation" in card and "base checkpoint" in card
