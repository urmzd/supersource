"""Reference evidence for the per-image SPDX SBOM practice."""

import json
from pathlib import Path


# WHY: each image inventory must be parseable SPDX and preserve package and dependency relationships.
# KIND: unit
def test_artifact_paths_are_declared():
    root = Path(__file__).resolve().parents[3]
    for name in ("engine", "gateway"):
        doc = json.loads((root / f"course/ref/docs/sbom/{name}.spdx.json").read_text())
        assert doc["spdxVersion"] == "SPDX-2.3"
        assert doc["SPDXID"] == "SPDXRef-DOCUMENT"
        assert doc["packages"] and doc["relationships"]
        ids = {package["SPDXID"] for package in doc["packages"]}
        assert len(ids) == len(doc["packages"])
        assert all(
            row["relatedSpdxElement"] in ids
            for row in doc["relationships"]
            if row["relationshipType"] != "DESCRIBES"
        )
    contract = (root / "course/contracts/allowed-deps.toml").read_text()
    assert "[rust.tl-engine]" in contract and "[go.tinyllm]" in contract
