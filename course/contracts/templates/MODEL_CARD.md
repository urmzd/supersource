# Model card: <model_id> <version>

## Model details

- **Developer:** <you>, as part of <system>.
- **Architecture:** <tl_arch and size: layers, hidden size, heads, KV heads,
  vocabulary, context length, parameters>.
- **Training:** <data (link the DATASHEET), tokens, steps, precision, seed,
  the train spec path and its sha256>.
- **Release:** <date>, `models/<model_id>/<version>/`, MANIFEST sha256
  <hex>.
- **Third-party components:** <weights or tokenizers you did not train, with
  their licenses (for example SmolLM2-135M-Instruct, Apache-2.0)>.
- **License:** <SPDX id>.

## Intended use

- **Primary uses:** <what the model is for in this system>.
- **Out of scope:** <uses the model must not serve; the usage policy rules
  that enforce them (gateway/policy.v1.yaml rule ids)>.

## Evaluation

| Suite | Metric | Value (95% CI) | Release threshold |
|---|---|---|---|
| <suite> | <metric> | <mean (low, high)> | <threshold> |

<Copy the numbers from evals/summary.json of this release; say which
subjects and seeds produced them.>

## Bias, risks, and limitations

- <Measured bias results (ethics.04) and what they mean.>
- <Safety evaluation results and known failure modes.>
- <Limitations of size, data, and language coverage.>

## Data

<Sources and licenses, from ledger.json of this release: every source must
allow `train`. PII handling (scrubbed counts from the corpus manifest).
Decontamination against the eval sets.>

<!-- contracts/templates/MODEL_CARD.md (ethics.03, dur.12): copy to
     MODEL_CARD.md; the ModelRelease gate refuses a release without it
     (formats/checkpoint.md). -->
