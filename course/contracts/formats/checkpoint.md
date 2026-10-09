# Checkpoints and released models

<!-- modules: L0.6 (checkpoint writer and reader), L11.1 (precision state), dur.09 and dur.11 (resume through the activity contract), dur.12 (release), C1
     conformance: formats/ -->

A **model directory** is what the engine serves and what `{tinyllm}` loads: `model.safetensors` ([safetensors.md](safetensors.md)), [`config.json`](config.schema.json), and, when `config.json` says `"tl_tokenizer": "file"`, `tokenizer.json` ([tokenizer.md](tokenizer.md)). A checkpoint and a released model are model directories with more files. All paths are relative to `/artifacts`.

## Checkpoint step directory

`runs/<run_id>/ckpt/step-<nnnnnn>/` (six digits, the optimizer step), holding:

| File | Content |
|---|---|
| `model.safetensors` | the parameters, F32 (the master weights, also under bf16 training) |
| `optimizer.safetensors` | per parameter `<name>`: AdamW `<name>.exp_avg` and `<name>.exp_avg_sq`; SGD with momentum and Muon `<name>.momentum_buffer`; all F32, metadata `{"format": "tinyllm", "optimizer": "<name>", "step": "<step>"}` |
| `config.json` | the model config, verbatim from the train spec |
| `tokenizer.json` | when the model uses one |
| `generation_config.json` | optional ([generation-config.schema.json](generation-config.schema.json)) |
| `trainer_state.json` | [trainer-state.schema.json](trainer-state.schema.json): step, tokens seen, the data cursor, the generator state, the learning rate, the spec's sha256, the git sha |
| `MANIFEST.json` | [manifest.schema.json](manifest.schema.json), `kind: checkpoint`: every other file with its sha256 and size |

`runs/<run_id>/ckpt/LATEST` is a text file holding the name of the newest complete step directory and a newline, for example `step-000500\n`.

### Writing atomically

A crash at any instant leaves either the previous checkpoint or the new one, never a mix:

1. Write every file into `step-<nnnnnn>.tmp/`, `fsync` each file.
2. Write `MANIFEST.json` last, `fsync` it and the directory.
3. `rename` `step-<nnnnnn>.tmp` to `step-<nnnnnn>`.
4. Write `LATEST.tmp`, `fsync`, `rename` it to `LATEST`, `fsync` the `ckpt/` directory.
5. Emit `{"kind": "ckpt", ...}` on the progress file (spec/subprocess-activity.md) only now.
6. Delete step directories beyond the spec's `keep_ckpts` newest, and any stale `*.tmp`.

### Resuming

`--resume <step dir>` (or, without a path, the directory `LATEST` names) loads the parameters, optimizer state, and `trainer_state.json`, after checking every file against `MANIFEST.json`. A directory whose manifest is missing or does not verify is skipped for the newest older one that verifies. Resuming continues **bitwise** as the uninterrupted run would have: the data cursor names the next window, the `shuffle` generator state continues its stream (spec/pcg32.md), and the step count drives the schedule. dur.11's kill test compares the final loss of a resumed run with an uninterrupted one on the smoke config.

## Released model

`models/<model_id>/<version>/`, written by `ModelRelease` (dur.12) from one checkpoint:

| File | Content |
|---|---|
| `model.safetensors` | weights, F32 or BF16, or int4 (`<name>.qweight` and `<name>.scales`, safetensors.md) |
| `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` | as in the checkpoint; `tokenizer_config.json` when a chat template is needed |
| `MODEL_CARD.md` | from [`templates/MODEL_CARD.md`](../templates/MODEL_CARD.md); the release gate fails without it |
| `ledger.json` | a JSON array of the [ledger](ledger.schema.json) rows of every source the training data came from; the gate fails when any lacks `train` in `allowed_uses` or is `revoked` |
| `evals/summary.json` | the EvalSuite summary ([eval-result.schema.json](eval-result.schema.json) `$defs/summary`) the gate read |
| `heads/<name>.json` | optional linear heads over this model's embeddings ([linear-head.schema.json](linear-head.schema.json)) |
| `MANIFEST.json` | `kind: release`, with `model_id` and `version` |

A released directory is immutable: a new version is a new directory.
