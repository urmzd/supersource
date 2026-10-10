<!-- ss:module craft.18 -->
# SBOM and supply chain

## Overview

| | |
|---|---|
| **Module** | `craft.18` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | SBOM per image, allowed-deps.toml |
| **Tests** | `course/tests/craft.18` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `SBOM and supply chain` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ss start craft.18
ss tests craft.18
ss check craft.18
```

---

## 1. Why now

A deployment is built from many direct and transitive packages. An SBOM and dependency policy help responders identify exposure and prove what was shipped.

## 2. Principles

Generate inventories from the exact image digest. Pin base images and dependencies, record licenses and provenance, and define a review path for exceptions. Regenerate after every release.

## 3. Worked example

Build each service image, capture its digest, generate an SPDX or CycloneDX SBOM, and compare every dependency with the allowlist. Investigate unpinned transitive dependencies and known vulnerabilities before signing off.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The artifact names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | `test_artifact_paths_are_declared` |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | `test_artifact_paths_are_declared` |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.17` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
