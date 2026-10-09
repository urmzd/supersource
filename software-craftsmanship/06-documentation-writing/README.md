# Documentation Writing

## Overview

- **Primary reference**: *Software Engineering at Google*, ch. 10 [Documentation](https://abseil.io/resources/swe-book/html/ch10.html) (free)
- **Supplementary**: [Architecture Decision Records](https://adr.github.io/) (free); [Diataxis](https://diataxis.fr/) (free); [Write the Docs](https://www.writethedocs.org/guide/) (free); *Docs for Developers* (Apress, recommended); the survey in [diagramming-and-documentation/02-documentation-writing](../../diagramming-and-documentation/02-documentation-writing/)
- **Prerequisites**: none beyond a repository you work in
- **Estimated time**: 1 to 2 h for the first record; the habit lasts the whole course

## Key Takeaways

- Name the document type before you write it: a tutorial, a how-to guide, a reference, or an explanation. Mixing two makes both worse.
- Decisions go in decision records: one decision per file, append-only, with the forces, the costs, and the rejected options.
- Docs live in the repository, change in the same pull request as the code, and are checked by CI where a check is possible.

## How to Study

- Write ADR-0001 for your own system (the first chapter), then read the records of a project you use (Rust RFCs, Kubernetes KEPs) and compare.
- For every document you write later in the course (runbooks, postmortems, the model card), name its type first.

---

# Concepts & Techniques

## Core Insight

The most expensive knowledge to lose is why the system is the way it is. Code shows what; tests show what must stay true; only writing preserves why. Small, dated, append-only records are cheap enough to write while the reasons are fresh.

## 1. Document types (Diataxis)

**Key ideas**:
- **Tutorial**: learning by doing; **how-to guide**: a task for someone who knows the basics (runbooks); **reference**: exact facts (contracts); **explanation**: why (decision records).

## 2. Architecture decision records

**Key ideas**:
- **Five sections**: Status, Context, Decision, Consequences, Alternatives considered.
- **Supersede, never edit**: a changed mind is a new record that points back.

## Chapters

<!-- ss:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `craft.02` | [Architecture decision records](01-architecture-decision-records.md) | practice | 1 |
<!-- /ss:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Software Architecture](../../systems/02-software-architecture/) | the decisions worth recording |
| [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/) | runbooks and postmortems, the how-to and the record of an incident |

## Company Relevance

| Company | Practice |
|---|---|
| Google | design docs before code; documentation as an engineering artifact |
| Amazon | written narratives for one-way-door decisions |
| Open source foundations | RFCs, PEPs, and KEPs as public decision records |
