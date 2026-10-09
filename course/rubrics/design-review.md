# Design review rubric

Used by the review modules (`review.*`) and by any ADR you ask a peer to read.
Read the document once end to end, then answer each line y or n. A no is a
review comment you write down, not a failure of the reviewer.

- [ ] The problem, its users, and the constraints are stated before any solution.
- [ ] At least two options are compared on the same criteria, with numbers where numbers exist.
- [ ] The chosen option names what it gives up, and why that cost is acceptable.
- [ ] Every interface between components names its contract (an `.h`, `.proto`, OpenAPI path, or schema).
- [ ] Failure modes are listed with how each is detected and how the system degrades.
- [ ] Capacity and latency budgets are estimated, with the arithmetic shown.
- [ ] Rollout and rollback are described, including data and contract migrations.
- [ ] Open questions are listed with an owner.
