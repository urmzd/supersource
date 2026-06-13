# Software Craftsmanship

The engineering practices, principles, and culture that separate senior/staff engineers from everyone else. How to write software that survives contact with reality.

## Overview

- **Primary references**:
  - *The Pragmatic Programmer* by Hunt & Thomas (20th Anniversary Edition) -- recommended purchase, the foundational text on professional software development
  - *Software Engineering at Google* by Winters, Manshreck, Wright -- [free online](https://abseil.io/resources/swe-book)
- **Supplementary**: *Clean Code* by Robert Martin, *A Philosophy of Software Design* by John Ousterhout, Martin Fowler's [Refactoring Catalog](https://refactoring.com/catalog/)
- **Prerequisites**: Some professional programming experience (this is a "how to think" track, not a "how to code" track)
- **Estimated time**: 3-4 weeks at 6-8 hrs/week

## Key Takeaways

- Good software is not about clever code -- it's about managing complexity over time
- The Pragmatic Programmer teaches individual craft; SWE at Google teaches organizational craft at scale
- Every practice exists because someone shipped a bug, lost a week, or burned out without it
- Staff+ engineers are distinguished by judgment calls about trade-offs, not raw coding ability

## How to Study

- Read one chapter per day from each book -- they complement each other perfectly
- After each chapter, identify one thing you can apply to your current project
- The Pragmatic Programmer is best read cover-to-cover; SWE at Google can be read selectively

---

# Part 1: The Pragmatic Programmer

## Core Insight

Software development is a craft. Pragmatic programmers take responsibility for their work, think critically about what they're doing, and continuously adapt their approach based on feedback.

## 1. A Pragmatic Philosophy

**Key principles**:
- **It's your life**: take charge of your career and your code. Don't say "I can't" -- say "I can, but here's what it costs"
- **The cat ate my source code**: take responsibility. Provide options, not excuses
- **Software entropy**: broken windows theory -- don't leave bad code unfixed, it gives permission for more
- **Good enough software**: know when to stop. Perfect is the enemy of shipped
- **Your knowledge portfolio**: invest regularly, diversify, review and rebalance

**How this appears at Staff+ level**: you set the quality bar for your team. If you tolerate broken windows, everyone does.

## 2. A Pragmatic Approach

**Key principles**:
- **DRY (Don't Repeat Yourself)**: every piece of knowledge has a single, unambiguous representation. Not just code -- also schemas, documentation, build scripts
- **Orthogonality**: components should be independent. Change one without affecting others. Test: if I change X, does Y break?
- **Reversibility**: keep decisions soft. Use configuration, dependency injection, feature flags. Assume requirements will change
- **Tracer bullets**: build thin vertical slices that work end-to-end. Not prototypes -- real code, minimal but complete
- **Prototypes**: throwaway code to explore ideas. Unlike tracer bullets, prototypes get discarded
- **Estimation**: learn to give estimates in ranges. Track your estimates vs actuals to calibrate

**Worked example**:
> Tracer bullet vs prototype: building a new API gateway. A *tracer bullet* would be one real endpoint, with real auth, real logging, real deployment -- but only one route. A *prototype* would be a script that tests latency between services, then gets thrown away.

## 3. The Basic Tools

**Key ideas**:
- **Power of plain text**: store knowledge in plain text when possible -- it survives everything
- **Shell games**: master the command line. Compose small tools. Automate everything you do twice
- **Version control**: everything goes in version control. Infrastructure, configs, docs, not just code
- **Debugging**: don't panic. Reproduce it first. Binary search the problem space. Read the error message

## 4. Pragmatic Paranoia

**Key principles**:
- **Design by Contract**: preconditions, postconditions, invariants. Define what a function promises
- **Dead programs tell no tales**: crash early. Don't rescue and continue with corrupted state
- **Assertive programming**: use assertions for things that "can't happen" -- they can and will
- **Resource management**: who allocates, deallocates. RAII, try-with-resources, context managers

## 5. Bend or Break

**Key principles**:
- **Decoupling**: law of Demeter -- "don't talk to strangers". Minimize what each module knows about others
- **Transforming programming**: think of programs as data pipelines, not as objects calling each other
- **Events and reactivity**: reduce coupling via events, pub/sub, reactive streams

## 6. Concurrency & While You Are Coding

**Key ideas**:
- **Temporal coupling**: identify what must be sequential vs what can be parallel
- **Refactoring**: refactor early, refactor often. Martin Fowler's catalog. Don't refactor and add features simultaneously
- **Testing**: test early, test often, test automatically. Tests are the first users of your code
- **Property-based testing**: generate random inputs, check invariants -- finds bugs that example-based tests miss
- **Naming things**: names reveal intent. If you can't name it, you don't understand it

---

# Part 2: Software Engineering at Google

## Core Insight

Software engineering is programming integrated over time. Google's lessons are about what happens when code must survive years, be maintained by thousands of engineers, and serve billions of users.

## 7. Culture

**Key concepts**:
- **What is software engineering?** -- not just writing code; it's writing code that must be maintained. Time, scale, and trade-offs distinguish engineering from programming
- **How to work well on teams** -- humility, respect, trust (HRT). Hide nothing, share early, accept feedback
- **Knowledge sharing** -- readability reviews, go/links, documentation as code, internal tech talks
- **Engineering for equity** -- build for everyone, test with diverse inputs, avoid encoding bias

**How this appears at Staff+ level**: you're responsible for team culture. If knowledge is siloed, that's your problem.

## 8. Processes

**Key concepts**:
- **Style guides and rules** -- consistency beats personal preference. Google's style guides exist to make code *readable by others*, not *writable by you*
- **Code review** -- every change is reviewed. Reviews are about readability and correctness, not gatekeeping. Keep changes small
- **Documentation** -- treat docs like code: version it, review it, test it, deprecate it
- **Testing overview** -- small tests (unit), medium tests (integration), large tests (end-to-end). Pyramid shape: many small, few large
- **Deprecation** -- removing old systems is as important as building new ones. Budget for it. Make it incremental

**Key insight**: code review at Google serves three purposes: correctness, knowledge transfer, and establishing norms. The social function matters as much as the technical one.

## 9. Testing at Scale

**Key concepts**:
- **Unit testing** -- fast, isolated, deterministic. Test behavior, not implementation. Google's Testing on the Toilet one-pagers
- **Test doubles** -- fakes (preferred at Google) vs mocks vs stubs. Avoid mock-heavy tests -- they're brittle and don't test real behavior
- **Larger testing** -- integration, end-to-end, load testing. Slower but catch interface bugs. Keep the test pyramid shape
- **Hyrum's Law** -- "with a sufficient number of users, all observable behaviors of your system will be depended on." Test for what you promise, not what happens to work

**Hyrum's Law in practice**: if your API returns JSON keys in sorted order (even though the spec doesn't require it), someone will depend on that ordering. When you stop sorting, their code breaks.

## 10. Tools and Infrastructure

**Key concepts**:
- **Version control and branch management** -- monorepo at Google. Trunk-based development. Feature flags over feature branches
- **Build systems** -- Bazel, reproducible builds, hermetic builds. Build once, cache forever
- **Code search** -- critical infrastructure. Every engineer can search all of Google's code instantly
- **Static analysis** -- compiler warnings, linters, type checkers. Make it automatic. Don't require humans to catch what machines can
- **Dependency management** -- the hardest problem in software engineering at scale. Diamond dependency problem. Minimum version selection
- **Continuous integration** -- presubmit checks, post-submit testing, release qualification
- **Large-scale changes (LSCs)** -- tools for making changes across thousands of files/repos atomically

**Staff+ insight**: the build system and CI pipeline are force multipliers. Investing a week to speed up CI by 30% pays for itself in a month across 100 engineers.

---

## Technique Catalog

| Principle | Source | When to Apply |
|-----------|--------|---------------|
| DRY | Pragmatic Programmer | When you find yourself copying logic (not just code) |
| Tracer bullets | Pragmatic Programmer | Starting a new project or major feature |
| Hyrum's Law | SWE at Google | Designing APIs, planning migrations, deprecating features |
| Small CLs | SWE at Google | Every code review (keep changes under ~200 lines) |
| Test pyramid | SWE at Google | Designing test strategy for a service |
| Broken windows | Pragmatic Programmer | Deciding whether to "fix it later" (don't) |
| Reversibility | Pragmatic Programmer | Making architectural decisions (keep them reversible) |
| Deprecation budget | SWE at Google | Planning roadmaps (explicitly budget for cleanup) |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Code review, testing | [Systems & Architecture](../systems/) | System design interviews test whether you'd build maintainable systems |
| Refactoring | [Algorithms](../algorithms/) | Refactoring algorithm code for clarity without changing behavior |
| DRY, orthogonality | [Software Architecture](../systems/02-software-architecture/) | Architecture patterns formalize these principles |
| CI/CD, build systems | [Cloud Native](../systems/03-cloud-native/) | Practical implementation of these processes |
| Deprecation, LSCs | [Observability](../systems/04-observability/) | Need observability to safely deprecate |
| Docs, diagrams, testing as artifacts | [Diagramming & Operations](../diagramming-and-operations/) | The hands-on counterpart: produce the C4 models, ADRs, tests, and K8s/Kafka topologies these principles call for |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Google | Directly -- their engineering culture IS this book. Readability reviews, Hyrum's Law | Culture, process |
| Anthropic | Production quality code, safety-critical engineering, thoughtful design | Pragmatic approach |
| Meta | Move fast with stable infrastructure. Code review culture | Testing, CI |
| Stripe | API design excellence, backwards compatibility, documentation | DRY, contracts |
| Amazon | Operational excellence, ownership, working backwards | Responsibility, estimation |
| Netflix | Freedom and responsibility, chaos engineering | Pragmatic paranoia |
| All Staff+ roles | Every company expects Staff+ engineers to drive engineering excellence | All principles |
