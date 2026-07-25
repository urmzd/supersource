# Grounded SQL Agent

## Overview

- **Problem**: turn natural-language questions into SQL against a database the
  model has never seen, execute it safely, and prove the answers are right.
- **Runnable**: [`safety_gate.py`](safety_gate.py) -- the deterministic SQL
  classifier, standard library only, `python safety_gate.py`. No API key needed.
- **Prerequisites**: SQL, and a working idea of what an LLM tool call is.
- **Estimated time**: 1-2 days

## Key Takeaways

- **Grounding beats model size.** Measured on the same questions, schema
  discovery took an open model from 8/10 to 10/10 and matched a frontier model's
  accuracy at roughly a fifteenth of the cost.
- **Never let the model be the only safety boundary.** A prompt saying "read
  only" is a preference. A classifier that refuses to execute anything else is a
  control.
- **Errors are the correction signal.** Returning a database error to the model
  as a tool error, inside a fixed retry budget, is what turns a failed query into
  a successful one without any prompt engineering.
- **Score result sets, not SQL text.** Two correct queries rarely look alike, and
  string-matching SQL measures conformity rather than correctness.
- **Trust what ran, not what was said.** The executed SQL and returned rows are
  the source of truth; the model's final message is metadata.

## How to Study

- Run the safety gate first, then try to get a write past it. The `BLOCKED` list
  is a catalogue of the ways a plausible classifier fails.
- Read the ablation table before the architecture. It is the evidence that
  justifies most of the design, and without it the schema tools look like
  over-engineering.
- Then design the eval harness for your own domain before writing any agent
  code. If you cannot state how you would score an answer, you cannot tell
  whether a change helped.

---

# Concepts & Techniques

## The Problem

A user asks "what are the top 5 best-selling genres by total sales?" against an
arbitrary relational database. The system must produce SQL, run it, and return
rows. Three things make it harder than a prompt template:

1. The model does not know the schema. Table and column names are the entire
   difficulty, and guessing them is the dominant failure mode.
2. Generated SQL is untrusted input that reaches a database.
3. "It looks right" is not a measurement. Something has to score it.

## 1. Grounding: the measured case

The most important result in this study is an ablation, not an architecture. The
same agent was run two ways over the same ten questions: *raw* (one query tool,
told nothing about the tables) and *grounded* (schema discovery tools available).

| Configuration | Executed | Correct | p50 latency | Cost/query |
|---------------|----------|---------|-------------|------------|
| Open 120B, raw | 9/10 | 8/10 | 6086 ms | $0.0012 |
| Open 120B, grounded | 10/10 | **10/10** | 3640 ms | $0.0014 |
| Frontier model, raw | 9/10 | 9/10 | 3998 ms | $0.0113 |
| Frontier model, grounded | 10/10 | **10/10** | 3820 ms | $0.0202 |

Two readings, and the second one matters more:

- **Grounding is what closes the gap.** The open model gains two questions and
  gets *faster*, because it stops burning attempts on guessed column names.
- **The benchmark flatters the raw configurations.** These questions ran against
  a well-known public sample schema that both models have partly memorised. The
  raw runs are leaning on that memory, which a private customer schema will not
  provide. The honest conclusion is not "raw is nearly as good" but "this
  benchmark cannot tell you how bad raw is."

That second point is the transferable one. **Know which part of your benchmark
your system is cheating on.** A public dataset measures capability and leaks
memorisation at the same time, and only the ablation exposes which you measured.

The economic conclusion holds regardless: a grounded open model matched a
grounded frontier model on accuracy at about 1/15 the cost per query.

## 2. Schema discovery as tools, not context

Two tools, deliberately split:

- `list_schemas()` -- browse table names and row counts. On a small database it
  returns full detail in the first call.
- `fetch_schemas(tables)` -- full columns, types, primary keys, and foreign keys
  for named tables.

Stuffing the whole schema into the system prompt works until it doesn't: at
hundreds of tables it exhausts context and buries the relevant tables in noise.
The tool split lets the model narrow first and fetch detail second.

**Where this stops working**: at warehouse scale, `list_schemas()` is itself too
large. The next step is a retrieval layer over table and column descriptions,
fetching only the top-k relevant tables per question -- which makes this a RAG
problem, and hands it to [Retrieval &
RAG](../../ai-platform-engineering/07-retrieval-and-rag/). Knowing the point at
which your design breaks is part of the design.

## 3. Bounded self-correction

When generated SQL fails, the error goes back to the model as a tool error, and
it tries again within a fixed budget.

```
generate -> classify -> execute
                |            |
                |            +-- database error --> return to model as tool error
                +-- blocked ---> return the reason as a tool error
                                          |
                                   retry, up to max_retries
```

Two properties make this work rather than loop:

- **The error is specific.** "No such column: Genre.Title" is a usable
  correction signal. "Query failed" is not.
- **The budget is hard.** Retries, model requests, tool calls, total tokens,
  result rows, per-request timeout, and a wall-clock deadline are all capped.
  Without a ceiling, a confused agent is an unbounded bill.

In the ablation above, the single raw-configuration failure was exactly this
budget doing its job: the model exhausted its retries guessing at column names
and the request was cut off rather than allowed to spiral.

## 4. Safety as layers, with the model outside all of them

| Layer | What it does | Can an application bug bypass it? |
|-------|--------------|-----------------------------------|
| Deterministic classifier | Blocks writes, admin, multi-statement; caps rows | Yes |
| Read-only connection | Re-checks at the data boundary (`PRAGMA query_only`, read-only DSN) | Yes |
| `SELECT`-only database role | Refuses writes at the engine | **No** |

The database role is the only real control. The other two exist to fail fast,
to give the model a correctable error, and to protect deployments where the role
cannot be set. Ordering them by strength, rather than treating them as three
equal "safety features," is what keeps the design honest.

The classifier contains no model, which is the point: it cannot be
prompt-injected and it behaves identically every run. Its interesting content is
lexical, not logical:

```sql
SELECT * FROM t WHERE name = 'Begin Again'    -- contains BEGIN, is a read
SELECT 'a;b' AS x                             -- contains ';', is ONE statement
SELECT * INTO archive FROM users              -- leads with SELECT, WRITES
SELECT * FROM (SELECT ... LIMIT 5)            -- has LIMIT, is NOT bounded
```

Every one of those is a way a keyword-matching classifier gets it wrong. The
implementation scans once into two parallel strings of equal length -- a
*cleaned* copy safe to execute, and a *masked* copy where literal contents are
blanked -- so text inside a string can never lie to a check. And it is an
**allowlist**: anything not positively recognised as a read is refused. A
denylist of bad keywords fails open on the first syntax nobody anticipated.

## 5. Evaluation by result-set equivalence

The evaluator compares **returned rows** against gold rows, not SQL text against
gold SQL. It ignores column naming and row order where those carry no meaning,
and accepts projection differences only when every gold value is reproduced with
the same row count.

This matters because there are many correct queries for one question. Scoring
SQL strings measures whether the model writes SQL the way you do; scoring result
sets measures whether it answered the question.

Alongside correctness, every answer records rows returned, attempts, latency,
token usage, model calls, and estimated cost -- so a prompt change that buys two
points of accuracy for triple the cost is visible as exactly that trade.

## 6. Trust the execution, not the narration

The executed SQL and the rows it returned are stored by the tool and treated as
the source of truth. The model's final structured message is useful metadata and
nothing more.

An agent that reports "I found 5 genres" after its query returned 3 rows is
reporting an intention. Systems that trust the narration are the ones that
confidently return numbers nobody ran.

## Build Log

1. **The eval set first.** Ten questions with gold answers, and a scorer that
   compares result sets. Before any agent code, decide what "right" means.
2. **The safety gate second**, with its own test suite. It is pure and
   deterministic, so it is the cheapest part of the system to get right, and
   everything downstream can assume it.
3. **A minimal agent**: one `execute_query` tool, no schema access. This is the
   raw baseline, and it exists to be beaten.
4. **Measure the baseline.** 8/10 on an open model. Without this number, every
   later improvement is a story rather than a result.
5. **Add schema tools**, re-run the same eval. 10/10, and faster.
6. **Add bounded retry** so tool errors become correction signals rather than
   failures, with hard caps on every resource.
7. **Add the read-only connection layer**, re-checking at the data boundary.
8. **Run the ablation grid**: raw versus grounded, open versus frontier, same
   questions. This is what turned "grounding seems better" into a cost argument.
9. **Record the artifacts.** Eval report, latency benchmark, per-question
   results, committed. A claim without a committed artifact is a memory.

The sequence is the lesson: the measurement infrastructure was built before the
thing being measured. Every architectural claim here is backed by a number that
existed before the architecture did.

## Technique Catalog

| Technique | When to apply |
|-----------|---------------|
| Ablate the grounding | Any RAG or tool-augmented system, to prove retrieval earns its cost |
| Ask what the benchmark leaks | Any public dataset a model may have memorised |
| Deterministic gate before execution | Any generated code, query, or command that will run |
| Allowlist, not denylist | Any classifier where unknown input must fail closed |
| Lex before you match keywords | Any rule applied to code or query text |
| Tool errors as correction signals | Any agent that can retry |
| Hard resource ceilings | Every agent loop, without exception |
| Score outputs, not artifacts | Any task with many correct forms |
| Trust the execution record | Any agent that both acts and narrates |
| Cost and latency per answer | Any LLM feature that will run more than once |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Eval harnesses, scoring, LLM-as-judge | [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) | The theory this study applies |
| Schema retrieval at warehouse scale | [Retrieval & RAG](../../ai-platform-engineering/07-retrieval-and-rag/) | Where schema discovery becomes retrieval |
| Least privilege, defence in depth | [Authorization & Access Control](../../ai-platform-engineering/08-authorization-and-access-control/) | Why the database role is the real control |
| Tool loops, streaming, token budgets | [Streaming & SSE](../../ai-platform-engineering/03-streaming-sse/) | The delivery half of the same agent |
| Test doubles and deterministic tests | [The Testing Mentality](../../software-craftsmanship/03-testing-mentality/) | Testing a system with a stochastic component |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| AI infrastructure vendors | Grounded open models versus frontier APIs on cost | The economic argument |
| Data platforms | Natural-language querying over customer schemas | Safety and schema scale |
| Any agent product | Tool loops, retry budgets, injection resistance | Bounded autonomy |
| Any Staff+ AI role | Measuring before architecting | Evidence over intuition |
