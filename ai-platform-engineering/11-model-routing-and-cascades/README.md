# Model Routing & Cascades

## Overview

- **Primary references** (all free): [RouteLLM paper](https://arxiv.org/abs/2406.18665) + [LMSYS write-up](https://www.lmsys.org/blog/2024-07-01-routellm/), [FrugalGPT](https://arxiv.org/abs/2305.05176), [LLMRouterBench](https://arxiv.org/abs/2601.07206) (Findings of ACL 2026)
- **Supplementary**: [FireRouter docs](https://docs.fireworks.ai/ecosystem/firerouter/overview), Fireworks' [*The Frontier Isn't a Model, It's a Router*](https://fireworks.ai/blog/the-frontier-isnt-a-model-its-a-router), [OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router), [Fireworks reinforcement fine-tuning docs](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models)
- **Prerequisites**: [LLM Evaluation](../09-llm-evaluation/) (harnesses, precision and recall), [LLM Systems & Inference](../../ml/04-llm-systems/) (prefill, KV cache, prompt caching), [Training & Frameworks](../01-training-and-frameworks/) (SFT, LoRA)
- **Estimated time**: 1-2 weeks at 8-10 hrs/week
- **Sources reviewed**: September 26, 2026. Vendor numbers and product behaviour change; treat every figure below as dated.

## Key Takeaways

- **Routing is three decisions, not one.** Which model answers, which endpoint serves it, and what happens after a failure. Products blur them; keep them apart when you evaluate.
- **The oracle proves opportunity, not a router.** An oracle picks the best model after seeing every outcome. A real router chooses before. The distance between them is the whole engineering problem.
- **The common failure is model recall, not difficulty estimation.** Routers tend to fail because they do not reach for the model that would have succeeded, even when it is in the pool.
- **Most of the value is in two or three complementary models.** Coverage of different capabilities beats a long candidate list.
- **A model switch can cost more than it saves.** Changing models abandons the prompt cache. A router that ignores cache state optimises the sticker price and raises the bill.
- **Escalate on category or on a verifier, not on model confidence alone.** Confidence is produced by the thing you are trying to check, and untrusted input can move it.
- **The router is a small share of the cost and decides the rest.** That makes precision the adoption metric and the labelled set the durable asset.

## How to Study

- Take an eval set you already have and run three models over it. Build the success matrix (task by model), then compute best single model, oracle, and best pair by hand. The gap between the first two is your opportunity.
- Write the simplest router that could work: a rule on one input feature. Measure how often it selects the non-default model when that model was the only one that would have succeeded. That number is model recall.
- Price one long session twice: once staying on a single model with warm cache, once switching models halfway. Use real cached-input rates.
- Write down your escalation policy before you build anything, including what happens when the router itself errors.

---

# Concepts & Techniques

## Core Insight

No single model is best at everything, and prices differ by more than an order of magnitude. If you could always send each request to the cheapest model that would succeed, you would get higher accuracy than the best model at a fraction of its cost. That statement is true and nearly useless, because it describes an oracle. Routing is the discipline of approximating the oracle with information available before the answer exists, and of knowing how much of the gap you actually closed.

## 1. Three Decisions That Share a Name

| Decision | Question | Typical owner |
|----------|----------|---------------|
| **Model selection** | Which model should answer this request? | A router: rules, a classifier, or a learned scorer |
| **Provider selection** | Which endpoint serves the chosen model? | A gateway, on price, availability, or latency |
| **Fallback** | What runs after an error or timeout? | An ordered list supplied by the caller |

Pinning a model name does not pin the provider, the quantization, or the cache. Using a gateway does not mean a learned router chose the model. When someone says "we use routing," ask which of the three they mean.

## 2. Router Families

| Family | Decides | Signal | Extra cost | Main weakness |
|--------|---------|--------|------------|---------------|
| **Rules and categories** | Before generation | Deterministic features: file paths, tool names, request type | None | Rules drift and do not generalise |
| **Predictive router** | Before generation | A model scores the prompt | One small inference | Misjudges prompts unlike its training data |
| **Cascade** | After generation | A scorer judges the cheap answer, then escalates | The cheap attempt is paid even when discarded | Latency, and a scorer that can be wrong |
| **Verifier-gated** | After generation | A deterministic check: tests pass, query executes, schema validates | The cheap attempt plus the check | Only works where a verifier exists |
| **Session-level router** | Per task or per user turn | Task features plus cache state | One decision per turn | Early choices shape every later step |

**Predictive routing: RouteLLM.** A router estimates how much a strong model would beat a weak one on this prompt, and a threshold turns that score into a choice. The project trained four routers on Chatbot Arena preference data: similarity-weighted ranking, matrix factorization, a BERT classifier, and a causal LLM classifier. Reported results against a GPT-4 baseline at 95% of its quality were cost reductions of over 85% on MT Bench, 45% on MMLU, and 35% on GSM8K. The spread is the lesson: savings depend on the workload. Routers trained only on Arena data performed poorly on MMLU until about 1,500 in-domain examples were added, under 2% of the training set.

**Cascades: FrugalGPT.** Call models in order from cheap to expensive and stop when a scoring function accepts the answer. The paper reports matching GPT-4 with up to 98% cost reduction, or 4% better accuracy at equal cost, on its evaluated tasks. A cascade pays for every rejected attempt, so it wins only when the cheap model is accepted most of the time.

**A retry loop is a cascade in time.** Returning an error to the same model and retrying, as in the [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/), escalates effort. A cascade escalates tier. Both need a hard budget.

## 3. The Oracle and What It Proves

An oracle router assigns each task to the model with the best measured outcome, breaking ties on cost. It needs the answer key, so it is an upper bound and nothing more.

Fireworks published an oracle analysis on DeepSWE v1.1: 113 agentic software-engineering tasks, 18 models, four rollouts per pair.

| Policy | Success | Cost per task |
|--------|---------|---------------|
| Best fixed model (GPT-6 Astra) | 74.1% | $6.52 |
| Claude Opus 5, fixed | 73.8% | $11.84 |
| Oracle over open-weight models only | 90.3% | $1.45 |
| Oracle over all 18 models | 97.6% | $1.88 |

Portfolio results from the same analysis: the best pair added 13.1 points over the best single model, the best trio reached 91.2%, and going from three models to eighteen added only 6.4 more points.

Three things to take from it:

1. **Complementarity is real.** Models fail on different tasks, so the union of a pool covers far more than its best member.
2. **Curation beats count.** Most of the gain arrives by the third model.
3. **None of this measures a router.** The analysis modelled the simplest policy, one model chosen at the start of a task and held. Whether a deployed router captures the opportunity is a separate measurement.

The same post reports production numbers: 2,334 internal coding sessions over four weeks at $7.42 per session, against $15.81 for a fixed Claude Opus 5, a 53% reduction. That is a cost comparison. Before treating any routing claim as a quality claim, ask for task success on the same sessions.

## 4. Why Real Routers Fall Short

LLMRouterBench re-evaluated the field under one framework: over 400K instances, 21 datasets, 33 models, and 10 routing baselines. Its findings are the counterweight to every oracle chart:

- Many routing methods perform about the same once evaluated identically.
- Several recent approaches, including commercial routers, fail to reliably beat the **best single model**.
- A large gap to the oracle persists, driven mainly by **model-recall failures**: the right model was in the pool and the router did not choose it.
- The embedding backbone matters little. Larger pools show diminishing returns against careful curation.

**Baselines every router must beat**: the best single model, a random split at the same cost, and a rule a person could write in ten minutes. If the router does not beat all three on your traffic, it is complexity without a return.

**Difficulty is the wrong target.** Predicting that a prompt is hard does not tell you which model handles it. A router needs to learn what each model is good at, which is why in-domain labelled outcomes matter more than general preference data.

## 5. Granularity

| Unit | Example | Trade-off |
|------|---------|-----------|
| Per request | Classify each API call | Most flexible; breaks cache and consistency in multi-step work |
| Per user turn | One model handles the turn and its tool calls | Keeps a tool loop coherent; cannot correct mid-turn |
| Per task or session | Pick once, hold | Best cache reuse; a wrong first choice is expensive |

In agent workloads, early decisions change later inputs. A model that reads the wrong file produces a different context for every following step, so per-step accuracy numbers do not compose. Evaluate complete trajectories.

A router that scores only the last message has a specific blind spot: when that message is "continue" or a tool result, it carries almost no signal about the task.

## 6. Cache-Aware Routing

Prompt caching bills previously seen prefix tokens at a reduced rate. Caches are per model, and usually per provider. Switching models discards the cache and pays full prefill on the new one.

Illustrative arithmetic, with made-up round rates:

| Option for the next turn, 100K tokens of context | Input rate | Input cost |
|---|---|---|
| Stay on the strong model, cache hit | $0.50 per 1M (cached) | $0.05 |
| Switch to a cheap model, cache miss | $0.50 per 1M (uncached) | $0.05 |
| Switch back to the strong model later, cache expired | $5.00 per 1M (uncached) | $0.50 |

The cheap model saved nothing on input for that turn, and the round trip cost ten times the turn it tried to optimise. A router has to price the state it abandons, not only the model it selects. This is why session-level routers hold a choice across a turn, and why "cheapest model per request" is often the most expensive policy for long contexts.

Record in every evaluation: cache hits, cache writes, the resolved model, and the resolved provider.

## 7. Escalation Policy

Escalation is the rule for leaving the cheap path. There are four ways to trigger it, and they fail differently.

| Trigger | How it works | Fails when |
|---------|--------------|------------|
| **Confidence** | Escalate when the model or a scorer reports low confidence | Confidence is miscalibrated, or the input manipulates it |
| **Category** | Named categories always escalate, whatever the model thinks | The category list is incomplete |
| **Verifier** | A deterministic check rejects the output | No verifier exists for the task |
| **Human override** | A person corrects the route | Volume exceeds attention |

**Category beats confidence on untrusted input.** If the content being routed can contain instructions, a confidence score is attacker-influenced. A rule such as "security and schema changes always get the strong reviewer" cannot be argued with, because no model evaluates it. This is the same principle as the deterministic gate in the [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/): the control sits outside the model.

**Decide the failure direction explicitly.**

| System | Router errors, then | Why |
|--------|--------------------|-----|
| Review or analysis fan-out | **Fail open**: run everything | Worst case is a known cost, never silent loss of coverage |
| Anything that executes or writes | **Fail closed**: refuse | Worst case must not be an unreviewed action |

**Treat the escalation rate as a metric.** A cost model that assumes 20% of calls escalate is wrong the day the rate becomes 40%. Alert on it, and cap it with a budget.

**An override is also a label.** When a person corrects a route, record the pair. The escape hatch and the training data are the same mechanism.

## 8. Economics: the Small Decision That Steers the Bill

Work from tokens, not from a price list. Every per-request figure is tokens times a rate, so every figure can be re-derived when a rate changes.

Worked example: a system with six specialised subagents, of which two are relevant to a typical request. Rates are illustrative.

| Call | Tokens in / out | Rate per 1M in / out | Cost |
|------|-----------------|----------------------|------|
| Frontier subagent | 15K / 1K | $2.50 / $15.00 | $0.0525 |
| Small tuned subagent | 9K / 1K | $0.20 flat | $0.0020 |
| Prompted classifier, small model | 19K / 0.1K | $0.10 flat | $0.0019 |
| Tuned classifier, small model | 2K / 0.1K | $0.10 flat | $0.0002 |

| Configuration | Per request | Derivation |
|---------------|-------------|------------|
| Run all six on the frontier model | $0.315 | 6 × $0.0525 |
| Prompt-routed, two frontier subagents | $0.107 | $0.0019 + 2 × $0.0525 |
| Tuned router, cascade, 20% escalate | $0.024 | $0.0002 + 2 × (0.8 × $0.0020 + 0.2 × $0.0525) |

Three independent levers produced that range:

1. **Routing**: which calls run at all.
2. **Tier**: which model runs them.
3. **Context**: how many tokens each call carries. Instructions and examples re-sent on every call are what tuning moves into weights.

The router is about 1% of the final figure and determines the other 99%.

**Precision pays twice.** A wrong route costs the call and produces a wrong output that someone has to read. At 20 outputs a week, 90% precision is two wrong ones every week, and 99% is one every five weeks. That is the difference between a tool people read and a tool people mute.

**Coverage and precision have to land together.** Fixing coverage alone means more calls, a larger bill, and more wrong outputs at once.

**Check how a tuned model is served before you trust a per-token number.** Some platforms serve adapters only on dedicated deployments billed by GPU time. At $7 per GPU-hour, that is about $5,040 a month whether or not traffic arrives, and it changes the break-even completely.

## 9. Tuning the Router: a Ladder

Two choices are independent: how much of the model changes, and what signal it trains against.

| How much changes | What it is | Use when |
|------------------|-----------|----------|
| **LoRA** | Base frozen, small low-rank adapters trained; artifact in megabytes | Narrow specialisation |
| **Full-parameter** | Every weight updated; a full checkpoint | New capability or domain |

| Signal | Needs | Example for a router |
|--------|-------|----------------------|
| **SFT** | Input and correct output pairs | Request and the route a person chose |
| **DPO** | Chosen and rejected pairs | The route that was overridden against its replacement |
| **RFT** | A scorer that returns a reward from 0 to 1 | Did the downstream task succeed on this route? |

Climb in order, and only when the harness shows the rung below has stopped paying:

1. **Prompted classifier.** No weights change, no training data required. It sets the baseline.
2. **SFT on historical labels.** Available as soon as labelled routes exist.
3. **Preference and reward tuning.** DPO on overrides, RFT on outcomes. These need live traffic, so they cannot be pulled forward.

**RFT in one paragraph.** You supply prompts and an evaluator, and training increases the score. Gold completions are optional, which makes RFT the fit when a task is verifiable and you lack reference answers. The evaluator can be rules, tests, a judge model, or a mix. The risk is reward hacking: the model learns to satisfy the evaluator instead of the task. Inspect rollouts, and stop when quality regresses or the evaluator saturates. A good eval harness is already most of a reward function.

**Be honest about label quality.** Labels that come from the subset of users who bother to label skew toward trained users, and some are wrong. Measure both before training on them.

## 10. Evaluating a Router

| Metric | Asks |
|--------|------|
| **Coverage** | Did a route run at all? |
| **Recall** | When a specialist was needed, was it selected? |
| **Precision** | When a route fired, was it correct? |
| **Latency** | Did the answer arrive while it was still useful? |
| **Cost per successful task** | Including retries, cache misses, and escalations |

Rules for a fair comparison:

- **Compare against the human baseline, measured.** If people route by hand today, the bar is their error rate, not perfection.
- **Agreement is not correctness.** Precision against human labels measures agreement. Send disagreements to adjudication.
- **Evaluate whole sessions**, held out from anything the router trained on.
- **Version the policy.** Router version, threshold, candidate models, prompts, and fallback order are part of the configuration under test. An untracked policy change invalidates the comparison.
- **Replay history first.** A harness that replays past traffic gives a baseline before any production change.

**What survives a model swap**: the harness, the labelled set, and the baseline. A tuned adapter is tied to a base model. Build the measurement first and keep it provider-agnostic.

## 11. Worked Product: FireRouter

A managed router from Fireworks, reviewed from public documentation.

| Aspect | Documented behaviour |
|--------|----------------------|
| Granularity | Chooses a model for each new user turn; that model handles the turn, including its tool calls |
| Model IDs | `firerouter`, a family alias such as `firerouter/opus`, or an exact pin such as `firerouter/claude-opus-5-5` |
| Custom routes | Up to eight models, `firerouter/{model1}/{model2}/...`; the first is primary |
| Preferences | `max-intelligence`, `more-intelligence`, `balanced`, `more-savings`, `max-savings`; support varies by harness |
| Reporting | The response `model` field names the model that served the request |
| Billing | You pay for the model that served each turn; closed models bill through your own provider credential; cached input bills at the serving model's cached rate |

What is not published, as reviewed: the features the router uses, its training data, and a measured comparison against the oracle. Read it as an instance of a session-level, cache-aware router, and apply section 10 before adopting it.

Notice the design choices against the sections above: per-turn granularity (section 5), cache awareness (section 6), and a user-set preference in place of a hidden threshold (section 2).

---

## Decision Cheat Sheet

| Situation | Reach for |
|-----------|-----------|
| A deterministic feature already separates the traffic | Rules |
| Most requests are easy, a few need a strong model | Predictive router with a tuned threshold |
| A cheap check can verify the answer | Verifier-gated cascade |
| Long agent sessions with large context | Session-level routing; price the cache |
| Untrusted content decides the route | Category escalation, never confidence alone |
| No labelled routes yet | Prompted classifier plus a replay harness |
| Labelled routes exist | SFT on them, LoRA first |
| A verifiable outcome and no gold answers | RFT with your eval as the reward |
| A vendor shows an oracle chart | Ask for the deployed router's numbers on matched tasks |

## Patterns Worth Internalizing

- **Upper bounds are not results.** An oracle, a best-case benchmark, and a vendor chart are all statements about opportunity.
- **Measure before you route.** The baseline comes first, because improvement cannot be shown against a number nobody took.
- **Price the state you abandon.** Cache, context, and consistency are costs of switching.
- **Put the control outside the model.** Category rules and verifiers cannot be persuaded.
- **The labelled set is the asset.** Models are replaced every few months; labels and harnesses are not.
- **Small, complementary, curated.** Two or three well-chosen models beat a long list.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Harness design, precision and recall, judges | [LLM Evaluation](../09-llm-evaluation/) | Measuring a routing policy |
| Prefill, KV cache, serving cost | [LLM Systems & Inference](../../ml/04-llm-systems/) | Why a switch is expensive |
| SFT, DPO, LoRA, small models | [Training & Frameworks](../01-training-and-frameworks/) | Tuning the router and the tiers |
| Retrieval over a large candidate set | [Retrieval & RAG](../07-retrieval-and-rag/) | Routing across hundreds of specialists is ranking |
| Deterministic gates, bounded retry | [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/) | Verifier-gated escalation in a real build |
| Least privilege, untrusted input | [Authorization & Access Control](../08-authorization-and-access-control/) | Why confidence is not a control |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| Fireworks | Session-level, cache-aware routing across open and closed models | FireRouter |
| OpenRouter | Gateway with provider selection, fallbacks, and optional model selection | Auto Router |
| LMSYS / Anyscale | Preference-trained predictive routing | RouteLLM |
| Stanford | Learned cascades with a scoring function | FrugalGPT |
| Any agent product | Category escalation and fail-open fan-out | Specialised subagents behind a classifier |
