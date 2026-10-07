# Field Craft: Discovery, POCs, Migrations, Escalations

Stage 10 of the FDE (inference) path. Stages 1 through 9 gave you the depth. This stage is how that depth turns into a customer outcome: asking the questions that size a deployment, running calls that end with decisions, writing a POC plan that can be failed, migrating traffic without regressions, and escalating so the performance team can act on the first read.

Everything here produces a written artifact. If a call or a week of work ends without one, it did not happen.

## 1. Discovery

Discovery has one output: a **deployment hypothesis** with numbers attached. Every question below exists because its answer changes a number in that hypothesis. Do not ask a question you cannot map to a decision.

Ask open questions about what they do today, not hypotheticals about what they would do. "What did last Tuesday's peak look like?" beats "What peak do you expect?" ([The Mom Test](#recommended-reading) is the long version of this rule.)

### Question bank

| Group | Questions | Changes |
|---|---|---|
| **Business outcome** | What product feature does this power? What happens to the business if it is slow, wrong, or down for an hour? Who complains first? What metric moves if this works? | Priority of latency versus quality versus cost; who signs off |
| **Workload shape** | Requests per second at average and peak? What does the daily and weekly curve look like? Can you export a day of request logs with token counts? Input and output token distributions (p50, p95, max)? How much of each prompt is a shared system prompt, few-shot block, or retrieved document? Streaming or not? Multi-turn sessions, and how long? Agent loops with tool calls? Batchable offline work mixed in? | GPU count, prefix caching value, chunked prefill, batch versus online split |
| **Latency SLOs** | Which latency does the user feel: **TTFT**, **TPOT** (time per output token), or end-to-end? What percentile, p50, p95, p99? Measured where: client, gateway, or server? What is the timeout today and what happens on timeout? | Batch size ceiling, replica count, speculative decoding, region placement |
| **Quality bar** | How do you know an answer is good today? Do you have an eval set, how large, and is it labelled? Who wrote the rubric? Do you use an LLM judge, and which model? What score is "good enough"? Any golden conversations that must never regress? | POC exit criteria, quantization tolerance, whether a fine-tune is in scope |
| **Model choice and fine-tune status** | Which model serves this today, closed or open, which version? Have you tried open models, which, and what broke? Do you have a fine-tune, LoRA adapters, how many? Is the training data yours to use? | Model size, multi-LoRA serving, fine-tuning scope |
| **Data residency, compliance, security** | Which certifications must the vendor hold: SOC 2 Type II, HIPAA (BAA), GDPR, ISO 27001? Is **zero data retention** required, including for logs and metrics? Any region constraint (EU, specific country)? Must traffic stay in your VPC or cloud account? Private networking (PrivateLink, VPC peering)? Who runs the security review and how long does it take? | Serverless versus dedicated versus BYOC; timeline |
| **Spend and cost target** | What do you spend per month on inference today? What is the target **cost per 1M tokens** or cost per request? Is spend growing faster than usage? Committed spend with another vendor? | Serverless versus dedicated breakeven, quantization, distillation |
| **Integration surface** | OpenAI SDK, Anthropic SDK, LangChain, a gateway (LiteLLM, OpenRouter, in-house)? Function calling, structured output or JSON schema, logprobs, n > 1, vision, embeddings? Retry and fallback logic today? | Migration effort, feature gaps to check before the POC |
| **Team and decision makers** | Who owns the integration day to day? Who decides, who can veto (security, finance, procurement)? Do they have an MLE, or is this an app team? Who is on call? | Call invitees, how much you build versus advise |
| **Timeline** | When does this need to be in production, and what drives that date? Any freeze windows, launches, renewals with the incumbent? | POC length, staffing |

Three questions to ask on every first call, even if time runs out:

1. Can you send a sample of real requests with token counts (sanitized is fine)?
2. How do you measure quality today?
3. What would make you say no to this project?

## 2. From answers to a deployment hypothesis

A hypothesis is a sized proposal you expect to be wrong by up to 2x and intend to correct with a benchmark. Write it down before the POC so the benchmark has something to confirm or refute. The formulas are derived in [Serving and Load](../../ml/04-llm-systems/serving-and-load/); this section applies them.

### Worked example: support copilot

Customer brief from discovery:

| Item | Value |
|---|---|
| Use case | Agent-assist copilot drafting replies inside a support tool |
| Traffic | 50 req/s peak (6 h/day), 15 req/s average |
| Tokens | 2,000 input (1,200 shared system prompt and policy text), 300 output |
| SLO | TTFT p95 < 500 ms, streaming; TPOT p95 < 40 ms (they read as it streams) |
| Quality | Today on a closed frontier API; eval set of 400 graded tickets; a 70B-class open model matched it in their spike |
| Constraints | SOC 2 Type II; no prompt logging |

Assume a Llama-3.3-70B-class dense model: 70B parameters, 80 layers, 8 KV heads, head dim 128. Hardware: H100 SXM 80 GB, about 3.35 TB/s HBM and about 1,979 dense FP8 TFLOPS peak.

**Step 1: token rates at peak.**

```text
prefill tokens/s = 50 x 2,000            = 100,000
uncached prefill = 50 x (2,000 - 1,200)  =  40,000   (shared prefix hits the cache)
decode tokens/s  = 50 x 300              =  15,000
```

**Step 2: compute.** Forward FLOPs per token are about 2N (attention FLOPs are small at 2k context).

```text
prefill: 40,000 x 2 x 70e9 = 5.6 PFLOP/s
decode : 15,000 x 2 x 70e9 = 2.1 PFLOP/s
total                      = 7.7 PFLOP/s
at ~40% of FP8 peak per GPU (~0.8 PFLOP/s): ~10 GPUs of pure compute
```

Without prefix caching, prefill alone is 14 PFLOP/s, nearly double the fleet. **Prefix caching is the largest single lever in this workload.** Confirm the shared prefix is byte-identical across requests: a timestamp or user name at the top of the system prompt destroys it.

**Step 3: concurrency (Little's law).** Requests in the system L = arrival rate x time in system.

```text
time in system = TTFT + 300 x TPOT = 0.5 + 300 x 0.040 = 12.5 s
L at peak      = 50 x 12.5 = ~625 concurrent sequences
```

**Step 4: KV cache memory.** Per token: 2 (K and V) x layers x KV heads x head dim x bytes.

```text
FP8 KV per token = 2 x 80 x 8 x 128 x 1 B = 160 KiB
per sequence     = 2,300 tokens x 160 KiB = ~360 MiB (full)
unique per seq   = 1,100 tokens x 160 KiB = ~170 MiB (shared prefix stored once per replica)
fleet at peak    = 625 x 170 MiB = ~105 GiB
```

A tensor-parallel-4 (TP4) replica has 320 GB of HBM. After ~70 GB of FP8 weights and ~30 GB of activations and runtime overhead, roughly 220 GB is left for KV. Memory is not the binding constraint; decode step time is.

**Step 5: decode step time per replica.** Decode is bandwidth-bound: each step reads the weights once and each sequence's KV.

```text
weights read : 70 GB / (4 x 3.35 TB/s x 0.7 efficiency)  = ~7.5 ms
KV read, B=200 seqs: 200 x 360 MiB / 9.4 TB/s            = ~8 ms
compute, B=200: 200 x 140 GFLOP / (4 x 0.8 PFLOP/s)      = ~9 ms (overlaps memory)
step time ~16-20 ms, plus interleaved prefill chunks -> TPOT ~25-35 ms
```

So about 200 sequences per TP4 replica fit under the 40 ms TPOT target. 625 / 200 = 3 to 4 replicas at peak.

**Step 6: TTFT check.** 800 uncached tokens x 140 GFLOP = 112 TFLOP on a TP4 replica at ~3.2 PFLOP/s is about 35 ms. A full cache miss (2,000 tokens) is about 90 ms. TTFT p95 under 500 ms holds if queueing stays bounded, which means keeping replicas below roughly 80% of saturation throughput and enabling **chunked prefill** so long prompts do not stall decode.

**Step 7: serverless versus dedicated.** Plug in current list prices; the numbers below are placeholders to show the method.

```text
monthly tokens = 15 req/s x 2,300 tok x 2.592e6 s = ~89B tokens
serverless     = 89,000 x $P per 1M tokens         (P = $0.90 -> ~$80k/month)
dedicated      = GPU-hours x $R per GPU-hour
                 peak 12 GPUs x 6 h + floor 4 GPUs x 18 h = 144 GPU-h/day
                 144 x 30 x $R                       (R = $4.00 -> ~$17k/month)
effective cost = $17k / 89,000 = ~$0.19 per 1M tokens, only if autoscaling holds that floor
```

**The hypothesis:**

| Decision | Proposal | Why | Risk to test |
|---|---|---|---|
| Model | 70B-class dense, customer's spike choice | Matched quality in their spike | Eval parity on all 400 tickets |
| Tier | Dedicated, autoscaling 1 to 4 replicas; serverless for week-1 evaluation | Steady volume and a p95 SLO | Scale-up time versus the morning ramp |
| GPUs | TP4 on H100, 3 replicas at peak (12 GPUs), range 10 to 16 | Steps 2 and 5 | Measured saturation point per replica |
| Quantization | FP8 weights and FP8 KV cache | Halves weight reads and KV memory versus BF16 | Eval delta must stay inside the agreed tolerance |
| Prefix caching | On; system prompt frozen and placed first | Halves prefill compute | Hit rate on real traffic |
| Chunked prefill | On | Protects TPOT when a long prompt arrives | p99 TPOT under mixed lengths |
| Speculative decoding | Test EAGLE or MTP drafts off-peak | Helps TPOT at low batch; gains shrink near saturation | Acceptance rate on support replies |

Every row in "risk to test" becomes a POC benchmark or eval.

## 3. Running a call

Rules for every call: send the agenda 24 hours ahead with the questions you need answered, name a note-taker (default: you), end with owners and dates, send notes within the same working day.

### Agenda templates

**Discovery (45 min)**

| Min | Item |
|---|---|
| 0-5 | Introductions, roles, confirm the goal of the call: understand the workload well enough to propose a deployment |
| 5-15 | Business outcome and what success looks like |
| 15-35 | Workload, SLOs, quality measurement, constraints (question bank, skip what they sent ahead) |
| 35-40 | Gaps: data you need (request sample, eval set) |
| 40-45 | Next steps: who sends what by when; propose the technical deep dive |

**Technical deep dive (60 min)**

| Min | Item |
|---|---|
| 0-5 | Recap discovery, confirm the hypothesis inputs are still right |
| 5-25 | Walk their architecture: request path, gateway, retries, where latency is measured |
| 25-45 | Present the deployment hypothesis and the math; invite them to break it |
| 45-55 | Feature gaps: tool calling, structured output, logprobs, regions |
| 55-60 | Agree POC scope candidates |

**POC kickoff (45 min)**

| Min | Item |
|---|---|
| 0-10 | Walk the written POC plan (section 4), line by line |
| 10-25 | Success criteria: confirm each metric, threshold, and how it is measured |
| 25-35 | Access, data transfer, environments, security sign-off |
| 35-45 | Owners, check-in cadence, readout date |

**POC readout (45 min)**

| Min | Item |
|---|---|
| 0-5 | One-slide verdict: criteria met, missed, or partial |
| 5-25 | Results per criterion with raw numbers and the config that produced them |
| 25-35 | What we would change for production; cost per 1M tokens at the measured operating point |
| 35-45 | Decision and next step: migrate, extend, or stop |

**Incident or escalation (30 min, can be called within the hour)**

| Min | Item |
|---|---|
| 0-5 | Impact: what users see, since when, how many requests |
| 5-15 | Timeline and what changed (their deploy, our deploy, traffic shift) |
| 15-25 | Current mitigation and next diagnostic step, with owners |
| 25-30 | Next update time; who is on point on each side |

During an incident, the customer needs an update cadence more than a root cause. Commit to "next update at 14:30" and keep it, even if the update is "no change".

### Note-taking template

```markdown
# <Customer> / <call type> / <YYYY-MM-DD>

Attendees: <name, role, side>
Goal of call: <one line>

## Facts learned
- <fact> (source: <person or doc>)

## Numbers
| Metric | Value | Confidence (measured / estimated / guessed) |

## Decisions
- <decision> (decided by <name>)

## Open questions
- <question> -> owner <name>, due <date>

## Action items
- [ ] <action> -> owner, due

## Hypothesis changes
- <what changed in the deployment hypothesis and why>
```

The **confidence** column is the most useful field. "Peak is 50 req/s" from a dashboard and from a guess are different inputs.

### Saying "I don't know"

Guessing on a technical call costs more than not knowing: the customer will build on the guess. Use a fixed form: what you know, what you do not, when they will hear back.

> "I know FP8 KV cache is supported on that engine. I don't know whether it is enabled for this model on our dedicated tier. I'll confirm with the performance team and reply by Thursday."

Then write it in the open questions list and reply by Thursday even if the answer is "still checking".

**Bring in an MLE** when the question is about quality: eval design, fine-tuning feasibility, why outputs differ. **Bring in the performance team** when you have exhausted configuration you control and have a repro (section 6). **Bring in research** when the ask needs a method that does not exist as a recipe yet. Do not bring specialists to a discovery call unless the customer brings theirs; it turns discovery into a lecture.

## 4. POC plan template

A POC exists to make a decision. Write criteria that can fail. "Evaluate latency" cannot fail; "TTFT p95 < 500 ms at 50 req/s open-loop for 30 minutes" can.

```markdown
# POC plan: <customer> / <workload>

## Decision this POC informs
<e.g. migrate the support copilot from <incumbent> to a dedicated deployment>

## Success criteria
| # | Criterion | Threshold | Measured how | Owner |
|---|---|---|---|---|
| 1 | Quality | >= baseline - 1.0 pt on 400-ticket eval, no golden-set regressions | Customer eval harness, same judge, same rubric | Customer MLE |
| 2 | TTFT p95 | < 500 ms at 50 req/s | Open-loop Poisson load, 30 min steady state, measured client-side | FDE |
| 3 | TPOT p95 | < 40 ms at 50 req/s | Same run | FDE |
| 4 | Error rate | < 0.1% non-2xx | Same run | FDE |
| 5 | Cost | <= $X per 1M tokens at measured operating point | GPU-hours x rate / tokens served | FDE + AE |
| 6 | Features | Tool calls and JSON schema parse on 100% of eval cases | Eval harness | Customer eng |

## Workload under test
- Request sample: <source, size, date range, sanitized how>
- Token distributions: input p50/p95/max, output p50/p95/max
- Shared prefix: <length, verified byte-identical>

## Configuration under test
- Model, revision, quantization, engine and version, GPU type and count, TP/PP/EP, key flags

## Benchmarks
| Run | Mode | Load | Purpose |
|---|---|---|---|
| A | Closed loop | Concurrency sweep 1, 8, 32, 128, 256 | Find saturation throughput per replica |
| B | Open loop | Poisson at 15, 35, 50, 65 req/s | SLO attainment at average, peak, and 30% over peak |
| C | Open loop | Replay of a real traffic hour | Realistic burstiness and length mix |
| D | Cold start | Scale 1 -> 3 replicas | Time to absorb the morning ramp |

## Timeline
| Week | Milestone |
|---|---|
| 1 | Access, data, serverless smoke test, eval baseline on incumbent |
| 2 | Dedicated deployment, runs A and B, eval on candidate config |
| 3 | Tuning, runs C and D, cost model |
| 4 | Readout |

## Exit criteria
- Success: criteria 1-4 and 6 met; 5 within 10% -> recommend migration
- Partial: quality met, latency missed -> one week extension with performance team
- Stop: quality gap > 3 pts after one prompt-adaptation iteration -> recommend fine-tune scoping or stop

## Owners and cadence
- Customer: <eng owner>, <decision maker>; Vendor: FDE, AE, perf contact
- Check-ins: Tuesdays and Fridays, 20 min, written update after each
```

**Closed loop versus open loop.** A closed-loop benchmark keeps N requests in flight and sends a new one when one finishes; it measures maximum throughput but hides queueing, because the load backs off when the server slows. An open-loop benchmark sends requests on a schedule (Poisson at a target rate) regardless of completions; it shows what users see when traffic does not wait. Use closed loop to find the saturation point and open loop to prove the SLO. Tools: [`vllm bench serve`](https://docs.vllm.ai/en/stable/cli/bench/serve.html) (supports a request rate), [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html), [GuideLLM](https://github.com/vllm-project/guidellm). Stage 8 covers both in depth.

## 5. Migration playbook: closed API to open model

Most migrations fail on quality, not latency, and most quality failures are formatting, not model capability.

| Step | Action | Done when |
|---|---|---|
| **1. Inventory** | List every call site: model, parameters, features used (tools, JSON mode, logprobs, n, vision, seed) | Spreadsheet with every call site and its parameters |
| **2. Endpoint swap** | Point the OpenAI-compatible client at the new `base_url` and model name behind a feature flag | A single request succeeds through the customer's real code path |
| **3. Template and parameter audit** | Check the items in the table below | Every item checked and noted |
| **4. Eval parity** | Run the customer's eval on incumbent and candidate with identical inputs; diff per case, not just aggregate | Delta inside the agreed tolerance; every golden case passes; regressions read by a human |
| **5. Prompt adaptation** | Fix regressions by prompt changes first, fine-tune only if prompt changes plateau | Delta inside tolerance or a scoped fine-tune proposal |
| **6. Shadow traffic** | Mirror a slice of production to the candidate, discard responses, log latency, errors, and outputs | 3+ days covering a weekly peak; no new error classes; latency within SLO |
| **7. Canary** | Route 1%, 10%, 50% of real traffic with automatic rollback triggers on error rate and latency | Each step holds for an agreed period with product metrics flat |
| **8. Cutover** | 100% to candidate; keep incumbent credentials and code path live | Two weeks clean |
| **9. Rollback plan** | Flag flip back to incumbent, tested during canary, with a named owner | Rollback executed once in staging and timed |

### Template and parameter differences that break parity

| Item | What goes wrong | Check |
|---|---|---|
| **Chat template** | Server applies a different template than the model was trained with; system prompt dropped or merged into the first user turn | Render the template locally from the model's `tokenizer_config.json` and compare to server-side token IDs |
| **Duplicate BOS** | Client prepends BOS and the template adds another | Inspect the first token IDs of a rendered prompt |
| **Stop tokens** | Generation runs past the turn end, or stops early on a token that appears in content | Compare `eos_token_id` lists in `generation_config.json` versus server config |
| **Sampling defaults** | Incumbent default temperature differs from the model card's recommended settings; `top_p`, `top_k`, repetition penalty unset | Pin every sampling parameter explicitly in client code |
| **Tool call format** | Model emits tool calls in its native format; parser on the server does not match the model family | Run every tool-using eval case and assert parsed `tool_calls` |
| **Structured output** | JSON mode on the incumbent is not schema-constrained decoding on the new server, or vice versa | Validate 100% of outputs against the schema |
| **Reasoning tokens** | Reasoning models emit thinking blocks the client does not strip, inflating output and cost | Check for reasoning fields or tags in raw responses |
| **Tokenizer** | Same text, different token count: context limits, `max_tokens`, and cost shift | Re-count eval prompts with the new tokenizer |
| **Context length** | Served max length is lower than the model card's | Query the server's model metadata; test the p99-length prompt |

## 6. Escalation and handoff

You escalate when you have done the work you can do. A report the performance team can act on without a meeting contains:

```markdown
# Escalation: <one-line symptom with a number>
e.g. "TPOT p95 doubles to 80 ms above 40 req/s on <model> dedicated, customer <X>"

Impact: <customer, users affected, SLO breached, since when>
Severity: <sev level and why>

## Environment
- Engine + version/commit: <...>
- Launch flags (full, verbatim): <...>
- Model + revision + quantization: <...>
- GPU type, count, TP/PP/EP, region, node IDs: <...>

## Traffic shape
- Rate: <req/s avg/peak>, arrival pattern: <steady/bursty>
- Input tokens p50/p95/max, output tokens p50/p95/max
- Prefix sharing: <length, hit rate if known>
- Streaming: <yes/no>; features: <tools, JSON schema, LoRA>

## Observed vs expected
| Metric | Expected | Observed | Source |

## Evidence
- Benchmark command and raw output: <link>
- Engine metrics over the window (queue depth, running/waiting seqs, KV usage, preemptions): <link>
- Trace or profile: <link>
- Minimal repro: <script that reproduces with synthetic or sanitized data>

## What I ruled out
- <hypothesis> -> <how ruled out>

## Ask
<what you need: root cause, config recommendation, patch, ETA>
```

The **"what I ruled out"** section separates an FDE escalation from a forwarded complaint. Hand-offs to an MLE follow the same shape with eval artifacts in place of traces (see the [role brief](README.md#what-you-hand-off-and-how)).

## 7. Common customer failure modes

| Symptom | Likely cause | First check |
|---|---|---|
| Quality drop after migration | Chat template mismatch or sampling defaults | Diff rendered token IDs; pin temperature and `top_p` |
| Quality drop after quantization | Outlier-sensitive layers or KV quantization on long context | Eval BF16 versus FP8 on the same cases; split long and short prompts |
| Outputs never stop or stop mid-sentence | Wrong stop tokens or `max_tokens` default | Inspect `finish_reason` distribution |
| Tool calls returned as plain text | Tool parser does not match the model family | Raw completion versus parsed response for one case |
| TTFT spikes when one long prompt arrives | No chunked prefill; long prefill blocks the batch | Engine config; correlate spikes with prompt length |
| TTFT high at moderate load | Queueing: replica past saturation, or autoscaler too slow | Waiting-queue metric; compare rate to closed-loop saturation |
| TPOT degrades as traffic grows | Batch too large for the TPOT target | Running sequences versus TPOT curve; cap max batch |
| Throughput falls, then requests restart | KV cache exhausted, preemption and recompute | Preemption counter, KV usage near 100% |
| Prefix cache hit rate near zero | Dynamic content (timestamp, user ID) at the start of the prompt | Diff the first 100 tokens of two requests |
| Cost overrun on dedicated | Low batch utilization: provisioned for peak, idle off-peak | GPU-hours versus tokens served by hour; autoscaling floor |
| Cost overrun on serverless | Volume crossed the dedicated breakeven, or output tokens grew (reasoning) | Monthly tokens versus breakeven; output length trend |
| Cold starts after idle | Scale-to-zero with large weights | Time from scale event to first 200; consider a minimum of one replica |
| Rate-limit errors at modest load | Account tier limits, not capacity | Response headers and tier limits |
| Nondeterministic outputs at temperature 0 | Batch-dependent numerics | Expected behavior; see [reproducibility](../../ml/04-llm-systems/serving-platforms.md#reproducibility-mechanism-availability-and-incentives) |
| Latency fine on server, slow for users | Client-side buffering, proxy without streaming, cross-region hops | Measure TTFT at the client and the server for one request |

## 8. Mock exercise

### Brief

> **Lexa** is a legal-tech company. Their contract-review feature sends a contract section and a 3,000-token instruction block to a closed frontier API and asks for a JSON list of risky clauses. Contract sections are 6,000 to 14,000 tokens; outputs are 400 to 1,200 tokens. Traffic is 3 req/s average, 8 req/s peak, weekdays 9:00 to 18:00 CET, near zero at night. A nightly job re-reviews 40,000 sections when the instruction block changes. Spend is $60k/month and growing 15% monthly. Their largest customers are EU law firms; contract text must stay in the EU and must not be retained. They have 150 hand-labelled sections with expert annotations. The VP of Engineering wants to cut cost by half by Q2; the CISO has not been involved yet. They say "latency is fine today, around 8 seconds per section."

### Deliverable

Write two documents (target two pages total):

1. **Discovery notes** using the [note template](#note-taking-template): facts from the brief with confidence marked, the ten questions you would still ask (each mapped to the decision it changes), and the risks you see.
2. **POC plan** using the [POC template](#4-poc-plan-template), including:
   - A deployment hypothesis with the math: model size class, tier for the online path, tier for the nightly job, GPU count, quantization, and whether prefix caching applies.
   - The online and batch workloads sized separately.
   - Success criteria for quality on their 150 labelled sections, schema validity, latency, residency, and cost per 1M tokens against their current spend.
   - The point at which you involve the CISO, and what you bring.

### Done when

- Every success criterion has a threshold, a measurement method, and an owner.
- The hypothesis states its token rates, compute estimate, and cost comparison, each with the formula visible.
- The plan names what you would do if the quality criterion fails.
- The nightly job is priced on a batch or async surface, not on the online deployment.
- A peer reading only your documents could run the POC without asking you a question.

## Recommended reading

| Resource | Why | Cost |
|---|---|---|
| [Dev versus Delta: Demystifying engineering roles at Palantir](https://blog.palantir.com/dev-versus-delta-demystifying-engineering-roles-at-palantir-ad44c2a6e87) | The original FDE ("Delta") model: one customer, many capabilities, versus product engineers | Free |
| [The Palantirization of everything](https://a16z.com/the-palantirization-of-everything/) (a16z) | Why startups copy the FDE model, and where it breaks: reusable primitives versus bespoke work | Free |
| *The Mom Test* by Rob Fitzpatrick | Asking about past behavior instead of opinions; the discovery rule in section 1 | **Paid** (book) |
| [Service Level Objectives](https://sre.google/sre-book/service-level-objectives/), *Site Reliability Engineering* ch. 4 | SLIs, SLOs, percentiles, and why you measure what users see | Free |
| [Implementing SLOs](https://sre.google/workbook/implementing-slos/), *The Site Reliability Workbook* | Turning an SLO into alerts and error budgets for the production step | Free |
| [Postmortem Culture](https://sre.google/sre-book/postmortem-culture/), *Site Reliability Engineering* ch. 15 | Blameless incident write-ups for escalation calls | Free |
| [LLM Serving Platforms](../../ml/04-llm-systems/serving-platforms.md) | Engine and platform layers you will be asked to compare | Free |

## Related curriculum

- [Role brief](README.md): what the company sells, who owns what
- [Serving and Load](../../ml/04-llm-systems/serving-and-load/): the formulas behind section 2
- [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/): eval design for POC quality criteria
- [Model Routing & Cascades](../../ai-platform-engineering/11-model-routing-and-cascades/): when the answer is two models, not one
