# Forward Deployed Engineer, Inference: Role Brief

Stage 0 of the FDE (inference) path. This page says what a training and inference cloud sells, what a **Forward Deployed Engineer (FDE)** owns inside one, and how the rest of the path builds the depth to do the job. Company facts below were checked against primary sources on October 6, 2026. Product surfaces change monthly; recheck before a call.

The target role: FDE (AI) at a company shaped like Fireworks AI, Together AI, or Baseten. You sit with the customer's engineers, go as deep or as wide as their problem needs, take over a workload from an MLE or a performance engineer when it is yours to carry, and relay well-formed problems back to research.

## What the company sells

All three sell GPU time wrapped in software. The wrapper differs: a per-token API, a dedicated deployment, a training job, or raw cluster capacity. An FDE has to know which wrapper fits a workload and what each one costs the customer in control and money.

| Surface | Fireworks AI | Together AI | Baseten |
|---|---|---|---|
| **Serverless per-token API** | Pay-per-token serverless models ([quickstart](https://docs.fireworks.ai/getting-started/quickstart)) | Serverless inference, OpenAI-compatible ([quickstart](https://docs.together.ai/docs/quickstart), [product](https://www.together.ai/serverless-inference)) | Model APIs, compatible with OpenAI Chat Completions and Anthropic Messages (beta); tool calling, structured output, JSON mode ([Model APIs](https://docs.baseten.co/inference/model-apis/overview)) |
| **Dedicated / on-demand deployment** | On-demand deployments on dedicated GPUs with autoscaling ([on-demand quickstart](https://docs.fireworks.ai/getting-started/ondemand-quickstart), [why on-demand](https://fireworks.ai/blog/why-gpus-on-demand)) | Dedicated endpoints, single-tenant GPUs ([docs](https://docs.together.ai/docs/dedicated-endpoints)) | Dedicated deployments of open or custom models ([first model](https://docs.baseten.co/development/model/build-your-first-model)) |
| **Fine-tuning** | Managed: SFT, DPO, LoRA. Training API: LoRA and full-parameter, custom loops, RL (GRPO and others via SDK). RFT listed ([intro](https://docs.fireworks.ai/fine-tuning/finetuning-intro)) | SFT, LoRA (default) or full fine-tuning, preference tuning with DPO ([overview](https://docs.together.ai/docs/fine-tuning-overview), [DPO](https://docs.together.ai/docs/fine-tuning/preference-tuning)). RL not listed in fine-tuning docs at review | Training Jobs and Loops ([training](https://docs.baseten.co/training/overview), [loops](https://docs.baseten.co/loops/overview)). Supported method list **not confirmed** |
| **Batch / async** | Batch inference ([docs](https://docs.fireworks.ai/guides/batch-inference)) | Batch API ([docs](https://docs.together.ai/docs/batch-inference)) | Async inference with webhooks on any dedicated deployment, queue up to 72 h ([docs](https://docs.baseten.co/inference/async)). No separate batch product **confirmed** |
| **Embeddings / rerank** | Embeddings and reranking ([docs](https://docs.fireworks.ai/guides/querying-embeddings-models)) | Embeddings and rerank API; rerank models need a dedicated endpoint ([rerank](https://docs.together.ai/docs/inference/embeddings/rerank)) | BEI engine for embeddings and reranking ([BEI](https://docs.baseten.co/engines/bei/overview)) |
| **Custom model deploy** | Custom model upload on on-demand deployments ([announcement](https://fireworks.ai/blog/custom-models-h100s-on-demand-deployments)); multi-LoRA serving ([blog](https://fireworks.ai/blog/multi-lora)) | Custom model upload ([docs](https://docs.together.ai/docs/custom-models)); Dedicated Container Inference for your own Docker image ([docs](https://docs.together.ai/docs/dedicated-container-inference)) | **Truss** (package model code) and **Chains** (multi-step, multi-model pipelines) ([Chains](https://docs.baseten.co/development/chain/overview), [Truss CLI](https://docs.baseten.co/reference/cli/truss/overview)) |
| **GPU clusters** | Not a headline product at review; **not confirmed** | H100, H200, B200, GB200 clusters with storage for training or large batch ([docs](https://docs.together.ai/docs/gpu-clusters-overview), [product](https://www.together.ai/gpu-clusters)) | Not offered as a raw cluster product; **not confirmed** |
| **BYOC / VPC / self-hosted** | Bring Your Own Cluster on customer Kubernetes, Private Preview, enterprise only, inference only ([BYOC](https://docs.fireworks.ai/ecosystem/integrations/byoc/overview)); SageMaker as a compute option ([blog](https://www.fireworks.ai/blog/aws-sagemaker)) | Together Enterprise Platform in customer VPC or on-prem ([announcement](https://www.together.ai/blog/introducing-the-together-enterprise-platform), [architecture](https://docs.together.ai/docs/together-deployments)) | Baseten Cloud, single-tenant isolated VPC, self-hosted in customer VPC, hybrid ([hosting options](https://docs.baseten.co/hosting-options/overview)) |
| **Engine work** | **FireAttention** custom kernels: V3 targets AMD MI300, V4 targets NVFP4 on B200 ([V3](https://fireworks.ai/blog/fireattention-v3), [V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200)) | Together inference engine, **Together Kernel Collection**, **ATLAS** runtime-learning speculator ([ATLAS](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas), [research](https://www.together.ai/blog/foundational-research-powering-efficient-inference-at-scale)) | **Baseten Inference Stack**: Engine-Builder-LLM on TensorRT-LLM, BIS-LLM for MoE, speculative decoding (EAGLE, MTP, n-gram), KV-aware routing ([guide](https://www.baseten.co/resources/guide/the-baseten-inference-stack/), [BIS-LLM](https://docs.baseten.co/engines/bis-llm/overview)) |

Notes on the table:

- Vendor speedup numbers (for example FireAttention versus vLLM, ATLAS throughput) are the vendor's own benchmarks on the vendor's chosen workload. Treat them as hypotheses to reproduce on the customer's traffic, the same rule as in [LLM Serving Platforms](../../ml/04-llm-systems/serving-platforms.md).
- Compliance claims change and are contractual. Fireworks publishes SOC 2 Type II and HIPAA status and a no-logging default for open-model prompts ([data security](https://docs.fireworks.ai/guides/security_compliance/data_security)). Together documents its posture at [privacy and security](https://docs.together.ai/docs/privacy-and-security). For any customer, get the current report from the trust center, not from a blog.
- Together also sells **Sandbox** and **Managed Storage** ([products](https://www.together.ai/products)); Baseten documents a **Frontier Gateway** for model labs ([overview](https://docs.baseten.co/overview)). These are outside this path.

**What is common across all three:** an OpenAI-compatible request surface, a serverless tier for evaluation, a dedicated tier for SLOs, fine-tuning feeding deployment, and a proprietary engine layer on top of or beside vLLM, SGLang, or TensorRT-LLM. The differentiator they sell is the engine and the people. You are part of the people.

## What an FDE owns

| Role | Owns | Does not own | Typical artifact |
|---|---|---|---|
| **FDE** | The customer's technical outcome: discovery, deployment hypothesis, POC, migration, first-line diagnosis, translating customer problems into internal tickets | Engine internals, model research, pricing authority | Discovery notes, POC plan, benchmark report, migration runbook, escalation ticket |
| **MLE (customer-facing or platform)** | Fine-tuning runs, eval pipelines, model packaging, quality debugging | Contract, account relationship | Training config, eval report, model card |
| **MTS / performance engineer** | Engine, kernels, scheduler, quantization recipes, per-model tuning | Customer relationship, POC scope | Engine flags, kernel patch, perf regression fix |
| **Research scientist** | New methods: speculators, quantization schemes, post-training recipes | Production SLOs | Paper, prototype, recipe |
| **Solutions architect** | Reference architectures, integration patterns across many accounts, pre-sales technical fit | Deep per-customer build | Architecture diagram, RFP answers |
| **Account executive** | Commercial relationship, pricing, contract, renewal | Technical claims | Order form, mutual action plan |

Titles overlap between companies. At some, the FDE and solutions architect are one person; at others, the FDE writes production code in the customer's repo. Ask in the interview which of the rows above the role covers.

### What you hand off and how

A handoff is accepted when the receiver can act without calling you. Bring evidence, not a narrative.

| To | When | Bring |
|---|---|---|
| **MTS / performance** | Latency or throughput misses SLO after config tuning you can do yourself | Engine version and commit, full launch flags, GPU type and count, traffic shape (QPS, input/output token histograms, prefix share), benchmark command and raw output, a trace or profile, minimal repro script |
| **MLE** | Quality gap after migration or fine-tune not converging | Eval set (or a sanitized sample), metric definition, baseline score versus current score, prompts with chat template applied, sampling params, training config and loss curves |
| **Research** | A customer problem no current recipe solves (a new architecture, a domain where speculators fail, a quantization that breaks a capability) | The pattern across customers, the measured gap, why the existing recipes were ruled out, the business value of solving it |
| **Solutions architect** | A pattern repeats across three or more accounts | The three accounts, the common integration, what you built each time |
| **Account executive** | Scope or commercial change: new workload, capacity request, SLO change | Revised deployment hypothesis with GPU count and cost per 1M tokens, risk list, timeline |

The reverse direction matters too. When you take over from an MLE or MTS, ask for the same artifacts. If they do not exist, writing them is your first deliverable.

## The customer journey

| Step | Customer question | FDE artifact |
|---|---|---|
| **Evaluate** | Is an open model good enough for us, and is this vendor fast enough? | Discovery notes, quality and latency smoke test on serverless, initial deployment hypothesis |
| **POC** | Does it hit our SLO and quality bar on our data? | POC plan with success criteria, benchmark harness, eval harness, POC readout |
| **Migrate** | How do we move from OpenAI or Anthropic APIs without regressions? | Migration runbook: endpoint swap, template diffs, eval parity report, shadow-traffic plan, rollback plan |
| **Production** | Will it stay up and stay fast? | SLO dashboard, alert thresholds, on-call contacts, escalation path, capacity plan |
| **Scale / optimize** | Can we pay less or serve more? | Cost per 1M tokens analysis, fine-tune or distillation proposal, quantization and speculative decoding experiments with eval deltas |
| **Renew** | Was it worth it? What is next year's workload? | Quarterly business review: SLO attainment, spend trend, incidents and resolutions, next workload hypothesis |

Every artifact in the right column is a document someone else can act on. If the journey stalls, the stalled step usually lacks its artifact.

## How to use this path

The stages are listed in `paths/fde-inference/path.tsv`. The `ss` CLI reads that file.

```bash
practice/bin/ss learn fde-inference            # list stages and your progress
practice/bin/ss learn fde-inference 3          # read stage 3
practice/bin/ss learn fde-inference --done 3   # mark stage 3 done
```

A PDF of the whole path is attached to each GitHub release.

| Stage | Title | You leave able to |
|---|---|---|
| 0 | Role brief | Describe what the company sells and where the FDE sits (this page) |
| 1 | History of language models | Place any model a customer names in its lineage and explain why it exists |
| 2 | Math of transformers | Derive attention, parameter counts, FLOPs, and KV cache size on a whiteboard |
| 3 | Architecture variants | Explain how GQA, MLA, MoE, sliding window, and long-context schemes change serving cost |
| 4 | Frameworks (PyTorch, JAX, Keras 3) | Read and modify a customer's model code in any of the three |
| 5 | Loading models from the Hub | Load, inspect, convert, and debug weights, tokenizers, and chat templates |
| 6 | Training and post-training | Scope a SFT, LoRA, DPO, or RL fine-tune and read its failure modes |
| 7 | Inference engine internals | Explain paged KV cache, continuous batching, chunked prefill, quantization, speculative decoding |
| 8 | Serving and load | Size a deployment from a traffic shape and run open and closed loop benchmarks |
| 9 | Full stack | Own an endpoint end to end: gateway, autoscaling, observability, cost |
| 10 | Field craft | Run discovery, write a POC plan, migrate a customer, escalate well ([field-craft.md](field-craft.md)) |

### Which stage answers which customer question

| Customer says | Stage that lets you answer |
|---|---|
| "Why is DeepSeek cheaper to serve than a dense model of the same size?" | 3 (MoE, MLA), 7 |
| "Our fine-tune works in Transformers but outputs garbage on your endpoint." | 5 (chat templates, tokenizers), 6 |
| "We wrote the model in JAX. Can you serve it?" | 4, 5 |
| "How many GPUs do we need for 50 req/s?" | 2 (FLOPs, KV size), 8 |
| "Latency spikes every few minutes." | 7 (chunked prefill, preemption), 8 |
| "Should we do LoRA or full fine-tuning? Or RL?" | 6 |
| "Who gets paged when this breaks at 3 a.m.?" | 9 |
| "Why should we trust an open model over GPT or Claude?" | 1, 10 |

Stages 1 through 9 build depth; stage 10 turns it into customer outcomes. If you already work in inference, read stage 10 first to see what the depth is for, then fill the gaps it exposes.

**Done when** for this stage: without notes, you can explain to a peer which product surface (serverless, dedicated, batch, fine-tune, BYOC) you would propose for three workloads: a 5 req/s internal chatbot, a nightly 200M-token classification job, and a regulated healthcare assistant that cannot leave the customer's VPC.

## Related curriculum

- [LLM Systems & Inference](../../ml/04-llm-systems/)
- [LLM Serving Platforms](../../ml/04-llm-systems/serving-platforms.md)
- [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/)
- [Model Routing & Cascades](../../ai-platform-engineering/11-model-routing-and-cascades/)
