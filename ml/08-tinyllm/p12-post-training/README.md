# Part 12: Post-Training (optional)

Turning the capstone base model into an instruction follower. Supervised fine-tuning with chat rendering, assistant-only loss, and packing with document masks; preference optimization with DPO (and IPO); GRPO with verifiable rewards and rollouts through your own server; and distillation. The whole part is optional and produces the second capstone, C2.

**Course passes**: 10, optional (L12.1 to L12.4, milestone MS-C2)

**Before you start**: the solve part `S-M10b` (Lagrangians and KL-regularized objectives) and the `kl_k3` estimator of `M11.1` in [Information Theory](../../../math/11-information-theory/); LoRA from [Part 6](../p06-objectives/); the trained capstone model (C1).

## Key ideas

- **SFT** is next-token prediction on curated conversations, with the loss masked to assistant tokens only.
- **DPO** turns a KL-regularized reward objective into a classification loss on preference pairs, with no reward model.
- **GRPO** samples a group of answers per prompt and uses the group-normalized reward as the advantage, with a KL penalty to the reference model.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L12.1` | SFT: chat rendering, assistant-only loss, packing with document masks | build | 10, optional |
| `L12.2` | DPO (+ IPO variant) | build | 10, optional |
| `L12.3` | GRPO with verifiable rewards, rollouts through the learner's server | build | 10, optional |
| `L12.4` | Distillation (forward/reverse KL, on-policy) | build | 10, optional |

## Chapters

<!-- ss:chapters -->
No chapters yet: they arrive with authoring batch B12 (course/DESIGN.md 9). `ss lint --fix-index` then fills this table from the registry.
<!-- /ss:chapters -->

## Going further

- Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155); Rafailov et al., [DPO](https://arxiv.org/abs/2305.18290); Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO).
- Hugging Face [TRL](https://github.com/huggingface/trl) (the oracle for the loss fixtures); the [Reinforcement Learning](../../03-reinforcement-learning/) topic for policy gradients.
