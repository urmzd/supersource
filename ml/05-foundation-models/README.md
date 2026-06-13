# Foundation Models & Architectures

## Overview

- **Primary reference**: *AI Engineering: Building Applications with Foundation Models* by Chip Huyen (O'Reilly, 2025) -- the definitive text on building *with* foundation models; free companion resources in [chiphuyen/aie-book](https://github.com/chiphuyen/aie-book)
- **Supplementary**: Stanford CRFM [*On the Opportunities and Risks of Foundation Models*](https://arxiv.org/abs/2108.07258) (free, the paper that named the field), [HuggingFace Transformers](https://huggingface.co/docs/transformers) + [Diffusers](https://huggingface.co/docs/diffusers) docs (free), Jay Alammar's [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) (free), Karpathy's [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (free), Lilian Weng's [*What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (free)
- **Prerequisites**: [Deep Learning](../02-deep-learning/) (transformers, attention, generative models), [Linear Algebra](../../math/03-linear-algebra/), [Probability](../../math/07-probability-statistics/)
- **Estimated time**: 4-5 weeks at 10-12 hrs/week

## Key Takeaways

- A **foundation model** is one pretrained on broad data at scale, then *adapted* to many downstream tasks -- the shift Chip Huyen calls **AI engineering** (building *with* models) vs ML engineering (building models)
- The 2018 transformer has **converged to a recipe**: decoder-only, pre-norm, RMSNorm, rotary positions, SwiGLU MLPs, grouped-query attention -- Llama, Gemma, Qwen, Mistral, and DeepSeek differ mostly at the margins
- **Attention is the architectural battleground**: MHA → MQA → GQA → MLA trades quality for KV-cache size; sliding-window and sparse variants trade global context for linear cost
- **Everything is tokens**: text (BPE), images (patches), and audio (neural-codec or spectrogram frames) all become token sequences a transformer consumes -- which is why one architecture went multimodal
- **Diffusion** is the other foundation-model family: instead of autoregressive next-token prediction, it learns to *denoise*. Modern diffusion is a transformer too (DiT), trained with flow matching

## How to Study

- Read Chip Huyen for the *engineering* view (adaptation, evaluation, serving cost), the original papers for the *architecture* view
- Build a tiny GPT from scratch (Karpathy) so the recipe in §2 is concrete, not vocabulary
- For each model family, read its **model card and tech report** -- that's where the real architectural choices (and their justifications) live
- Treat text, audio, vision, and diffusion as four instantiations of one idea (sequence modeling), not four separate fields

---

# Concepts & Techniques

## Core Insight

Before ~2020 you trained a model *per task*. Foundation models invert this: pretrain once on internet-scale data to learn general representations, then *adapt* (prompt, RAG, fine-tune) to thousands of tasks. The transformer is the substrate because it scales predictably and is modality-agnostic -- give it a sequence of tokens, any tokens, and it learns the structure. The whole landscape below is variations on "how do we tokenize the input, how do we attend over it, and do we generate autoregressively or by denoising."

## 1. What Is a Foundation Model

**Stanford CRFM 2108.07258 + Chip Huyen *AI Engineering***

**Key ideas**:
- **Pretraining + adaptation**: self-supervised pretraining (next-token, masked, or denoising) learns general features; adaptation specializes them
- **Emergence & scaling**: capabilities appear with scale (in-context learning, chain-of-thought) -- ties to [Deep Learning §11](../02-deep-learning/) scaling laws
- **The AI-engineering shift**: most practitioners now *adapt* existing models rather than train them. The new stack is **prompt engineering → RAG → fine-tuning → agents → dataset engineering**, with **latency and cost** as first-class constraints (Huyen's framing)
- **Adaptation ladder** (cheap → expensive): prompting → few-shot → RAG → PEFT/LoRA → full fine-tune → continued pretraining

## 2. Transformer Architecture Variants

**The 2018 paper, and what 7 years of iteration changed**

**Key ideas**:
- **Three shapes**: *encoder-only* (BERT -- understanding), *decoder-only* (GPT/Llama/Gemma -- generation, now dominant), *encoder-decoder* (T5, Whisper -- seq2seq, translation/ASR)
- **Pre-norm vs post-norm**: modern models put LayerNorm *before* the sublayer (pre-norm) for stable deep training
- **RMSNorm** replaces LayerNorm (cheaper, no mean subtraction) -- Llama, Gemma
- **Rotary Position Embedding (RoPE)** replaces learned/sinusoidal positions -- rotates Q/K by position, extrapolates to longer context; ALiBi is the linear-bias alternative
- **Gated MLPs**: SwiGLU / GeGLU replace the ReLU FFN -- `(Swish(xW) ⊙ xV)W₂`, better quality per parameter
- **Tokenization**: BPE (GPT), SentencePiece/Unigram (Llama, Gemma) -- subword units; vocab size trades sequence length for embedding size
- **Mixture-of-Experts (MoE)**: replace the dense MLP with routed experts -- more parameters, same per-token FLOPs (Mixtral, DeepSeek, Gemma 4 MoE)

**The modern decoder-only recipe** (Llama/Gemma/Qwen converged here): decoder-only + pre-norm + RMSNorm + RoPE + SwiGLU + GQA, optionally MoE.

## 3. Attention Mechanisms

**The architectural battleground -- mostly a fight over the KV cache**

| Mechanism | KV heads | Tradeoff |
|-----------|----------|----------|
| **MHA** (multi-head) | = query heads | Best quality, biggest KV cache |
| **MQA** (multi-query) | 1 (shared) | Tiny KV cache, some quality loss |
| **GQA** (grouped-query) | a few groups | The standard compromise (Llama 2+, Gemma) |
| **MLA** (multi-head latent) | low-rank latent | DeepSeek: compress KV into a latent, big cache savings |

**Other axes**:
- **Scaled dot-product** (the core): `softmax(QKᵀ/√d)V`, O(n²) in sequence length
- **Sliding-window / local attention**: each token attends to a fixed window (Mistral, Gemma alternate local/global layers) -- linear in length
- **Sparse / block attention**: attend to a structured subset (Longformer, BigBird)
- **Cross-attention**: queries from one stream attend to keys/values from another -- decoder→encoder (T5), text→image (diffusion conditioning)
- **Linear / sub-quadratic**: Performer, and the SSM cousins (Mamba) that drop attention entirely for linear recurrence
- **FlashAttention**: not a new *mechanism* but the systems realization that makes exact attention memory-efficient (see [LLM Systems §5](../04-llm-systems/))

## 4. Modalities: Text vs Audio vs Vision

**One architecture, four tokenizers**

**Key ideas**:
- **Text**: BPE/SentencePiece subwords → embedding table. The original setting
- **Vision**: **patchify** the image into a grid of patches, linearly embed each as a token (ViT). Pixels → patch tokens
- **Audio**: two routes --
  - *Spectrogram frames*: mel-spectrogram → frame tokens (Whisper's encoder)
  - *Neural audio codecs*: EnCodec / SoundStream quantize audio into discrete codes (RVQ), so audio becomes a token stream a transformer generates (audio LMs, TTS)
- **The unification**: once every modality is tokens, a single decoder can ingest interleaved text+image+audio tokens -- this is how models went **multimodal**
- **Fusion patterns**: *dual-encoder* contrastive (CLIP -- align image and text embeddings); *vision-language* (project image features into the LLM's token space -- LLaVA, PaliGemma); *natively multimodal* (Gemma 3/4, Gemini -- trained on mixed modalities from the start, incl. native audio in Gemma 4 E2B/E4B)
- **Text vs audio, concretely**: text is discrete and low-rate; audio is continuous and high-rate (16k+ samples/sec), so audio needs a codec/spectrogram front-end to get to a manageable token rate, and latency/streaming matter far more

## 5. Diffusion Models

**The non-autoregressive foundation-model family -- HF [Diffusers](https://huggingface.co/docs/diffusers)**

**Key ideas**:
- **Forward process**: gradually add Gaussian noise over T steps -- `q(x_t | x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t)I)`
- **Reverse process**: learn to denoise. Train a network `ε_θ(x_t, t)` to predict the noise; loss = `E‖ε − ε_θ(x_t, t)‖²` (DDPM)
- **Score view**: `ε_θ` is (up to scale) the score `∇ log p(x_t)`; sampling is following the score back to data (score-based / SDE)
- **Latent diffusion** (Stable Diffusion): diffuse in a VAE's compressed latent space, not pixels -- far cheaper. Decode the final latent to an image
- **Classifier-free guidance (CFG)**: `ε̂ = ε_uncond + w·(ε_cond − ε_uncond)` -- trade diversity for prompt adherence
- **DiT (Diffusion Transformer)**: replace the U-Net with a transformer over latent patch tokens; condition on timestep/class via **adaptive LayerNorm (adaLN-zero)**. Scales like LLMs
- **Flow matching / rectified flow**: instead of the noisy DDPM chain, learn a **velocity field** `v_θ(x_t,t)` along the straight path `x_t = (1−t)x_0 + t·x_1`; loss = `E‖v_θ − (x_1 − x_0)‖²`. Fewer, straighter sampling steps. Used by **Stable Diffusion 3**, **Flux.1**, Lumina
- **MMDiT (multimodal DiT)**: separate text and image token streams with joint attention (SD3, Flux)
- **Samplers**: DDIM, DPM++ -- ODE solvers that cut sampling from ~1000 steps to ~20

**Why it matters**: diffusion dominates image/video/audio generation; the architecture (transformer + flow matching) is converging with the LLM stack, so the two families increasingly share infrastructure.

## 6. The Current Model Landscape (2026)

**Open-weight families** (read the tech report + model card):

| Family | Notable variants | Architectural notes |
|--------|------------------|---------------------|
| **Gemma** (Google) | 3 (270M/1B/4B/12B/27B), **3n** (E2B/E4B, MatFormer + per-layer embeddings, edge), **4** (E2B/E4B/26B-MoE/31B-dense, native audio + video), PaliGemma, MedGemma, CodeGemma, ShieldGemma, RecurrentGemma | multimodal, local+global attention, single-GPU focus |
| **Llama** (Meta) | 3.x, 4 (MoE) | the reference open decoder-only recipe |
| **Qwen** (Alibaba) | dense + MoE, VL, audio | strong multilingual + multimodal |
| **DeepSeek** | V3/R1 (MLA + fine-grained MoE) | MLA attention, reasoning (RL) |
| **Mistral** | 7B, Mixtral (MoE), sliding-window | efficiency-focused |
| **Phi** (Microsoft) | small, data-curated | "textbooks" data quality over scale |

**Closed/frontier**: GPT (OpenAI), Claude (Anthropic), Gemini (Google) -- API-only, larger, multimodal.

**How to read a model**: size + active params (dense vs MoE), context length, attention type (GQA/MLA/sliding), modality support, license, and the quantized GGUF/checkpoint availability ([Quantization](../04-llm-systems/quantization/)) that decides whether you can actually run it.

---

## Architecture Component Catalog

| Component | Old | Modern | Why changed |
|-----------|-----|--------|-------------|
| Normalization | LayerNorm (post) | RMSNorm (pre) | cheaper, more stable deep |
| Positions | learned/sinusoidal | RoPE / ALiBi | extrapolates, relative |
| MLP | ReLU FFN | SwiGLU/GeGLU | better quality/param |
| Attention | MHA | GQA / MLA | smaller KV cache |
| Capacity | dense | MoE | params without FLOPs |
| Long context | full attention | sliding-window + global | linear cost |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Transformers, attention, scaling laws, VAE/diffusion basics | [Deep Learning](../02-deep-learning/) | The architecture foundations |
| KV cache, GQA/MLA, FlashAttention, serving | [LLM Systems & Inference](../04-llm-systems/) | Running these models |
| GGUF/AWQ/FP8 quantization | [Quantization](../04-llm-systems/quantization/) | Making models fit |
| candle/diffusers/transformers runtimes | [Inference Frameworks](../04-llm-systems/frameworks/) | Where they execute |
| KL divergence, score, entropy | [Information Theory](../../information-theory/) | Diffusion + VAE objectives |
| RLHF/DPO alignment | [Deep Learning §12](../02-deep-learning/) | Post-training foundation models |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google/DeepMind | Gemma/Gemini architecture, multimodal, diffusion (Imagen/Veo) | PhD-level |
| OpenAI | GPT architecture, multimodal, scaling | PhD-level |
| Anthropic | Transformer internals, interpretability, post-training | PhD-level |
| Meta | Llama recipe, open-weight strategy | Expert |
| HuggingFace | Transformers + Diffusers, the open ecosystem | Expert |
| Black Forest Labs / Stability | Flux, Stable Diffusion -- flow matching, MMDiT | Expert |
