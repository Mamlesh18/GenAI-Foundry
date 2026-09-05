# 07 · Transformer Architecture

Module 01 gave you attention — the mechanism. This module assembles it into the machine, and
updates the 2017 design to what a model shipping in 2026 actually looks like.

---

## 1. The full picture

A modern LLM is a **decoder-only transformer**. End to end:

```
  token ids                    [4, 1893, 374, ...]
      |
  [ embedding lookup ]         ids -> vectors, one row per token
      |
  [ + positional info ]        RoPE, applied inside attention in modern models
      |
  +------------------------+
  |  x N layers            |   N = 12 (small) to 100+ (frontier)
  |                        |
  |   [ LayerNorm ]        |   pre-LN: normalise BEFORE the sublayer
  |   [ Masked MHA ]       |   tokens mix information here
  |   [ + residual ]       |
  |                        |
  |   [ LayerNorm ]        |
  |   [ Feed-forward ]     |   each token processed independently
  |   [ + residual ]       |
  +------------------------+
      |
  [ final LayerNorm ]
      |
  [ output projection ]        hidden -> vocabulary size
      |
  [ softmax ]                  probability distribution over the next token
```

The stack is **the same block repeated**. There is no clever hierarchy — depth and width are the
whole story. That uniformity is exactly why it scales so well.

---

## 2. The two halves of a block

Every layer does two distinct jobs, and keeping them straight clarifies everything:

| Sublayer | Direction | What it does |
|---|---|---|
| **Attention** | *across* tokens | Gathers relevant context from other positions |
| **Feed-forward** | *within* a token | Processes that gathered information |

Attention is the communication step; the FFN is the computation step. Roughly two-thirds of the
parameters live in the FFN — the intuition that "the knowledge is in the feed-forward layers, the
routing is in attention" is crude but useful.

---

## 3. What changed since 2017

The paper's architecture works. Almost none of its specific choices survived contact with scale.

| Component | 2017 | 2026 | Why |
|---|---|---|---|
| **Structure** | Encoder–decoder | **Decoder-only** | Simpler, and next-token prediction scales better |
| **Position** | Sinusoidal, added at input | **RoPE**, applied in attention | Far better length generalisation |
| **Norm placement** | Post-LN | **Pre-LN** | Post-LN needs warmup and is unstable when deep |
| **Norm type** | LayerNorm | **RMSNorm** | Same effect, cheaper — drops the mean-centering |
| **Activation** | ReLU | **SwiGLU** | Consistently better loss for the same parameter count |
| **Attention** | Multi-head (MHA) | **GQA / MQA** | Shrinks the KV cache, which dominates inference memory |
| **Bias terms** | Yes | Mostly **removed** | No measurable loss, fewer parameters |
| **Attention kernel** | Naive | **FlashAttention** | Same maths, IO-aware; makes long context affordable |
| **Sparsity** | Dense | Sometimes **MoE** | Route each token to a few experts: more capacity, same compute per token |

Two of these matter enough to spell out:

**Grouped-Query Attention (GQA).** During generation, every past token's keys and values are cached
(the [KV cache](../../03-inference/01-kv-cache/)). With 64 query heads and 64 KV heads, that cache
is enormous. GQA shares one set of K/V across a group of query heads — 64 query heads, 8 KV heads —
cutting cache memory ~8× with almost no quality loss. This is why long context became practical.

**Mixture of Experts (MoE).** Replace the FFN with many parallel "expert" FFNs and a router that
sends each token to only 2 of them. A model can then hold 400B parameters but activate only 30B per
token: the capacity of a huge model at the inference cost of a medium one.

---

## 4. Scale, and what it costs

| Model size | Layers | d_model | Heads | Runs on |
|---|---|---|---|---|
| ~125M | 12 | 768 | 12 | a laptop CPU |
| ~7B | 32 | 4096 | 32 | one consumer GPU (quantized) |
| ~70B | 80 | 8192 | 64 | multiple datacentre GPUs |
| 400B+ | 100+ | 16384+ | 128 | a cluster |

Two numbers worth carrying around:

- **Inference memory ≈ parameters × bytes-per-parameter.** A 7B model at fp16 needs ~14GB; at
  4-bit, ~4GB. This is why [quantization](../../03-inference/03-quantization/) matters.
- **Attention is O(n²) in sequence length.** Doubling the context quadruples attention compute.
  FlashAttention fixes the *memory* cost, not the asymptotic compute cost.

**Scaling laws** ([Kaplan et al.](https://arxiv.org/abs/2001.08361),
[Chinchilla](https://arxiv.org/abs/2203.15556)) showed loss falls predictably with parameters, data
and compute — and that most early large models were badly undertrained. Chinchilla's finding, that
you want roughly 20 tokens per parameter, redirected the whole field toward smaller models trained
on far more data.

---

## 5. Exercises

1. **Read a real config.** Open `config.json` for Llama 3 8B on Hugging Face. Map every field to
   something in this README. Which entries are you unable to explain?
2. **Compute the parameter count.** For `d_model=4096`, `n_layers=32`, `d_ff=14336`,
   `vocab=128256`: calculate parameters in embeddings, attention and FFN separately. Does your
   total land near 8B?
3. **Size the KV cache.** For that model at 8192 context, batch size 1, fp16: how much memory does
   the KV cache need with MHA (32 KV heads) versus GQA (8 KV heads)?
4. **Pre-LN vs post-LN.** Train a small transformer both ways without warmup. One will diverge.
   Explain why in terms of gradient flow through the residual path.
5. **Ablate.** Take a small working transformer and remove, one at a time: residual connections,
   layer norm, positional encoding. Train each for a fixed budget and record the loss. Rank them by
   how badly they were missed.

---

## 6. Projects to build and test

### Intermediate — Modern GPT from scratch
Build a decoder-only model with RMSNorm, RoPE, SwiGLU and GQA. Train it on TinyStories or a
Gutenberg text.

**How to test it:** compare validation loss against a vanilla 2017-style model of identical
parameter count and training budget. The modern one should win. Report by how much, per change, by
ablating one modernisation at a time.

### Intermediate — Architecture visualiser
Load a pretrained model and generate a diagram of its actual architecture from its config: layer
count, dimensions, head configuration, parameter distribution per component.

**How to test it:** run it against Llama, Mistral and Qwen. Your parameter total must match the
number on the model card.

### Advanced — Scaling law replication
Train 5 models of increasing size on the same data with a fixed compute budget each. Plot loss
against parameters on log-log axes.

**How to test it:** you should recover a straight line. Compare its slope to the published
Chinchilla exponent. Where your line deviates is where your training setup is suboptimal — and
finding out why is the actual lesson.

### Advanced — Implement FlashAttention
Write a tiled, IO-aware attention kernel (Triton, or careful PyTorch).

**How to test it:** assert numerical equivalence with naive attention to `1e-4`, then benchmark
peak memory and throughput at sequence lengths 512 → 8192. You should see memory go from quadratic
to linear.

---

## 7. Resources

**Build it**
- [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy. Still the best two hours available.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — clean, readable, actually trains.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the paper as executable code.
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — a live leaderboard of architectural and optimiser improvements over nanoGPT.

**Understand it**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar.
- [Transformer Circuits](https://transformer-circuits.pub/) — Anthropic's interpretability work; what the layers actually learn.
- [The Transformer Family v2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — Lilian Weng's survey of variants.

**The modernisations**
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) · [GQA](https://arxiv.org/abs/2305.13245) · [RMSNorm](https://arxiv.org/abs/1910.07467) · [SwiGLU](https://arxiv.org/abs/2002.05202) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [Switch Transformer (MoE)](https://arxiv.org/abs/2101.03961)

**Scaling**
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · [Chinchilla](https://arxiv.org/abs/2203.15556)

---

**Previous:** [06 · Embeddings](../06-embeddings/) ·
**Next track:** [02 · Training](../../02-training/)
