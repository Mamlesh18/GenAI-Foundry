# 02 · Training

How a model goes from random weights to something that answers your questions the way you want.

Foundations taught you to *use* a model. This track teaches you to *change* one — and, just as
importantly, when not to.

---

## The pipeline

Every assistant model you have used went through some version of this:

```
  random weights
        |
   [ PRETRAINING ]          trillions of tokens, next-token prediction
        |                   learns language, facts, reasoning patterns
        v
    BASE MODEL              a brilliant autocomplete; a terrible assistant
        |
   [ FINE-TUNING / SFT ]    thousands of (instruction, good response) pairs
        |                   learns to ANSWER rather than CONTINUE
        v
  INSTRUCT MODEL            helpful, but no taste about which answer is better
        |
   [ PREFERENCE TUNING ]    preference pairs: "response A is better than B"
        |                   learns tone, helpfulness, refusal behaviour
        v
    CHAT MODEL              what you download from Hugging Face
```

**LoRA** and **QLoRA** are not separate stages. They are *how* you run stages 2 and 3 without a
datacentre — the difference between needing 120GB of GPU memory and 6GB.

---

## Modules

| # | Module | You will learn | GPU needed |
|---|---|---|---|
| 01 | [Pretraining](01-pretraining/) | Next-token prediction at scale, data pipelines, scaling laws, why this costs millions | none (toy only) |
| 02 | [Fine-Tuning](02-fine-tuning/) | SFT, loss masking, dataset design, catastrophic forgetting | none for the demo |
| 03 | [LoRA](03-lora/) | Low-rank adaptation — train 0.1% of the parameters, keep most of the quality | none for the demo |
| 04 | [QLoRA](04-qlora/) | 4-bit base model + LoRA — fine-tune a 7B model on a free Colab T4 | free T4 |
| 05 | [Preference Tuning](05-preference-tuning/) | RLHF, DPO, ORPO, KTO — teaching a model which answer is *better* | free T4 |

Each module has:

- **`README.md`** — the concept, the maths where it matters, and honest guidance on when to use it
- **`examples/*.py`** — short, CPU-runnable code that builds the mechanism from scratch, so you can
  see it rather than trust it
- **`examples/*.yaml`** — a real [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) config
  that does the same thing at real scale

---

## Read this before you train anything

Most people who want to fine-tune should not. Work through this honestly:

| Your problem | Train? | Do this instead |
|---|---|---|
| It does not know facts about my company | **No** | [RAG](../06-rag/) |
| It needs current information | **No** | [RAG](../06-rag/) or [tools](../07-agents/) |
| Output format is inconsistent | **No** | [Structured outputs](../skills/03-structured-outputs/) |
| It reasons badly on my task | **Usually no** | Better prompting, then a better model |
| I need a consistent voice or style | **Yes** | SFT on examples of that style |
| I need a narrow task far cheaper | **Yes** | Fine-tune a small model to match a big one |
| My prompt is 3,000 tokens of instructions | **Yes** | Bake them into the weights |
| I need domain-specific output (clinical notes, legal clauses) | **Probably** | SFT on real examples |

**The rule: fine-tuning teaches behaviour, not knowledge.** It changes *how* a model responds, not
*what* it knows. If you want it to know something, that is a retrieval problem.

And before any of it: **measure a prompted baseline.** Roughly half of fine-tuning projects turn out
not to beat a well-written prompt, and you can only discover that if you wrote the number down
first. See [08 · Evaluation](../skills/08-evaluation-and-testing/).

---

## What it costs

Memory needed to train a model, by method. This table is the single most practical thing on this
page — it is why LoRA and QLoRA exist.

| Method | Precision | 7B | 14B | 70B |
|---|---|---|---|---|
| Full fine-tuning | bf16 (32-bit states) | ~120 GB | ~240 GB | ~1200 GB |
| LoRA | 16-bit | ~16 GB | ~32 GB | ~160 GB |
| **QLoRA** | **4-bit** | **~6 GB** | **~12 GB** | **~48 GB** |

*Source: [LLaMA-Factory hardware requirements](https://github.com/hiyouga/LLaMA-Factory). Treat as
order-of-magnitude — actual usage depends on sequence length, batch size and optimizer.*

A free Colab or Kaggle T4 has 16GB. That row marked **QLoRA / 4-bit / ~6 GB** is the reason an
individual can fine-tune a 7B model at all.

### Why full fine-tuning needs so much

For every parameter you train, in mixed-precision AdamW:

| What | Bytes per parameter |
|---|---|
| Weights (bf16) | 2 |
| Gradients (bf16) | 2 |
| Optimizer state — momentum (fp32) | 4 |
| Optimizer state — variance (fp32) | 4 |
| Master weights (fp32) | 4 |
| **Total** | **~16** |

7B × 16 bytes ≈ 112GB, before activations. LoRA trains ~0.1% of the parameters, so that 16 bytes
applies to almost nothing — you are left paying mostly for the frozen base model.

---

## The tool: LLaMA-Factory

Every module here includes a working [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
config. It unifies pretraining, SFT, LoRA/QLoRA, DPO, ORPO, KTO, PPO and reward modelling across
100+ models behind one YAML file and one command — which means you can change *method* without
rewriting your training script.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

llamafactory-cli webui        # browser UI, no YAML needed — start here
```

Then, for any module in this track:

```bash
llamafactory-cli train  <config>.yaml     # train
llamafactory-cli chat   <config>.yaml     # talk to the result
llamafactory-cli export <config>.yaml     # merge the adapter into the base model
```

You can override any YAML field on the command line:

```bash
llamafactory-cli train config.yaml learning_rate=1e-5 num_train_epochs=1
```

> **Learn the mechanism first, then use the tool.** The from-scratch code in each module exists so
> that when a LLaMA-Factory config says `lora_rank: 8`, you know exactly what that changes. A YAML
> file you cannot debug is worse than no tool at all.

---

## Getting a GPU for free

You do not need to buy hardware for modules 03–05.

| Where | GPU | Limit |
|---|---|---|
| [Google Colab](https://colab.research.google.com) | T4 16GB | a few hours per session |
| [Kaggle Notebooks](https://www.kaggle.com/code) | T4 ×2 / P100 | ~30 GPU hours per week |
| [Lightning AI](https://lightning.ai) | varies | free monthly credits |

[Unsloth](https://github.com/unslothai/unsloth) publishes free Colab notebooks that fine-tune real
models in under an hour, and runs roughly 2× faster than a stock setup with less memory. It is the
fastest path from zero to a trained adapter.

---

## Resources for the whole track

- [Hugging Face LLM Course — fine-tuning chapters](https://huggingface.co/learn/llm-course)
- [LLaMA-Factory docs](https://llamafactory.readthedocs.io/) · [paper (ACL 2024)](https://arxiv.org/abs/2403.13372)
- [PEFT documentation](https://huggingface.co/docs/peft) — the library behind LoRA everywhere
- [TRL documentation](https://huggingface.co/docs/trl) — SFT, DPO, PPO, reward modelling
- [Unsloth](https://github.com/unslothai/unsloth) — fastest practical fine-tuning, free notebooks
- [Sebastian Raschka — Finetuning LLMs](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — the clearest written overview of the whole landscape

---

**Previous track:** [01 · Foundations](../01-foundations/) ·
**Next track:** [03 · Inference](../03-inference/)
