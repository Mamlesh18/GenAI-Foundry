# 01 · Foundations

The ground floor. By the end of this track you will understand *what a large language model
actually is*, be able to call one from code, and know how to talk to it well enough that the
model does what you intended rather than what you literally typed.

No GPU required. No maths beyond matrix multiplication and a willingness to stare at it.

---

## Modules, in order

| # | Module | You will learn | Time |
|---|---|---|---|
| 01 | [Attention Is All You Need](01-attention-is-all-you-need/) | The 2017 paper that made modern AI possible: self-attention, Q/K/V, multi-head attention, positional encoding | 3–4 h |
| 02 | [LLMs](02-llms/) | What an LLM is, how it generates text, and **your first API call** (Groq / Gemini) | 2–3 h |
| 03 | [Prompting](03-prompting/) | The anatomy of a prompt, sampling parameters, why the same prompt gives different answers | 2 h |
| 04 | [Prompting Techniques](04-prompting-techniques/) | Zero-shot, few-shot, chain-of-thought, self-consistency, ReAct, structured output, and more | 4 h |
| 05 | [Tokenization](05-tokenization/) | How text becomes numbers; BPE; why LLMs can't count letters | 2 h |
| 06 | [Embeddings](06-embeddings/) | Meaning as geometry; similarity search; the basis of RAG | 2 h |
| 07 | [Transformer Architecture](07-transformer-architecture/) | The full block, decoder-only models, and how a modern LLM is assembled | 4 h |

### Why this order?

Module 01 comes first on purpose. Most courses build up to attention over several weeks. We do the
opposite: read the paper early, tolerate the parts that don't land yet, then let modules 05–07
retroactively fill in the details. You learn a mechanism faster when you already know what it's for.

If you prefer strict bottom-up learning, read **05 → 06 → 01 → 07 → 02 → 03 → 04** instead. Both work.

---

## What "done" looks like

You can consider this track complete when you can, without looking anything up:

- [ ] Explain in your own words what a query, key and value are, and why attention is a weighted average.
- [ ] Write a working LLM API call from a blank file in under five minutes.
- [ ] Explain what `temperature`, `top_p` and `max_tokens` do, and pick sensible values for a given task.
- [ ] Name five prompting techniques and say when each is the right tool.
- [ ] Explain why an LLM struggles with "how many r's are in strawberry".
- [ ] Sketch a transformer block on a whiteboard from memory.

---

## Track-level projects

These need most of the track. Full briefs live in [`../projects/`](../projects/).

1. **Mini self-attention from scratch** — implement scaled dot-product attention in NumPy, verify it against PyTorch's `F.scaled_dot_product_attention`.
2. **CLI chatbot with memory** — a terminal chat app with conversation history, streaming output and a token budget.
3. **Prompt technique benchmark** — run one task through six prompting techniques, score the outputs, and write up which won and why.
4. **Tokenizer explorer** — a small tool that shows how different tokenizers split the same text and what that costs you.

---

## Core resources for the whole track

- **Andrej Karpathy — Neural Networks: Zero to Hero** — <https://karpathy.ai/zero-to-hero.html> (build a GPT from scratch; the single best free resource here)
- **Jay Alammar — The Illustrated Transformer** — <https://jalammar.github.io/illustrated-transformer/>
- **Hugging Face LLM Course** — <https://huggingface.co/learn/llm-course>
- **Microsoft — Generative AI for Beginners** — <https://github.com/microsoft/generative-ai-for-beginners>
- **The Prompt Report: A Systematic Survey of Prompting Techniques** — <https://arxiv.org/abs/2406.06608>
