# Resources

Curated, not comprehensive. Everything listed here is something worth spending your time on.

Last reviewed: **September 2026**. Model IDs, pricing and free-tier limits change constantly —
always confirm against the provider's own docs before relying on a number.

---

## Start here

If you read or watch only three things:

1. **[Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)** — Karpathy, 1 hour, no maths. The best overview that exists.
2. **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** — Jay Alammar. The diagrams that make attention click.
3. **[Prompt Engineering Guide](https://www.promptingguide.ai/)** — the reference you will return to weekly.

---

## Free API access

Get a key, no credit card required. Enough to complete every foundations module.

| Provider | Key | Playground | Notes |
|---|---|---|---|
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | [playground](https://console.groq.com/playground) | Extremely fast; OpenAI-compatible; open-weight models |
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | [AI Studio](https://aistudio.google.com) | Huge context window; multimodal; generous free tier |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai) | — | One key, many models; some free |
| **Hugging Face** | [huggingface.co](https://huggingface.co/settings/tokens) | [Spaces](https://huggingface.co/spaces) | Inference API plus every open model |

**Free compute:** [Google Colab](https://colab.research.google.com) (T4 GPU) ·
[Kaggle Notebooks](https://www.kaggle.com/code) (30 GPU hours/week) ·
[Lightning AI](https://lightning.ai) (free monthly credits)

---

## Courses

| Course | By | Why |
|---|---|---|
| [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Karpathy | Build a GPT from scratch. The best free resource in the field, by a distance. |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) | Hugging Face | Practical, current, free. Assumes Python fluency. |
| [Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners) | Microsoft | 21 lessons, MIT-licensed, actively maintained. |
| [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | DeepLearning.AI | 90 minutes, free, the best prompting primer. |
| [Practical Deep Learning for Coders](https://course.fast.ai/) | fast.ai | Top-down; you train a real model in lesson one. |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | Simon Prince | Free textbook. The maths, done properly. |

---

## Papers, in reading order

**The foundation**
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the Transformer
2. [BERT](https://arxiv.org/abs/1810.04805) (2018) — encoder-only, bidirectional
3. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020) — in-context learning at scale
4. [Scaling Laws](https://arxiv.org/abs/2001.08361) (2020) · [Chinchilla](https://arxiv.org/abs/2203.15556) (2022) — how big, trained on how much
5. [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) — how raw models became assistants

**Prompting & reasoning**
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903) · [Zero-shot CoT](https://arxiv.org/abs/2205.11916) · [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [ReAct](https://arxiv.org/abs/2210.03629) · [Tree of Thoughts](https://arxiv.org/abs/2305.10601) · [Reflexion](https://arxiv.org/abs/2303.11366)
- [The Prompt Report](https://arxiv.org/abs/2406.06608) — 58 techniques, surveyed

**Retrieval**
- [RAG](https://arxiv.org/abs/2005.11401) · [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [Lost in the Middle](https://arxiv.org/abs/2307.03172)

**Efficiency**
- [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) · [GQA](https://arxiv.org/abs/2305.13245)

---

## Blogs worth following

- [Lilian Weng](https://lilianweng.github.io/) — deep, careful surveys. Start with the agents and transformer-family posts.
- [Sebastian Raschka](https://sebastianraschka.com/blog/) — implementation-focused, always with code.
- [Jay Alammar](https://jalammar.github.io/) — visual explanations, unmatched.
- [Simon Willison](https://simonwillison.net/tags/llms/) — the best running commentary on what is actually happening.
- [Chip Huyen](https://huyenchip.com/blog/) — production ML systems.
- [Transformer Circuits](https://transformer-circuits.pub/) — interpretability; what the layers really learn.
- [Anthropic Engineering](https://www.anthropic.com/engineering) — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) is required reading.

---

## Tools

| Purpose | Tools |
|---|---|
| **Frameworks** | [LangChain](https://python.langchain.com/) · [LlamaIndex](https://docs.llamaindex.ai/) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [DSPy](https://dspy.ai/) |
| **Structured output** | [Pydantic](https://docs.pydantic.dev/) · [Instructor](https://python.useinstructor.com/) · [Outlines](https://github.com/dottxt-ai/outlines) |
| **Vector search** | [Chroma](https://www.trychroma.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [Qdrant](https://qdrant.tech/) · [pgvector](https://github.com/pgvector/pgvector) |
| **Embeddings** | [Sentence-Transformers](https://sbert.net/) · [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| **Evaluation** | [Ragas](https://docs.ragas.io/) · [DeepEval](https://github.com/confident-ai/deepeval) · [Promptfoo](https://promptfoo.dev/) |
| **Observability** | [LangSmith](https://smith.langchain.com/) · [Langfuse](https://langfuse.com/) · [Phoenix](https://github.com/Arize-ai/phoenix) |
| **Serving** | [vLLM](https://docs.vllm.ai/) · [SGLang](https://docs.sglang.ai/) · [Ollama](https://ollama.com/) · [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| **Fine-tuning** | [Unsloth](https://github.com/unslothai/unsloth) · [PEFT](https://huggingface.co/docs/peft) · [TRL](https://huggingface.co/docs/trl) · [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) |
| **Understanding** | [Tiktokenizer](https://tiktokenizer.vercel.app) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [minbpe](https://github.com/karpathy/minbpe) |

---

## Datasets & benchmarks

- [Hugging Face Datasets](https://huggingface.co/datasets) — the hub for everything
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — grade-school maths; the standard test for reasoning techniques
- [MMLU](https://huggingface.co/datasets/cais/mmlu) — broad knowledge across 57 subjects
- [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) — code generation
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — small enough to train a model on yourself
- [Project Gutenberg](https://www.gutenberg.org/) — public-domain text for character-level training
- [LMSYS Chatbot Arena](https://lmarena.ai/) — human preference rankings across models
- [HELM](https://crfm.stanford.edu/helm/) — Stanford's holistic evaluation

---

## Safety & security

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — start here if you are shipping anything
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Anthropic on prompt injection](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks)

---

## Contributing

Add a link only if you have actually read or used it, and say in one line why it earns a place.
Prefer primary sources over summaries. Prune anything that has gone stale.
