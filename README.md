# GenAI-Foundry

An open, ordered learning repository for **Generative AI** — built for students starting from
zero and for professionals who want to fill gaps in a specific layer of the stack.

Everything here is designed to be *read, run, and broken on purpose*. Each folder has a README
that explains the idea in plain language, links to primary sources, and ends with exercises and
projects you can actually build.

---

## How this repository is organised

There are three parallel tracks. Use them together.

| Track | Folder | What it gives you |
|---|---|---|
| **Concepts** | `01-foundations` … `08-multimodal` | The *why*. Ordered theory, from attention to agents. |
| **Skills** | `skills/` | The *how*. Self-contained, hands-on capabilities you can practise in an afternoon. |
| **Projects** | `projects/` | The *proof*. Buildable things that combine several skills. |
| **Resources** | `resources/` | Curated papers, courses, blogs, tools, datasets. |

A concept folder tells you what a KV cache is. A skill folder makes you implement one. A project
folder makes you ship something that needs one.

---

## Learning path (concepts)

1. **[`01-foundations`](01-foundations/)** — Attention, LLMs, prompting, tokenization, embeddings, transformers. **Start here.**
2. **[`02-training`](02-training/)** — Pretraining, fine-tuning, LoRA, QLoRA, preference tuning.
3. **[`03-inference`](03-inference/)** — KV cache, batching, quantization, paged attention, speculative decoding.
4. **[`04-serving`](04-serving/)** — vLLM, SGLang, distributed inference.
5. **[`05-evaluation`](05-evaluation/)** — Benchmarks, hallucination testing, evaluation workflows.
6. **[`06-rag`](06-rag/)** — Retrieval-augmented generation for grounded answers.
7. **[`07-agents`](07-agents/)** — Tool use and multi-step agent systems.
8. **[`08-multimodal`](08-multimodal/)** — Text, image, audio and video together.

---

## Where do I start?

**Complete beginner.** Go to [`01-foundations`](01-foundations/) and work through modules 01 → 04
in order. You will have made your first LLM API call within about 20 minutes of module 02, using a
free API key — no GPU, no credit card.

**Developer who already ships features with an LLM API.** Skip to
[`01-foundations/04-prompting-techniques`](01-foundations/04-prompting-techniques/), then jump to
[`06-rag`](06-rag/) and [`07-agents`](07-agents/).

**ML engineer moving into GenAI infrastructure.** Start at [`02-training`](02-training/) and
[`03-inference`](03-inference/); use foundations as reference only.

---

## Setup

You only need Python 3.10+ and a free API key to do the first several modules.

```bash
git clone <this-repo>
cd GenAI-Foundry

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r 01-foundations/02-llms/examples/requirements.txt
```

Then get a free key from one of these and export it:

- **Groq** — fastest to get started, generous free tier: <https://console.groq.com/keys>
- **Google Gemini** — free tier via AI Studio: <https://aistudio.google.com/apikey>

```bash
# macOS/Linux
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="..."

# Windows PowerShell
$env:GROQ_API_KEY="gsk_..."
$env:GEMINI_API_KEY="..."
```

> **Never commit an API key.** Keys belong in environment variables or a `.env` file that is
> already covered by `.gitignore`.

---

## Conventions used in every folder

Every module README follows the same shape, so you always know where to look:

1. **What this is** — the idea in a paragraph, no jargon.
2. **Why it matters** — what breaks without it.
3. **Core concepts** — the mental model.
4. **Examples** — runnable code in an `examples/` folder.
5. **Exercises** — small, checkable tasks.
6. **Projects** — bigger things to build and test.
7. **Resources** — papers, docs and articles worth your time.

---

## Contributing

Adding material is welcome. Keep the shape above, prefer primary sources over blog summaries,
and make sure every code sample actually runs before it lands.

## License

See [LICENSE](LICENSE).
