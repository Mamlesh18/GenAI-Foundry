# 02 · Large Language Models

Module 01 gave you the mechanism. This module gives you the machine — and gets you to your
**first working API call in about ten minutes**, free, with no GPU and no credit card.

---

## 1. What an LLM actually is

Strip away the chat interface and a large language model does exactly one thing:

> **Given a sequence of tokens, predict a probability distribution over the next token.**

That is the entire job. Everything else — reasoning, translation, code, refusals, personality — is
an emergent consequence of doing that one job extremely well at enormous scale.

To produce a sentence, the model runs that prediction in a loop:

```
prompt:  "The capital of France is"
step 1:  P(next) = { " Paris": 0.89, " a": 0.03, " located": 0.02, ... }  -> pick " Paris"
step 2:  "The capital of France is Paris"  -> pick "."
step 3:  "The capital of France is Paris." -> pick <end-of-sequence>  -> stop
```

This is **autoregressive generation**: each output token is appended to the input, and the whole
thing runs again. Two consequences worth internalising early:

- **The model has no memory between calls.** A "conversation" is an illusion created by resending
  the entire history on every request. Statelessness is the default; memory is something *you*
  build.
- **Cost and latency scale with the text.** You pay for input tokens and output tokens. Output
  tokens are generated one at a time, so a 1000-token answer takes roughly 10× as long as a
  100-token one.

### The three ingredients

| Ingredient | What it means | Rough scale today |
|---|---|---|
| **Parameters** | The learned weights — the model's "knowledge" | 1B (phone) to 1T+ (frontier) |
| **Training data** | Text it learned from | trillions of tokens |
| **Context window** | How much it can read at once | 8K to 1M+ tokens |

### How a raw model becomes a useful assistant

A model fresh out of pretraining is a very good autocomplete and a terrible assistant — ask it a
question and it may cheerfully generate twenty more questions. Three stages fix that:

1. **Pretraining** — predict the next token across the internet. Learns language, facts, reasoning
   patterns. Costs millions of dollars.
2. **Instruction tuning (SFT)** — fine-tune on curated (instruction, good response) pairs. Teaches
   it to *answer* rather than *continue*.
3. **Preference tuning (RLHF / DPO)** — train against human preferences about which of two
   responses is better. Teaches tone, helpfulness, and refusal behaviour.

Covered in depth in [`../../02-training`](../../02-training/).

### What LLMs are bad at, and why

Understanding the failure modes is more useful than memorising the capabilities:

| Failure | Why it happens |
|---|---|
| **Hallucination** | It is optimised to produce *plausible* text, not *true* text. It has no notion of "I don't know" unless trained to express one. |
| **Arithmetic** | Numbers are chopped into arbitrary tokens; there is no calculator inside. Give it a tool instead. |
| **Counting letters** | It sees tokens, not characters. "strawberry" may be 2–3 tokens, so the r's are not individually visible. See [05 · Tokenization](../05-tokenization/). |
| **Knowledge cutoff** | It knows nothing after its training data ended. Fix with retrieval ([06 · RAG](../../06-rag/)). |
| **Non-determinism** | Sampling is random by default. Same prompt, different answer. See [03 · Prompting](../03-prompting/). |
| **Lost in the middle** | With a long context, information buried in the middle is attended to less reliably than the start or end. |

---

## 2. Your first API call

You have two excellent free options. **Do both** — seeing the same idea through two different SDKs
teaches you which parts are the concept and which are just one vendor's spelling.

| | **Groq** | **Google Gemini** |
|---|---|---|
| Get a key | <https://console.groq.com/keys> | <https://aistudio.google.com/apikey> |
| Play in browser first | <https://console.groq.com/playground> | <https://aistudio.google.com> |
| Free tier | Yes, no card required | Yes, no card required |
| Best for | Extreme speed; OpenAI-compatible API | Huge context window; multimodal |
| Models run | Open-weight (Llama, GPT-OSS, Qwen) | Google's proprietary Gemini |

> **Start in the browser playground before you write any code.** Type a few prompts, drag the
> temperature slider, watch what changes. Ten minutes there will make the code obvious.

### Step 1 — Get a key

Sign in at one of the links above, create an API key, and copy it. Groq keys start with `gsk_`.

### Step 2 — Store it safely

Never paste a key into a source file. Use an environment variable:

```bash
# macOS / Linux
export GROQ_API_KEY="gsk_your_key_here"
export GEMINI_API_KEY="your_key_here"
```

```powershell
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
$env:GEMINI_API_KEY="your_key_here"
```

Or create a `.env` file next to the examples (it is already git-ignored):

```
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=your_key_here
```

> **If you ever commit a key, it is compromised.** Revoke it in the console immediately and issue a
> new one. Bots scrape public repositories for keys within minutes.

### Step 3 — Install and run

```bash
cd examples
pip install -r requirements.txt
python 01_first_call_groq.py
```

### The whole thing, in nine lines (Groq)

```python
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Explain attention in one sentence."}],
)
print(response.choices[0].message.content)
```

### The same thing with Gemini

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain attention in one sentence.",
)
print(response.text)
```

> **Model IDs move fast.** The current lists live at
> [Groq models](https://console.groq.com/docs/models) and
> [Gemini models](https://ai.google.dev/gemini-api/docs/models). If you get a "model not found"
> error, that is the first place to look — it is the single most common reason a copied snippet
> stops working. Google is also migrating toward a newer `client.interactions.create(...)` surface;
> check [the quickstart](https://ai.google.dev/gemini-api/docs/quickstart) for the current shape.

### Anatomy of a request

Every chat-style LLM API is a variation on the same handful of fields:

| Field | What it does |
|---|---|
| `model` | Which model to use. Bigger is smarter, smaller is faster and cheaper. |
| `messages` | The conversation so far, as a list of `{role, content}` objects. |
| `temperature` | Randomness, roughly 0–2. See [03 · Prompting](../03-prompting/). |
| `max_tokens` | Hard cap on output length. Prevents runaway cost. |
| `stream` | If true, tokens arrive as they are generated instead of all at once. |

And the three roles you will use constantly:

- **`system`** — standing instructions: persona, rules, output format. Set once, applies throughout.
- **`user`** — what the human said.
- **`assistant`** — what the model previously said. You send these back to give it memory.

---

## 3. Examples

All scripts are in [`examples/`](examples/) and run standalone.

| File | What it demonstrates |
|---|---|
| [`01_first_call_groq.py`](examples/01_first_call_groq.py) | The minimal call, plus reading token usage from the response |
| [`02_first_call_gemini.py`](examples/02_first_call_gemini.py) | The same task through a different SDK |
| [`03_streaming_and_params.py`](examples/03_streaming_and_params.py) | Streaming output, and temperature compared side by side |
| [`04_chat_with_memory.py`](examples/04_chat_with_memory.py) | A working terminal chatbot — where "memory" actually comes from |

---

## 4. Exercises

1. **Make it fail.** Send a request with a deliberately wrong model name. Read the error carefully.
   Now send one with an invalid API key. Being able to recognise these two errors instantly will
   save you hours later.
2. **Count the cost.** Print `response.usage` for three prompts of very different lengths. Look up
   the model's per-million-token price and compute what 10,000 such calls would cost.
3. **Delete the memory.** In `04_chat_with_memory.py`, stop appending assistant replies to the
   history. Ask a follow-up question that depends on the previous turn. Explain exactly what you
   observe.
4. **Feel the temperature.** Run the same creative prompt five times at `temperature=0` and five
   times at `temperature=1.5`. Then run a factual prompt at both. Which setting suits which task?
5. **Hit a limit.** Set `max_tokens=20` and ask for a long explanation. Inspect
   `response.choices[0].finish_reason`. Why does this field matter in production code?
6. **Swap the provider.** Take `01_first_call_groq.py` and make it run against Gemini by changing as
   few lines as possible. What was genuinely different, and what was only spelled differently?

---

## 5. Projects to build and test

### Beginner — Terminal chatbot with a personality
Extend `04_chat_with_memory.py`: add a system prompt that gives it a persona, a `/reset` command
that clears history, a `/save` command that writes the transcript to JSON, and a running token
counter shown after each turn.

**How to test it:** confirm it remembers your name from turn 1 at turn 5; confirm `/reset` makes it
forget; confirm the saved JSON reloads into a working conversation.

### Beginner — Text utility CLI
A single command that takes a file plus a mode (`summarise`, `translate`, `extract-keywords`,
`fix-grammar`) and prints the result.

**How to test it:** run every mode against the same 500-word article. Then run it against an empty
file, a 100,000-word file, and a file of pure emoji. It should degrade gracefully, not crash.

### Intermediate — Provider abstraction layer
Write one `ask(prompt, model=...)` function that works across Groq and Gemini behind a common
interface, with retry-and-backoff on rate limits and a cost estimate per call.

**How to test it:** run the same 20 prompts through both providers, log latency and cost, and
produce a small comparison table. Force a rate limit by firing 50 rapid requests and confirm your
backoff recovers instead of crashing.

### Intermediate — Token budget guard
A wrapper that refuses to send a request whose estimated tokens exceed a configured budget, and
truncates conversation history intelligently (keeping the system prompt and the most recent turns)
when the context window fills up.

**How to test it:** feed it a conversation long enough to overflow the context window and confirm
it trims rather than erroring, and that the system prompt always survives.

---

## 6. Resources

**Get started / play**
- [Groq Console](https://console.groq.com) · [Quickstart](https://console.groq.com/docs/quickstart) · [Models](https://console.groq.com/docs/models)
- [Google AI Studio](https://aistudio.google.com) · [Gemini quickstart](https://ai.google.dev/gemini-api/docs/quickstart) · [Models](https://ai.google.dev/gemini-api/docs/models)
- [Anthropic Claude API](https://docs.claude.com) — excellent docs; paid, but the reference material is worth reading regardless
- [OpenRouter](https://openrouter.ai) — one API key, hundreds of models, useful for comparing

**Understand**
- [What Is ChatGPT Doing and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) — Stephen Wolfram. Long, and the best single explanation of next-token prediction.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Karpathy, 1 hour, no maths. Watch this if you watch nothing else.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) — free, hands-on, well maintained.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — the GPT-3 paper; where in-context learning was first shown at scale.
- [Training language models to follow instructions](https://arxiv.org/abs/2203.02155) — the InstructGPT paper; how raw models became assistants.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — why position in a long context changes what the model actually uses.

---

**Previous:** [01 · Attention Is All You Need](../01-attention-is-all-you-need/) ·
**Next:** [03 · Prompting](../03-prompting/)
