# 01 · First LLM App

**Level:** Beginner · **Time:** ~2 hours

---

## 1. What you will be able to do

Write, from an empty file, a working application that calls an LLM API — with streaming output,
conversation memory, error handling and a token budget — without copying from a tutorial.

---

## 2. Prerequisites

- Python 3.10+ and basic comfort with the language
- A free API key: [Groq](https://console.groq.com/keys) or [Gemini](https://aistudio.google.com/apikey)
- Concepts: [02 · LLMs](../../01-foundations/02-llms/)

---

## 3. The concept, briefly

An LLM API is a **stateless function**: text in, text out. It remembers nothing between calls.

Everything that feels like a conversation is you resending the entire history every turn. Once that
lands, the design questions become obvious: what do you keep in history, what do you drop when it
gets too long, and who pays for resending it all every time (you do).

Three practical properties shape every app you will build:

- **It is slow.** Hundreds of milliseconds to many seconds. Stream, or your UI feels broken.
- **It is metered.** You pay per token, in *and* out. Every resent history costs again.
- **It fails.** Rate limits, timeouts, content filters, truncation. Handle them or your app dies in
  front of a user.

---

## 4. Build it

1. **Make the simplest possible call.** One file, one request, print the response. Confirm your key
   works before adding anything else. See
   [`../../01-foundations/02-llms/examples/01_first_call_groq.py`](../../01-foundations/02-llms/examples/01_first_call_groq.py).

2. **Read the metadata.** Print `usage.prompt_tokens`, `usage.completion_tokens` and
   `finish_reason` on every call. Build this habit now — it is how you debug cost and truncation
   later.

3. **Add streaming.** Set `stream=True` and print chunks as they arrive. Measure time-to-first-token
   versus total time. The gap is why streaming exists.

4. **Add a system prompt.** Give it a persona and a length constraint. Verify the constraint
   actually holds across ten different questions.

5. **Add conversation memory.** Keep a `messages` list; append both the user turn and the assistant
   reply. Prove it works by telling it your name and asking three turns later.

6. **Add commands.** `/reset`, `/history`, `/tokens`, `/quit`. `/history` should print the raw
   message list — seeing exactly what gets resent every turn is the single most clarifying moment
   in this skill.

7. **Handle failure.** Wrap the call in try/except. Catch rate limits specifically and retry with
   exponential backoff. Catch everything else and print something useful instead of a stack trace.

8. **Add a token budget.** Track cumulative tokens. When history approaches the context window,
   drop the oldest turns — but never the system prompt.

---

## 5. Checkpoint

Your app must:

- [ ] Answer a question, streaming the response as it generates
- [ ] Remember a fact from turn 1 when asked at turn 6
- [ ] Forget it after `/reset`
- [ ] Report accurate cumulative token usage
- [ ] Survive an invalid API key with a clear message, not a traceback
- [ ] Survive a rate limit by retrying, not crashing
- [ ] Trim old history when the conversation gets long, keeping the system prompt
- [ ] Run end to end without you looking at any tutorial

Ten minutes of continuous conversation should cost you nothing but your free tier.

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| API key pasted into the source | Committed to git, scraped by bots within minutes | Environment variable or `.env` |
| Not appending the assistant reply to history | The model contradicts itself and forgets its own statements | Append both roles every turn |
| No `max_tokens` | One runaway response burns your quota | Always set it |
| Ignoring `finish_reason` | Truncated answers shown to users as if complete | Check for `'length'` |
| Rebuilding the client each call | Wasted connection setup, slower | Build once, reuse |
| Unbounded history | Requests grow until they exceed the context window and hard-fail | Trim, keeping the system prompt |

---

## 7. Going further

- **Next skill:** [02 · Prompt Engineering](../02-prompt-engineering/)
- Add a `/save` and `/load` so conversations survive a restart
- Add a provider switch so the same app runs on Groq or Gemini
- Put a web UI on it with Streamlit or Gradio — about 20 lines
- Projects: see [`../../projects/`](../../projects/)

**Links:** [Groq quickstart](https://console.groq.com/docs/quickstart) ·
[Gemini quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
