# 09 · Shipping to Production

**Level:** Advanced · **Time:** ~6 hours

---

## 1. What you will be able to do

Take a working LLM prototype and make it survive real users: bounded cost, acceptable latency,
graceful failure, and resistance to abuse.

---

## 2. Prerequisites

- Skills: [08 · Evaluation & Testing](../08-evaluation-and-testing/) and something worth shipping
- Concepts: [03 · Inference](../../03-inference/), [04 · Serving](../../04-serving/)

---

## 3. The concept, briefly

A prototype has one user who is forgiving, one request at a time, and no budget. Production has
none of those. Four things change:

**Cost is unbounded by default.** A per-request cost that seems trivial becomes a real bill at
volume — and a bug or an abusive user can multiply it overnight. You need caps at the request,
user and system level.

**Latency is a product decision.** Several seconds is normal for an LLM. Streaming changes the
*perceived* latency far more than any optimisation of the actual latency.

**Everything fails.** Rate limits, timeouts, provider outages, content filters, truncation. Each
needs a defined behaviour that is not "500 error".

**Users are not all friendly.** Prompt injection, attempts to extract your system prompt, and
people using your expensive endpoint as a free general-purpose chatbot.

---

## 4. Build it

1. **Instrument first.** Log every request: input size, output size, latency, cost, model, and
   outcome. You cannot optimise what you are not measuring, and this takes an hour.

2. **Set hard limits.** Max input length, `max_tokens` on every call, per-user rate limits, and a
   daily spend cap that actually stops serving. Test that the cap triggers.

3. **Add retries with backoff.** Exponential backoff with jitter on rate limits and 5xx. Do *not*
   retry on 4xx — you will just spend money repeating the same invalid request.

4. **Add timeouts and a fallback.** Every call gets a timeout. On failure, fall back to a smaller
   model, a cached answer, or an honest error message. Never a stack trace.

5. **Add caching.** Exact-match caching on identical requests is free money. Semantic caching
   (embedding similarity above a threshold) catches more, at the risk of returning a subtly wrong
   cached answer — measure that risk before enabling it.

6. **Use prompt caching.** Most providers cache long, stable prompt prefixes at reduced cost. Put
   the fixed part of your prompt first and the variable part last. Often a large saving for a
   trivial reordering.

7. **Route by difficulty.** Send easy requests to a small fast model and hard ones to a large one.
   A classifier or a simple heuristic often cuts cost substantially with no quality loss — verify
   with your evaluation set.

8. **Stream everything user-facing.** Time-to-first-token is the number users feel.

9. **Add input and output guards.** Length and content checks on input; on output, check for leaked
   system prompt, PII and injected instructions before display.

10. **Handle the abuse cases.** Rate limit by user, not just by IP. Detect and reject off-topic use.
    Assume your system prompt will leak and design so that leaking it is not a catastrophe.

11. **Set up alerts.** Error rate, p95 latency, cost per hour, cache hit rate. Alert on the
    derivative, not just the level — a sudden 3× cost rise matters even below your cap.

---

## 5. Checkpoint

Under a load test of 100 concurrent users:

- [ ] p95 latency within your stated target
- [ ] No unhandled exceptions
- [ ] Rate limits trigger correctly and recover
- [ ] The daily spend cap actually stops serving when hit
- [ ] Cache hit rate measured, and the cost saving quantified
- [ ] Simulated provider outage falls back gracefully
- [ ] A prompt-injection attempt does not leak the system prompt
- [ ] Every request is traceable in logs by id
- [ ] You can state cost per request and per user, and project the monthly bill

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| No spend cap | A bug or abuser produces a shocking invoice | Hard cap that stops serving |
| Retrying 4xx errors | Money spent repeating an invalid request | Retry 429 and 5xx only |
| No timeout | Requests hang, connections exhaust | Timeout on every call |
| No caching | Paying repeatedly for identical requests | Exact-match cache first |
| Variable content early in the prompt | Prompt caching never hits | Stable prefix, variable suffix |
| One model for everything | Overpaying for easy requests | Route by difficulty |
| No streaming | Feels broken to users | Stream user-facing output |
| System prompt holds secrets | It leaks; assume it will | Keep secrets out of prompts entirely |
| Alerting only on absolute thresholds | A 3× cost rise below the cap goes unnoticed | Alert on rate of change |

---

## 7. Going further

- **Related skill:** [10 · Fine-Tuning](../10-fine-tuning/) — sometimes the cheapest path to lower cost per request
- Self-host with [vLLM](https://docs.vllm.ai/) when volume makes API pricing unattractive — then
  compare total cost honestly, including your own time
- Add [quantization](../../03-inference/03-quantization/) if you are self-hosting
- Build a canary deployment so model upgrades roll out to 5% of traffic first
- Add a feedback button and route the negatives into your golden dataset

**Links:** [vLLM](https://docs.vllm.ai/) ·
[Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) ·
[OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
