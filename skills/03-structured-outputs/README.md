# 03 · Structured Outputs

**Level:** Beginner · **Time:** ~3 hours

---

## 1. What you will be able to do

Get JSON out of a model that your code can rely on — validated against a schema, with retries on
failure and a sane fallback — so that a bad generation degrades instead of crashing.

This is the least glamorous and most load-bearing skill in the list.

---

## 2. Prerequisites

- Skills: [01 · First LLM App](../01-first-llm-app/), [02 · Prompt Engineering](../02-prompt-engineering/)
- Concepts: [Structured Output](../../01-foundations/04-prompting-techniques/#39-structured-output)
- `pip install pydantic`

---

## 3. The concept, briefly

The moment model output feeds code instead of a human, prose becomes a bug. You need a guaranteed
shape.

There are three levels of guarantee, and you should understand why the first two are not enough:

1. **Ask nicely.** "Return JSON matching this schema." Works ~95% of the time. The other 5% is a
   `JSONDecodeError` in production, and it correlates with your hardest inputs.

2. **Constrain decoding.** Provider JSON mode / structured output / tool calling. The decoder is
   restricted so invalid JSON is *impossible*. This is a real guarantee — but only of **syntax**.
   The model can still return the wrong keys or nonsense values.

3. **Validate.** Parse into a typed schema with `pydantic`, check the semantics, and retry with the
   error message when it fails. This is the only level that guarantees your *contract*.

Use all three together. Level 2 without level 3 is the most common mistake in production LLM code.

---

## 4. Build it

1. **Define the schema first, in code.** A `pydantic` model with types, enums, and field
   constraints. The schema is the contract; the prompt is just how you communicate it.

2. **Generate the prompt from the schema.** Use `Model.model_json_schema()` rather than hand-writing
   the description. Now they cannot drift apart.

3. **Ask nicely and see it work.** Run 20 varied inputs. Count failures. Note that it mostly works —
   this is exactly why people stop here.

4. **Add fence stripping.** Models wrap JSON in ` ```json ` fences even when told not to. Strip
   them before parsing.

5. **Add validation.** Parse with `Model.model_validate_json()`. Catch `ValidationError` and keep
   the message — it is precise and actionable.

6. **Add retry-with-error.** On failure, send the model its own output plus the validation error
   and ask for a correction. Cap at 3 attempts. This recovers most failures on the first retry.

7. **Add native JSON mode.** Set `response_format={"type": "json_object"}` (Groq/OpenAI) or a
   response schema (Gemini). Rerun your 20 inputs and compare the failure rate.

8. **Add a fallback.** After 3 failed attempts, return a documented default or raise a typed
   application error. Never let a parse failure become a 500.

9. **Break it deliberately.** Set `max_tokens` too low so the JSON truncates mid-object. Confirm
   your validator catches it and your retry recovers. This is the failure mode you will actually
   hit in production, because it correlates with large inputs.

---

## 5. Checkpoint

Run 50 varied inputs, including adversarial ones, through your extractor:

- [ ] Every response either validates against the schema or returns your documented fallback
- [ ] Zero unhandled exceptions
- [ ] Truncated output (low `max_tokens`) is caught and recovered, not crashed on
- [ ] Markdown fences are handled
- [ ] Enum fields never contain a value outside the enum
- [ ] Retries are capped and logged
- [ ] You can state your measured failure rate before and after adding native JSON mode

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| `json.loads()` with no try/except | Crash on the first malformed response | Always wrap and validate |
| Trusting JSON mode alone | Valid JSON, wrong keys, silent bad data | Validate the schema too |
| No fence stripping | Fails on ` ```json ` wrappers | Strip before parsing |
| Retrying without the error message | The model repeats the same mistake | Feed back the exact validation error |
| Uncapped retries | Infinite loop, unbounded cost | Cap at 3, then fall back |
| Schema drift between prompt and code | Silent mismatch nobody notices for weeks | Generate the prompt from the schema |
| Deeply nested schemas | Failure rate climbs sharply | Flatten, or split into multiple calls |
| `max_tokens` too low for the schema | Truncated JSON on your largest inputs | Size it against your worst case |

---

## 7. Going further

- **Next skill:** [04 · Tool Calling](../04-tool-calling/) — same guarantees, applied to function arguments
- Try [Instructor](https://python.useinstructor.com/) or [Outlines](https://github.com/dottxt-ai/outlines) for constrained generation
- Measure how much the failure rate rises with schema depth, and write it down
- Add a metric for "retries per successful extraction" — it is an excellent early warning that a
  prompt or model change has degraded something

**Links:** [Pydantic](https://docs.pydantic.dev/) ·
[Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output) ·
[Example code](../../01-foundations/04-prompting-techniques/examples/05_structured_output.py)
