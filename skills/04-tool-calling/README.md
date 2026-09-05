# 04 · Tool Calling

**Level:** Intermediate · **Time:** ~4 hours

---

## 1. What you will be able to do

Expose your own Python functions to a model, let it decide when to call them, execute them safely,
and feed the results back — with argument validation, error recovery and a hard cap on the loop.

---

## 2. Prerequisites

- Skills: [03 · Structured Outputs](../03-structured-outputs/)
- Concepts: [ReAct](../../01-foundations/04-prompting-techniques/#35-react-reason--act)

---

## 3. The concept, briefly

Tool calling (a.k.a. function calling) is structured output pointed at your code. You describe your
functions as JSON schemas; the model, instead of answering, returns *which function to call and with
what arguments*; you execute it and hand back the result.

The loop:

```
  user message
      |
  model  --(no tool needed)-->  final answer
      |
   tool call {name, arguments}
      |
  YOUR CODE validates and executes     <-- the model never runs anything itself
      |
   result appended to messages
      |
  back to the model  (repeat, capped)
```

The critical line is the middle one. **The model does not execute anything.** It emits a request.
Everything about whether that request is safe is your responsibility — validation, authorisation,
rate limiting, and deciding whether this particular call should happen at all.

It fixes the two structural weaknesses of an LLM: it cannot know anything current, and it cannot
compute reliably. A tool call replaces a guess with a fact.

---

## 4. Build it

1. **Write two real functions first.** A calculator and something that fetches live data (weather,
   exchange rates, a local file search). Make them work standalone, with tests, before the model
   ever sees them.

2. **Describe them as schemas.** Name, description, parameters with types and descriptions. The
   *description* is a prompt — the model chooses tools by reading it. Vague descriptions produce
   wrong tool choices, and that is the number one cause of bad tool use.

3. **Make one round trip.** Pass `tools=[...]`, ask a question that needs one, and inspect
   `response.choices[0].message.tool_calls`. Do not execute yet — just look at what came back.

4. **Validate the arguments.** Parse them into a `pydantic` model. The model *will* eventually send
   a string where you wanted an int, or omit a required field.

5. **Execute and return.** Append the assistant's tool-call message, then a `tool` role message with
   the result and matching `tool_call_id`. Call the model again with the extended history.

6. **Loop it.** Wrap steps 3–5 in a `while` with `MAX_ITERATIONS`. Multi-step questions need
   several calls. An uncapped loop is an uncapped bill.

7. **Handle tool errors as data.** When your function raises, return the error message as the tool
   result rather than crashing. The model can often recover — retry with different arguments, or
   explain the problem to the user.

8. **Handle parallel calls.** Modern models return several tool calls at once. Execute them all and
   return a result for *every* `tool_call_id`, or the next request will be malformed.

9. **Add a safety boundary.** Allowlist which tools exist. Never `eval()` model-supplied strings.
   Never interpolate model output into SQL or a shell command. Require confirmation for anything
   destructive or outward-facing.

---

## 5. Checkpoint

- [ ] The model picks the right tool for a question that needs one
- [ ] It answers directly, with no tool call, when none is needed
- [ ] It chains two tools for a question requiring both
- [ ] Invalid arguments are caught by validation, not by your function crashing
- [ ] A tool that raises is reported back and the model recovers or explains
- [ ] Parallel tool calls all get results
- [ ] The loop stops at `MAX_ITERATIONS` with a clear message instead of spinning
- [ ] A prompt-injection string in a tool result does not cause another tool to fire
- [ ] You can print a full trace of every call, argument and result

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Vague tool descriptions | Wrong tool chosen, or none | Write the description as a prompt; say when *not* to use it |
| Trusting arguments | Crashes and injection risk | Validate with a schema, always |
| No iteration cap | Infinite loop, runaway cost | `MAX_ITERATIONS`, always |
| Missing a `tool_call_id` in the response | Malformed conversation, API error | Return a result for every call |
| Letting the model execute code | Arbitrary code execution | The model requests; your code decides and runs |
| Too many tools | Choice accuracy drops sharply past ~10–15 | Group them, or route in two stages |
| Treating tool output as trusted | Prompt injection via retrieved content | Treat every tool result as hostile input |
| No confirmation on destructive actions | It deletes something | Human approval for irreversible operations |

---

## 7. Going further

- **Next skills:** [05 · Embeddings & Vector Search](../05-embeddings-and-vector-search/), then
  [07 · Building Agents](../07-building-agents/)
- Learn [MCP](https://modelcontextprotocol.io/) — a standard protocol for exposing tools to models
- Add per-tool timeouts, rate limits and structured logging of every invocation
- Build a tool that calls another model, and think hard about the cost of that

**Links:** [ReAct paper](https://arxiv.org/abs/2210.03629) ·
[Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) ·
[Groq tool use](https://console.groq.com/docs/tool-use) ·
[Minimal ReAct example](../../01-foundations/04-prompting-techniques/examples/04_react_loop.py)
