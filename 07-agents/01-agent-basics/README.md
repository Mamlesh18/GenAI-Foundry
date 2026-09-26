# 01 · Agent Basics

> **In one sentence:** an agent is a **loop** around a language model — the model decides what to do
> next, your code does it, the result goes back into the context, and round it goes until the job is
> done or you stop it.

That is the entire concept. Everything else in this track is a refinement of those four lines.

---

## 1. Why a loop at all

A model on its own maps text to text. It cannot look anything up, cannot calculate reliably, and
cannot check whether what it just said was true. Give it a loop and a few functions and all three
become possible:

```
   task
     |
     v
  +-------------------------------------------------+
  |  MODEL: what should I do next?                  |  <--+
  +-------------------------------------------------+     |
     |                                                    |
     |  "call search('headcount')"                        |
     v                                                    |
  +-------------------------------------------------+     |
  |  YOUR CODE: is this allowed? then run it        |     |
  +-------------------------------------------------+     |
     |                                                    |
     |  observation: "The company has 420 employees."     |
     +----------------------------------------------------+
     |
     |  ...until the model answers, or the step cap is hit
     v
   answer
```

**The most important line in that diagram is the middle one.** The model never executes anything —
it *asks*. Your code decides whether to run it. Every safety property an agent has lives at that
boundary.

The pattern of interleaving reasoning with actions is called
**[ReAct](https://arxiv.org/abs/2210.03629)** (*Synergizing Reasoning and Acting in Language
Models*), and it is what nearly every agent framework implements under the hood.

---

## 2. The loop, running

From [`agent_loop.py`](examples/agent_loop.py), answering *"How many laptops does the company buy
per year, given the headcount?"*:

```
step 1: search('headcount')        -> The company has 420 employees.
step 2: search('laptop refresh')   -> Laptops are refreshed every 3 years.
step 3: calculator('420 / 3')      -> 140.0
step 4: ANSWER -> 140.0 laptops per year
```

Two facts the model did not have, one calculation it cannot do reliably, and an answer grounded in
both — in 4 model calls.

---

## 3. Workflow or agent? (usually: workflow)

Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
draws the line precisely:

> "**Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
> **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and
> tool usage."

So: is the *sequence of steps* known before the request arrives? If yes, you want a workflow — code,
not a loop. The script measures the difference on two tasks:

| | Fixed pipeline | Agent |
|---|---|---|
| **Task A** (always the same shape) correct? | **yes** | yes |
| Task A — model calls | **0** | 4 |
| **Task B** (varies per request) correct? | **no** | **yes** |
| Task B — model calls | 0 | 3 |

On task A the pipeline is right with **zero model calls**: the steps were known in advance, so no
decisions had to be made at runtime. Deterministic, testable, nearly free. On task B the pipeline
confidently answers a question nobody asked, because its steps are hardcoded — and the agent's
ability to choose what to look up is the only thing worth paying for.

**The rule: prefer the least agency that solves the problem.** A fixed pipeline you can test beats a
loop you have to supervise. Anthropic put it as: *"Success in the LLM space isn't about building the
most sophisticated system. It's about building the right system for your needs."*

---

## 4. Reliability compounds — the table that explains agent disappointment

An agent's task succeeds only if **every** step succeeds. Simulated over 20,000 trials per cell:

| Steps | 99% per step | 95% per step | 90% per step |
|---|---|---|---|
| 1 | 99.0% | 95.1% | 90.1% |
| 3 | 97.1% | 85.9% | 72.9% |
| 5 | 95.1% | 77.6% | 59.1% |
| 10 | 90.5% | **59.8%** | 34.7% |
| 20 | 81.8% | 35.8% | 12.1% |

A step that works **95%** of the time — which would be a good model — gives you **60%** end-to-end
over ten steps. The model did not get worse; 0.95¹⁰ is simply 0.60.

This is not only arithmetic. [τ-bench](https://arxiv.org/abs/2406.12045) measured real agents on
customer-service tasks and found leading function-calling agents succeeded on **under 50%** of tasks
and were badly inconsistent — `pass^8 < 25%` in the retail domain, meaning they rarely succeed eight
times out of eight on the same task. [GAIA](https://arxiv.org/abs/2311.12983) found humans at
**92%** against **15%** for a well-equipped GPT-4.

### What to do about it

| Strategy | Why it works |
|---|---|
| **Fewer steps** | Every step removed multiplies your success rate |
| **Check between steps** | Retry the *step*, not the whole task |
| **Make steps recoverable** | A retryable step cannot fail the task |
| **Human at the expensive points** | Approval before anything irreversible |
| **Measure `pass^k`, not `pass@1`** | Succeeding once is not the same as being reliable |

---

## 5. The four failure modes, and the code that contains them

| Failure | What happens | The fix, in code |
|---|---|---|
| **No stopping condition** | The model asks for the same tool forever | `MAX_STEPS`, plus a cost cap. Non-negotiable |
| **Tool errors** | An exception kills the run | Return the error *as an observation*; the model can then fix its own mistake |
| **A tool that does not exist** | The model invents `delete_everything` | An allowlist: unknown tool → error string, never execution |
| **Untrusted tool output** | A web page says "ignore your instructions" and the loop obeys | Treat every tool result as hostile input — see the Production module later in this track |

The script demonstrates all four. The looping model is stopped by the cap at 4 steps; the calculator
refuses `__import__("os")` because it validates its input rather than trusting it.

---

## 6. Examples

```bash
python examples/agent_loop.py      # instant, no dependencies, no API key
```

[`agent_loop.py`](examples/agent_loop.py) implements the loop in about 60 lines, compares it against
a fixed pipeline, simulates compounding reliability, and triggers each failure mode deliberately.
The model is mocked with deterministic rules so the output reproduces exactly — which also means it
cannot show you how a *real* model misbehaves. That is what the Production module later in this track is for.

---

## 7. Exercises

1. **Read the trace.** In section 1, what information made step 3 possible? What would have happened
   if step 2 had returned "No results found."?
2. **Remove the cap.** Set `max_steps` very high and run `LoopingModel`. Watch what an uncapped
   agent does, then put the cap back.
3. **Break a tool.** Make `search` raise an exception instead of returning a string. Does the loop
   survive? Now make it return an error observation — which behaviour would you want in production?
4. **Add a tool.** Give the agent a `today()` tool and a task that needs it. How much of the mock
   model's policy did you have to change, and what does that tell you about where the complexity
   lives in a real agent?
5. **Do the arithmetic.** Your agent needs 8 steps and each is 97% reliable. What is the end-to-end
   success rate? How many steps could you afford if you needed 90% overall?
6. **Convert an agent to a workflow.** Take task A and write the pipeline version without looking at
   `fixed_workflow`. What did you gain, and what did you give up?

---

## 8. Projects to build and test

### Beginner — An agent with three real tools
Build the loop against a real model API, with `search`, `calculator` and `read_file`.

**How to test it:** 10 tasks needing different tool combinations. Report success rate, mean steps and
mean cost. Then give it a task no tool can solve and confirm it says so rather than inventing.

### Intermediate — Workflow vs agent, measured on your own task
Implement the same real task both ways.

**How to test it:** run 20 varied inputs through both. Report success rate, latency and cost. State
which you would ship — and if it is the workflow, that is a successful project, not a failed one.

### Advanced — A reliability harness
Run one agent task 20 times and report `pass@1` and `pass^k`.

**How to test it:** the gap between those two numbers is your real reliability story. Then add
between-step validation and show the gap narrowing.

---

## 9. Resources

**Start here**
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) — Anthropic. The best single article on this subject; the workflow/agent distinction above is theirs.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the paper behind the loop.

**Go deeper**
- [Reflexion](https://arxiv.org/abs/2303.11366) — agents that critique their own attempts and retry.
- [τ-bench](https://arxiv.org/abs/2406.12045) — measured agent reliability, and the `pass^k` metric.
- [GAIA](https://arxiv.org/abs/2311.12983) — a benchmark where humans score 92% and agents 15%.
- [AI Agent Failure Modes: tool-calling errors, infinite loops, propagation](https://www.openlayer.com/blog/ai-agent-failure-modes-tool-calling-loops-propagation) — a practical catalogue of what goes wrong.

**Related in this repo**
- [01-foundations/04 · ReAct](../../01-foundations/04-prompting-techniques/#35-react-reason--act) — the prompting pattern
- [06-rag/05 · Agentic RAG](../../06-rag/05-agentic-rag/) — this loop applied to retrieval
- [skills/07 · Building Agents](../../skills/07-building-agents/) — the hands-on checkpoint version

---

**Track:** [07 · Agents](../) · **Next:** [02 · Tool Use](../02-tool-use/)
