# 07 · Agents

> **In one sentence:** an agent is a language model in a **loop**, with tools it can ask for and a
> stopping rule — so instead of producing one answer, it takes steps towards a goal.

This is the most over-promised topic in the field and the one where careful engineering matters
most. The track is built around a single organising idea: **agency is a cost, not a feature.** Use
the least of it that solves your problem.

---

## 1. What changes when you add a loop

| | A plain model call | An agent |
|---|---|---|
| Interaction | one request, one answer | many steps, each informed by the last |
| Can it act? | no | yes — via tools your code runs |
| Can it recover from a mistake? | no | yes, if you let it see the error |
| Cost | predictable | variable, and can run away |
| Testable? | mostly | only with effort, because runs differ |

The loop buys **recovery** and **flexibility**, and it costs **predictability**. Everything in this
track is about getting the first two without losing too much of the third.

---

## 2. The anatomy

```
                        +---------------------------------+
   task  --------->     |   MODEL: what should I do next? |
                        +---------------------------------+
                                     |
                       action request (tool name + arguments)
                                     v
                        +---------------------------------+
                        |   YOUR CODE                     |
                        |   - is this tool allowed?        |
                        |   - are the arguments valid?     |
                        |   - does a human need to approve?|
                        |   - run it, catch failures       |
                        +---------------------------------+
                                     |
                              observation
                                     |
   +---------------------------------+
   |   CONTEXT: task + every step so far  (this fills up -- module 03)
   +---------------------------------+
                                     |
                      answer, or the step/cost cap
```

Four things to notice, because they are where the work is:

1. **The model never executes.** It requests; your code decides. Every safety property lives there.
2. **The context grows with every step**, and it is finite. Managing it is its own discipline.
3. **Something must stop the loop** — a step cap, a cost cap, or both.
4. **Every observation is untrusted input**, because a tool result can be written by someone else.

---

## 3. The modules

| # | Module | What it covers |
|---|---|---|
| 01 | [Agent Basics](01-agent-basics/) | The loop, ReAct, workflow vs agent, and why reliability compounds |
| 02 | Tool Use | Schemas, argument validation, error recovery, MCP |
| 03 | Memory and Context | The context budget, scratchpads, summarisation, long-term memory |
| 04 | Agent Patterns | Chaining, routing, parallelisation, orchestrator-workers, evaluator-optimizer, reflection |
| 05 | Multi-Agent | Supervisors, handoffs, debate — and when one agent is better |
| 06 | Frameworks | LangChain, LangGraph, CrewAI, AutoGen, the vendor SDKs, and whether you need any |
| 07 | Production | Agent evaluation, tracing, guardrails, prompt injection, human-in-the-loop, cost caps |

*(Modules are added one at a time; links appear as each lands.)*

---

## 4. The uncomfortable facts, up front

Before building anything, know what the measurements say:

- **Reliability compounds.** A 95%-reliable step gives **60%** success over ten steps. Module 01
  measures this.
- **Agents are inconsistent, not just imperfect.** [τ-bench](https://arxiv.org/abs/2406.12045) found
  leading function-calling agents succeed on under 50% of customer-service tasks, and `pass^8 < 25%`
  in retail — they rarely succeed eight times out of eight on the *same* task.
- **The gap to humans is still wide on open-ended work.**
  [GAIA](https://arxiv.org/abs/2311.12983): humans **92%**, a well-equipped GPT-4 **15%**.
- **Most "agent" problems are workflow problems.** If you know the steps in advance, write the steps.

None of this means agents are useless — it means the engineering is about *containing* variance:
fewer steps, checks between them, caps, traces, and a human at the expensive moments.

---

## 5. How to work through this track

**If you are new to agents:** 01 → 02 → 04. That is enough to build something real and know why it
behaves as it does.

**If you already have an agent that misbehaves:** 03 (context is usually the problem), then
07 (evaluation and guardrails).

**If someone is asking you for a multi-agent system:** read 05 first, then decide — the honest
answer is often one agent with better tools.

---

## 6. Glossary

| Term | Meaning |
|---|---|
| **Tool / function** | A function your code exposes for the model to request |
| **Tool call** | The model's structured request: a name plus arguments |
| **Observation** | What a tool returned, fed back into the context |
| **Trajectory** | The full sequence of steps taken on one task |
| **Step / iteration** | One pass of the loop |
| **Scratchpad** | Working notes the agent keeps across steps |
| **Handoff** | One agent passing control to another |
| **Supervisor / orchestrator** | An agent whose job is delegating to other agents |
| **Guardrail** | A check outside the model that constrains what may happen |
| **`pass^k`** | Whether the agent succeeds on *all* k attempts — the honest reliability measure |
| **MCP** | Model Context Protocol, a standard way to expose tools to models |

---

## 7. Resources for the whole track

**Start here**
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) — Anthropic. Read it before writing any agent code; the workflow/agent distinction and the five patterns come from here.
- [ReAct](https://arxiv.org/abs/2210.03629) — the reason-and-act loop that underlies nearly every framework.

**Measure the claims**
- [τ-bench](https://arxiv.org/abs/2406.12045) — tool-agent-user tasks, and the `pass^k` metric.
- [GAIA](https://arxiv.org/abs/2311.12983) — general assistant tasks that are easy for humans and hard for agents.
- [AI agent failure modes](https://www.openlayer.com/blog/ai-agent-failure-modes-tool-calling-loops-propagation) — a practical catalogue of how they break.

**Related in this repo**
- [06-rag/05 · Agentic RAG](../06-rag/05-agentic-rag/) — the loop applied to retrieval
- [skills/04 · Tool Calling](../skills/04-tool-calling/) · [skills/07 · Building Agents](../skills/07-building-agents/) — the hands-on versions with pass/fail checkpoints
- [05 · Evaluation](../05-evaluation/) — how to tell whether any of this works

---

**Previous track:** [06 · RAG](../06-rag/) · **Next track:** [08 · Multimodal](../08-multimodal/)
