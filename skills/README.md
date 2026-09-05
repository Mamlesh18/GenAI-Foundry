# Skills

The concept folders (`01-foundations` onward) teach you *why*. This folder teaches you *how*.

A **skill** is one self-contained, practical capability you can learn in a sitting and then reuse
forever. Each has a clear definition of done: a checkpoint you either pass or do not. No skill is
complete because you read it — it is complete because you built the thing and it worked.

---

## The skill tree

Skills build on each other. Arrows mean "do this first".

```
  01 First LLM App
        |
        +--> 02 Prompt Engineering
        |         |
        |         +--> 03 Structured Outputs
        |                    |
        |                    +--> 04 Tool Calling ------+
        |                                               |
        +--> 05 Embeddings & Vector Search              |
                  |                                     |
                  +--> 06 RAG Pipeline                  |
                            |                           |
                            +-------> 07 Agents <-------+
                                          |
        08 Evaluation & Testing <---------+
                  |
                  +--> 09 Shipping to Production
                            |
                            +--> 10 Fine-Tuning  (only when prompting genuinely is not enough)
```

| # | Skill | You will be able to | Level | Time |
|---|---|---|---|---|
| 01 | [First LLM App](01-first-llm-app/) | Call an LLM API, stream responses, manage conversation state | Beginner | 2 h |
| 02 | [Prompt Engineering](02-prompt-engineering/) | Diagnose a bad output and pick the technique that fixes it | Beginner | 4 h |
| 03 | [Structured Outputs](03-structured-outputs/) | Get reliable, schema-validated JSON out of a model | Beginner | 3 h |
| 04 | [Tool Calling](04-tool-calling/) | Let a model use your functions safely | Intermediate | 4 h |
| 05 | [Embeddings & Vector Search](05-embeddings-and-vector-search/) | Build semantic and hybrid search over your own data | Intermediate | 4 h |
| 06 | [RAG Pipeline](06-rag-pipeline/) | Answer questions over documents, with citations and no hallucinations | Intermediate | 6 h |
| 07 | [Building Agents](07-building-agents/) | Ship a multi-step agent that does not loop forever or leak your keys | Advanced | 8 h |
| 08 | [Evaluation & Testing](08-evaluation-and-testing/) | Prove a change made things better instead of guessing | Intermediate | 5 h |
| 09 | [Shipping to Production](09-shipping-to-production/) | Handle cost, latency, failures, caching and abuse | Advanced | 6 h |
| 10 | [Fine-Tuning](10-fine-tuning/) | Adapt a model with LoRA when prompting has genuinely run out | Advanced | 8 h |

---

## How to use a skill folder

Every one follows the same structure:

1. **What you will be able to do** — the concrete capability, stated as a verb.
2. **Prerequisites** — skills and concepts assumed.
3. **The concept, briefly** — enough theory to act, with links to the deep version.
4. **Build it** — numbered steps, from empty folder to working thing.
5. **Checkpoint** — the test that decides whether you have the skill.
6. **Common mistakes** — the specific ways people get this wrong.
7. **Going further** — where to take it next.

Work in your own directory. Do not copy code into this repo unless you are contributing it back —
the point is the muscle memory of building it yourself.

---

## An honest note on order

**Do 01 → 03 before anything else.** Almost everyone skips structured outputs and goes straight to
agents, then spends a fortnight debugging a system that fails because a model returned prose where
their code expected JSON. Reliable output shape is the foundation everything else stands on.

**Do 08 earlier than feels natural.** Without evaluation you cannot tell whether your changes are
improvements. Most people build for months on vibes, then discover they cannot answer "is this
better than what we had in March?"

**Do 10 last, and probably not at all.** Fine-tuning is the answer far less often than people
expect. Better prompting, better retrieval and better evaluation solve most problems more cheaply.
Reach for it when you have measured a specific gap that prompting cannot close.

---

## Contributing a skill

Copy [`_TEMPLATE/`](_TEMPLATE/), keep the seven sections, and make sure the checkpoint is
genuinely pass/fail. A skill without a checkpoint is a blog post.
