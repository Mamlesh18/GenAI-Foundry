# 05 · Agentic RAG

> **In one sentence:** instead of retrieving once with the user's question, the model *decides* what
> to search for, judges what comes back, and searches again until it has what it needs.

---

## 1. Why one search is sometimes not enough

Ordinary RAG is a straight line: embed the question → retrieve → answer. It works well when the
answer sits in one passage. It cannot work when:

| Situation | Why one search fails |
|---|---|
| **Multi-hop question** | "Who manages the team that owns the billing service?" The manager's name is in a document that never mentions billing |
| **The question is a bad query** | "What about the second one?" means nothing to a retriever |
| **Nothing useful comes back** | A fixed pipeline has no way to try again |
| **Wrong source** | The answer is in the database, not the documents |

The agentic fix: **make retrieval a loop the model controls.**

---

## 2. The loop

```
        question
           |
      [ ROUTE ]  ---- no retrieval needed ----> answer directly
           |
      [ PLAN ]   what should I search for?
           |     (rewrite the question; split it into sub-questions)
           v
   +-> [ RETRIEVE ]
   |       |
   |  [ GRADE ]   is each chunk actually relevant?
   |       |
   |  enough evidence? --- no ---+     (up to MAX_STEPS)
   |       |                     |
   |      yes                    |
   |       |         [ REFORMULATE ] use what you just learned
   |       v                     |     to write the next query
   |  [ ANSWER ]  <--------------+
   |   with citations, or NOT_FOUND
```

Five roles, each one a prompt you write: **router**, **planner**, **retriever**, **grader**,
**generator**. [`agentic_rag.py`](examples/agentic_rag.py) implements all five.

---

## 3. What it looks like running

The trace from the script, for *"Who manages the team that owns the billing service?"*:

```
RETRIEVE #1: "Who manages the team that owns the billing service?" -> docs [0, 2]
  GRADE doc 0: relevant -- keep
  GRADE doc 2: nothing distinctive in common -- discard
  PLAN: still missing a link -> search "Team Aurora managed by"
RETRIEVE #2: "Team Aurora managed by" -> docs [1, 3]
  GRADE doc 1: relevant -- keep
  GRADE doc 3: nothing distinctive in common -- discard
  PLAN: evidence is sufficient -> answer
ANSWER: Team Aurora is managed by Priya Nair, who reports to the VP of Engineering.
```

**The second query appears nowhere in the question.** The agent wrote it after learning from the
first hop that billing belongs to Team Aurora. That is the whole idea.

---

## 4. Does it help? — measured

From the same script, across three kinds of question:

| Question type | Naive correct | Agentic correct | Naive searches | Agentic searches |
|---|---|---|---|---|
| single-hop | 2/3 | 2/3 | 1.0 | 1.0 |
| **multi-hop** | **1/3** | **3/3** | 1.0 | 2.0 |
| **unanswerable** | **0/2** | **2/2** | 1.0 | 1.0 |

Three honest readings:

1. **Multi-hop is where it wins**, and it wins decisively.
2. **Single-hop gains nothing** and still pays for grading and planning.
3. **The grader is what produces refusals.** It discards documents that merely look similar, so the
   generator has nothing to write from and correctly says NOT_FOUND. Naive RAG answered all of them,
   because its nearest chunk is always *something*.

---

## 5. The patterns, and their names

| Pattern | What it does |
|---|---|
| **Query rewriting** | Turn the question (or a follow-up) into a good standalone search query |
| **Decomposition** | Split "compare A and B" into separate retrievals |
| **Multi-hop** | Use what hop 1 found to write hop 2's query |
| **Grading** | Judge each retrieved chunk before trusting it |
| **Self-correction** | Weak evidence → search again, widen, or say so. [CRAG](https://arxiv.org/abs/2401.15884) adds a web-search fallback |
| **Self-reflection** | Critique the draft answer against the evidence. [Self-RAG](https://arxiv.org/abs/2310.11511) trains the model to do this with special tokens |
| **Routing** | Choose the source: documents, SQL, the web, or no retrieval at all |
| **[HyDE](https://arxiv.org/abs/2212.10496)** | Write a hypothetical answer and search with *that* — answers resemble documents more than questions do |

---

## 6. What it costs

| Cost | Detail |
|---|---|
| **Latency** | Three searches plus three model calls is several seconds before the first token |
| **Money** | Each planning and grading step is an LLM call; 3–10× a naive pipeline |
| **Unpredictability** | Two identical questions can take different paths |
| **Debuggability** | Without a trace you cannot tell why it answered as it did |
| **Loops** | An uncapped agent can search forever |

Non-negotiables: **cap the loop** (`MAX_STEPS`), **log the full trace**, and **cap the cost** per
request.

---

## 7. When to reach for it

**Use agentic RAG when** questions genuinely span multiple documents, users ask follow-ups in
conversation, you have several sources needing a routing decision, or "I don't know" must be
reliable.

**Stay with a straight pipeline when** questions are single-hop lookups, latency matters more than
completeness, or — most importantly — **you have not yet fixed retrieval**. An agent searching a
badly chunked corpus just fails more expensively. Chunking, hybrid search and reranking come first,
every time.

---

## 8. Building one

| Tool | Notes |
|---|---|
| **Nothing** | The script here is ~250 lines of standard library. A loop, a few prompts, and a cap |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Explicit graph of states; good when the loop gets complex |
| [LlamaIndex](https://docs.llamaindex.ai/) | Query engines, routers and sub-question decomposition built in |
| [DSPy](https://dspy.ai/) | Optimises the prompts in the pipeline rather than hand-tuning them |

See [07 · Agents](../../07-agents/) for the general agent loop, of which this is one special case:
retrieval is simply the tool.

---

## 9. Examples

```bash
python examples/agentic_rag.py     # instant, no dependencies
```

Shows naive RAG failing a multi-hop question, prints the full agent trace, and measures both
approaches across single-hop, multi-hop and unanswerable questions with their retrieval cost.

---

## 10. Exercises

1. **Read the trace.** In section 2 of the output, which step turns a failure into a success? What
   information made the second query possible?
2. **Remove the grader.** Make `MockLLM.grade` always return `True`. What happens to the
   unanswerable questions, and why?
3. **Lower the cap.** Set `MAX_STEPS = 1`. Which questions break? That is your measure of how much
   the loop is doing.
4. **Add a three-hop question.** Write one needing three documents (for example, which office the
   manager of the team owning a service is based in). Does the planner reach it?
5. **Count the cost.** If each search costs 20 ms and each LLM call costs 800 ms and $0.002, what is
   the latency and price of one agentic answer versus a naive one?
6. **Route first.** Add a rule that sends greetings and small talk straight to an answer with no
   retrieval. How much does that save on a realistic traffic mix?

---

## 11. Projects to build and test

### Beginner — Query rewriting for conversation
Add a step that rewrites a follow-up question into a standalone one using the chat history, before
retrieval.

**How to test it:** a 5-turn conversation with pronouns and ellipsis ("what about the other one?").
Compare retrieval recall with and without the rewrite.

### Intermediate — A grader you can trust
Build a real relevance grader with an LLM and validate it against your own labels.

**How to test it:** hand-label 50 (query, chunk) pairs, then report the grader's precision and
recall. A grader that discards good evidence is worse than no grader at all.

### Advanced — Agentic vs naive, on your data
Build both over the same corpus and run a labelled question set containing single-hop, multi-hop
and unanswerable questions.

**How to test it:** report accuracy by question type, mean searches, mean latency and cost per
question. Then answer honestly: does the agentic version earn its cost on *your* traffic mix?

---

## 12. Resources

**Start here**
- [Agentic RAG Explained in 3 Levels of Difficulty](https://machinelearningmastery.com/agentic-rag-explained-in-3-levels-of-difficulty/) — a gentle build-up from plain RAG.
- [Agentic RAG: Architectures, Tradeoffs, and How to Build It](https://mastra.ai/articles/agentic-rag) — practical patterns and where they break.

**Papers**
- [Agentic Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2501.09136) — the taxonomy of these architectures.
- [Self-RAG](https://arxiv.org/abs/2310.11511) — training a model to retrieve and critique itself.
- [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884) — grading retrieval and falling back to web search.
- [HyDE](https://arxiv.org/abs/2212.10496) — searching with a hypothetical answer.

**Related in this repo**
- [07 · Agents](../../07-agents/) — the general loop
- [01-foundations/04 · ReAct](../../01-foundations/04-prompting-techniques/#35-react-reason--act) — the reason/act pattern this is built on

---

**Previous:** [04 · Retrieval](../04-retrieval/) · **Track:** [06 · RAG](../) ·
**Next track:** [07 · Agents](../../07-agents/)
