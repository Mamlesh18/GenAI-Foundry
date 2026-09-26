# 06 · RAG

> **In one sentence:** RAG (Retrieval-Augmented Generation) finds the documents relevant to a
> question and puts them in the prompt, so the model answers from *your* data instead of from
> memory.

It is the most common way to build something useful with an LLM, and the technique that turns
"a model that sounds knowledgeable" into "a system that can cite the handbook".

---

## 1. The problem it solves

A model knows only what it saw in training ([01-foundations/02](../01-foundations/02-llms/)). So it
cannot answer questions about your contracts, your codebase, or last Tuesday — and when asked, it
will often invent something ([05-evaluation/02](../05-evaluation/02-hallucination/)).

RAG's answer is disarmingly simple: **look it up first, then answer**. Instead of asking the model
to recall, you hand it the relevant page and ask it to read.

---

## 2. How it works

Two pipelines. The first runs when documents change; the second runs on every question.

```
INDEXING  (offline, runs when your documents change)

   documents (PDF, HTML, Markdown, database rows)
        |
   [ 1. CHUNK ]        cut into passages of a few hundred words
        |
   [ 2. EMBED ]        each chunk becomes a vector
        |
   [ 3. STORE ]        vectors + text + metadata -> a vector store


QUERYING  (online, runs for every question, in milliseconds)

   "How many leave days do I get?"
        |
   [ 4. EMBED the question ]
        |
   [ 5. SEARCH ]       find the nearest chunks     <-- optionally: keyword
        |                                              search too, then rerank
   [ 6. BUILD PROMPT ] context + question + "answer ONLY from this;
        |               say NOT_FOUND if it is not here; cite sources"
        |
   [ 7. GENERATE ]     the model reads and answers
        |
   "24 days per year [handbook/leave.md]"
```

Step 6 is the entire "augmented" part. There is no magic in RAG — it is a well-built prompt with
the right paragraphs pasted into it. [`examples/mini_rag.py`](examples/mini_rag.py) does all seven
steps in about 200 lines and prints the finished prompt so you can see it.

---

## 3. The modules

| # | Module | The question it answers |
|---|---|---|
| 01 | [Chunking](01-chunking/) | Where do I cut the documents? (the step that most often decides quality) |
| 02 | [Embeddings](02-embeddings/) | What does "similar" mean, and which model should I use? |
| 03 | [Vector Stores](03-vector-stores/) | Where do the vectors live, and which provider should I pick? |
| 04 | [Retrieval](04-retrieval/) | How do I actually find the right passage? |
| 05 | [Agentic RAG](05-agentic-rag/) | What if one search is not enough? |

---

## 4. When to use RAG — and when not to

| Situation | Use |
|---|---|
| Answering from private, large or changing documents | **RAG** |
| The answer must be attributable to a source | **RAG** (citations) |
| Teaching the model a *format*, *tone* or *task* | [Fine-tuning](../02-training/02-fine-tuning/) |
| The whole corpus is small (a few thousand words) | Just put it in the prompt |
| The task needs live actions, not documents | [Tools and agents](../07-agents/) |

**RAG vs fine-tuning** is the question people most often get backwards:

| | RAG | Fine-tuning |
|---|---|---|
| Adds | knowledge | behaviour, style, format |
| Update cost | re-index a document, seconds | retrain, hours to days |
| Can cite sources | yes | no |
| Handles changing data | yes | no |

They are complementary, not alternatives. If you want a model that *answers in your house style*
**and** *from your documents*, you fine-tune for the first and retrieve for the second.

**"Why not just use a long context window?"** Models now accept hundreds of thousands of tokens,
so why retrieve at all? Because sending 200,000 tokens costs 200,000 tokens' worth of money and
latency on *every* request, models attend less reliably to material buried in the middle
([Lost in the Middle](https://arxiv.org/abs/2307.03172)), and your corpus is probably far larger
than any context window anyway. Retrieval is how you send 2,000 relevant tokens instead of 200,000
mostly-irrelevant ones.

---

## 5. Where RAG actually fails

Almost every disappointing RAG system fails in the same place, and it is not the model:

| Failure | Where it happens | Fix |
|---|---|---|
| The right passage was never retrieved | steps 1–5 | **Start here.** Better chunking, hybrid search, reranking |
| It retrieved the right passage and still answered wrongly | step 7 | Stronger grounding instruction, a better model |
| It answered when it should have refused | step 6 | Add an explicit NOT_FOUND escape hatch |
| It cited the wrong source | step 7 | Verify citations; require source ids |
| The documents themselves are wrong or stale | step 0 | No retrieval strategy fixes bad source data |

**The diagnostic rule: when an answer is wrong, look at what was retrieved before you touch the
prompt.** Most of the time the model was faithfully answering from the wrong paragraph.

---

## 6. Measuring it

RAG has two halves and they fail differently, so measure them separately
([05 · Evaluation](../05-evaluation/)):

| Metric | Half | Question |
|---|---|---|
| Context recall | retrieval | Did we fetch the passage containing the answer? |
| Context precision | retrieval | How much of what we fetched was useful? |
| Faithfulness | generation | Is every claim supported by the retrieved text? |
| Answer correctness | both | Is the final answer right? |
| Refusal accuracy | both | Does it say NOT_FOUND exactly when it should? |

[Ragas](https://docs.ragas.io/) implements most of these. Build a set of 30–50 real questions with
known answers before you tune anything — otherwise you are guessing.

---

## 7. Frameworks

You do **not** need one to start; `mini_rag.py` has no dependencies beyond numpy. They become
worthwhile when you need many document loaders and integrations.

| Framework | Good for |
|---|---|
| [LlamaIndex](https://docs.llamaindex.ai/) | RAG-first; loaders, indexes and query engines |
| [LangChain](https://python.langchain.com/) | Broad ecosystem, many integrations |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Agentic RAG loops with explicit state |
| [Haystack](https://haystack.deepset.ai/) | Production pipelines, strong on search |
| [DSPy](https://dspy.ai/) | Optimising the prompts and pipeline programmatically |

---

## 8. Examples

```bash
pip install numpy                     # sentence-transformers strongly recommended
python examples/mini_rag.py
```

[`examples/mini_rag.py`](examples/mini_rag.py) is the whole track in one file: it chunks a small
handbook, embeds it, retrieves, prints the exact prompt, answers three questions with citations,
and **refuses** a fourth that the handbook cannot answer. Each module then goes deeper into one step.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **Chunk** | A passage of a document, the unit that gets embedded and retrieved |
| **Embedding** | A vector representing a piece of text's meaning |
| **Vector store** | The database that holds vectors and searches them by similarity |
| **Top-k** | How many chunks you retrieve per question |
| **Hybrid search** | Combining keyword search with vector search |
| **Reranking** | A slower, more accurate second pass over a shortlist |
| **Grounding** | Instructing the model to answer only from the provided context |
| **Faithfulness** | Whether the answer is actually supported by that context |
| **Multi-hop** | A question needing facts from more than one document |

---

## 10. Exercises

1. **Trace the pipeline.** Run `mini_rag.py` and follow one question through all seven steps. Which
   step would you change first if the answer were wrong?
2. **Break retrieval.** Delete the overlap in `chunk_document` and ask a question whose answer sits
   at a chunk boundary. What happens, and why can no prompt fix it?
3. **Remove the escape hatch.** Take the NOT_FOUND line out of the prompt template. Ask the
   unanswerable question. What does the system do instead?
4. **Decide the approach.** For each: a support bot over 10,000 help articles; a model that always
   replies in your company's tone; a tool that reads today's stock prices. RAG, fine-tuning, or
   tools?

## 11. Projects

- **Beginner — RAG over your own notes.** Point `mini_rag.py` at a folder of your own markdown
  files. *Test:* write 10 questions you know the answers to, and measure how many it gets right and
  how many it wrongly refuses.
- **Intermediate — Add evaluation.** Build a 30-question set with known source documents and
  measure context recall and faithfulness separately. *Test:* deliberately break chunking and
  confirm context recall drops while faithfulness does not.
- **Advanced — Ship one.** A document Q&A service with hybrid retrieval, reranking, citations and
  refusal, behind an API. *Test:* report recall@5, faithfulness, refusal accuracy, p95 latency and
  cost per question — all five, or you have not measured it.

---

## 12. Resources

**Start here**
- [What is RAG?](https://aws.amazon.com/what-is/retrieval-augmented-generation/) — AWS. A clear, vendor-neutral explanation of the concept.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the original 2020 paper that named it.
- [LlamaIndex: high-level concepts](https://docs.llamaindex.ai/) — the indexing/querying split, clearly documented.

**Go deeper**
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic. Measured improvements from adding context to chunks; a rare article with real numbers.
- [Ragas documentation](https://docs.ragas.io/) — the metrics that tell you which half is broken.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — why stuffing everything into a long context is not a substitute for retrieval.

---

**Previous track:** [05 · Evaluation](../05-evaluation/) · **Next track:** [07 · Agents](../07-agents/)
