# 01 · Chunking

> **In one sentence:** chunking decides where your documents get cut, and therefore what can ever
> be retrieved — because **retrieval returns chunks**.

It is the least glamorous step in RAG and the one that most often decides whether the whole thing
works.

---

## 1. Why the cut matters

If the answer to a question is split across two chunks, **no retriever, reranker or model can put
it back together.** The information simply is not in any one unit that retrieval can return.

[`chunking_strategies.py`](examples/chunking_strategies.py) measures exactly this. It takes a
handbook and eight questions whose answers occupy known spans, then asks of each strategy: *for how
many questions does some single chunk contain the whole answer?*

| Strategy | Chunks | Avg words | Answers intact |
|---|---|---|---|
| fixed 20 words, no overlap | 8 | 19 | **62%** |
| fixed 20 words, 50% overlap | 14 | 20 | 100% |
| fixed 60 words, no overlap | 3 | 50 | 88% |
| fixed 60 words, 25% overlap | 3 | 60 | 100% |
| whole document as one chunk | 1 | 150 | 100% |
| 3 sentences per chunk | 5 | 30 | 100% |
| by section heading | 4 | 34 | 100% |

With 20-word chunks and no overlap, **three of eight answers were cut in half** before retrieval
even began. Adding overlap recovered all three.

---

## 2. The trade-off, in one line

```
  small chunks              large chunks
  precise retrieval         vague retrieval
  lost context      <-->    context preserved
  answers get cut           answers survive
  cheap prompts             expensive prompts
```

**Small chunks** produce a sharp vector about one idea — easy to match precisely — but lose the
surrounding context and cut answers in half.

**Large chunks** keep everything together, but one vector now has to represent many topics, so it
matches everything weakly and nothing strongly. You also pay for irrelevant text in every prompt,
and give the model more material to be distracted by.

---

## 3. The strategies

| Strategy | How | When |
|---|---|---|
| **Fixed size** | N tokens/words per chunk, with overlap | The default. Simple, predictable, hard to beat |
| **Recursive** | Split on paragraphs, then sentences, then words, until small enough | The usual production default (LangChain's `RecursiveCharacterTextSplitter`) |
| **Sentence / paragraph** | Natural language boundaries | Prose without headings |
| **Structure-aware** | Split on headings, list items, table rows, code functions | **Best when the document has structure** — use what the author already marked |
| **Semantic** | Split where the topic shifts, detected by embedding similarity | Sounds principled; benchmarks often find it does not repay its cost |
| **Late chunking** | Embed the whole document first, then pool token vectors per chunk | Newer; keeps document-wide context inside each chunk vector |

---

## 4. Defaults that work

- **~500 tokens per chunk, 10–20% overlap.** Published comparisons keep finding recursive splitting
  at roughly this size hard to beat. Shorter (256–512) suits precise factoid lookup; longer
  (512–1,024) suits analytical questions needing more context.
- **Split on structure first**, falling back to size only inside a section.
- **Keep metadata on every chunk**: source, section, date, author, permissions. You need it to cite,
  to filter ([03 · Vector Stores](../03-vector-stores/)), and to debug.
- **Then measure on your own questions.** Chunking is the cheapest parameter to change and has the
  largest effect on retrieval quality.

---

## 5. Give each chunk its context

A chunk that reads *"reimbursed up to 40 pounds per day"* does not say what it is about. Retrieval
for "meal allowance" may never find it.

The cheap fix, shown in the script, is to prepend the document and section title:

```
[Employee Handbook > Expenses] Travel expenses must be submitted within 30 days...
```

Anthropic's **[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)** does a
richer version: an LLM writes a sentence situating each chunk in its document, prepended before
embedding. Their measured results, on top-20 retrieval failure rate:

| Technique | Failure rate reduction |
|---|---|
| Contextual embeddings | 35% (5.7% → 3.7%) |
| + contextual BM25 | 49% (5.7% → 2.9%) |
| + reranking | **67% (5.7% → 1.9%)** |

It costs one cheap LLM call per chunk at index time — paid once, benefiting every query afterwards.

---

## 6. Beyond flat chunks

| Technique | Idea |
|---|---|
| **Sentence-window retrieval** | Embed single sentences for precision, but return the surrounding paragraph for context |
| **Parent-document retrieval** | Search small chunks, hand the model the larger parent section |
| **[RAPTOR](https://arxiv.org/abs/2401.18059)** | Recursively summarise chunks into a tree, so both details and overviews are retrievable |
| **[GraphRAG](https://arxiv.org/abs/2404.16130)** | Build an entity graph, for questions that span a whole corpus rather than one passage |

All of these exist because flat, fixed-size chunks are a compromise. Reach for them when you have
measured the compromise hurting.

---

## 7. Examples

```bash
python examples/chunking_strategies.py     # instant, no dependencies
```

Measures answer survival across eight strategies, shows which questions get cut in half by
small chunks, and demonstrates title-prepending.

---

## 8. Exercises

1. **Find the broken answers.** Run the script. Which three questions does 20-word chunking destroy,
   and what do they have in common?
2. **Sweep the size.** Try 30, 40, 80 and 120 words with 15% overlap. Plot answers-intact against
   chunk size. Where does it plateau?
3. **Chunk your own document.** Take a real document of yours and chunk it three ways. Print the
   chunks and read them. Would *you* be able to answer a question from any one of them alone?
4. **Structure beats size.** Find a document where fixed-size chunking cuts a table or code block in
   half. What would a structure-aware splitter do instead?
5. **Write the context line.** For three chunks from your document, write the one-sentence context
   an LLM would prepend. How much easier would they be to retrieve?

---

## 9. Projects to build and test

### Beginner — A chunk inspector
A script that chunks a document and prints each chunk with its word count and metadata.

**How to test it:** read 20 chunks yourself and mark whether each is self-contained. That
percentage is a better quality signal than any automatic metric at this stage.

### Intermediate — Chunk size experiment
Build 20 questions with known answers over your own corpus, then measure retrieval recall at four
chunk sizes and two overlap settings.

**How to test it:** publish the table. The best setting should beat the worst by a clear margin —
if not, your questions are too easy or your corpus too small.

### Advanced — Contextual chunking
Use a cheap model to write a one-sentence context for each chunk at index time, prepend it, and
re-measure retrieval.

**How to test it:** report recall before and after, plus the one-off indexing cost. Compare your
improvement against Anthropic's reported 35% and explain any difference.

---

## 10. Resources

- [Chunking Strategies for RAG: A Complete Guide](https://atlan.com/know/chunking-strategies-rag/) — a thorough tour of the options.
- [RAG Chunking Strategies: The 2026 Benchmark Guide](https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/) — measured comparisons rather than opinions.
- [Best Chunking Strategies for RAG](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) — practical, with code.
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic; the numbers quoted above.
- [RAPTOR](https://arxiv.org/abs/2401.18059) · [GraphRAG](https://arxiv.org/abs/2404.16130) — structures beyond flat chunks.

---

**Track:** [06 · RAG](../) · **Next:** [02 · Embeddings](../02-embeddings/)
