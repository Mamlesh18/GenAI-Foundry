# 04 · Retrieval

> **In one sentence:** retrieval is the step that decides which passages the model gets to read —
> and when a RAG answer is wrong, this is usually where it went wrong.

---

## 1. Two ways to search, with opposite weaknesses

| | Keyword search (BM25) | Vector search (dense) |
|---|---|---|
| Matches | exact words | meaning |
| Great at | codes, names, SKUs, rare terms | paraphrases, synonyms, intent |
| Blind to | "forgot my login" ≠ "reset password" | `INV-2024-8835` vs `INV-2024-8836` |
| Needs | an inverted index | an embedding model |

**BM25** scores documents by how often the query's words appear, weighted so that rare words count
more and long documents are not unfairly favoured. It has been the backbone of search for decades
and is still excellent at what it does.

**Dense retrieval** compares embeddings ([02 · Embeddings](../02-embeddings/)). It finds passages
that mean the same thing in different words, and struggles exactly where exactness matters.

---

## 2. Measured, not assumed

[`hybrid_search.py`](examples/hybrid_search.py) tests the usual claim that "hybrid beats both" on a
corpus containing six near-identical invoice numbers, five similar error codes, and prose that
users will paraphrase.

**Recall@1 — was the right document first?**

| Method | Overall | Identifier | Paraphrase | Mixed |
|---|---|---|---|---|
| BM25 (keyword) | 64% | **100%** | **0%** | 100% |
| dense (vectors) | 73% | **40%** | **100%** | 100% |
| hybrid (RRF) | **45%** | 60% | 0% | 100% |
| hybrid + rerank | **82%** | 100% | 50% | 100% |

**Recall@3 — was it anywhere in the top three?**

| Method | Overall | Identifier | Paraphrase | Mixed |
|---|---|---|---|---|
| BM25 (keyword) | 64% | 100% | 0% | 100% |
| dense (vectors) | 82% | 60% | 100% | 100% |
| hybrid (RRF) | 82% | 100% | 50% | 100% |
| **hybrid + rerank** | **100%** | **100%** | **100%** | **100%** |

Four things worth sitting with:

1. **Each single method has a catastrophic column.** BM25 scores **0%** on paraphrases; dense
   scores **40%** on identifiers. Those are not small gaps to paper over — they are whole query
   types the system cannot serve.
2. **Dense retrieval's weakness is specific.** It handled every `ERR-44xx` code, but ranked the
   right *invoice* 5th or 6th: six lines differing by one digit are nearly the same vector.
3. **Naive fusion can make things worse.** Hybrid RRF scores **45%** at recall@1 — below either
   method alone — because fusing a rank-1 result with a rank-14 result produces something in the
   middle. It rescues the catastrophic failures (identifier 60%→100% and paraphrase 0%→50% at @3),
   but it also drags good results down.
4. **Reranking is the reliable win:** 100% at recall@3, 82% at recall@1.

**The honest conclusion is not "always use hybrid."** It is: your corpus decides. Build a labelled
query set, measure **per query type**, and keep what wins. Hybrid earns its place when a real share
of your traffic is exact lookups — codes, SKUs, names — and on large corpora where keyword matching
stays sharp while the vector space gets crowded.

---

## 3. Reciprocal Rank Fusion

How do you combine a BM25 score of 14.2 with a cosine similarity of 0.71? You don't — you combine
the **ranks**:

```
score(document) = sum over systems of  1 / (k + rank in that system)      k = 60 by convention
```

No score normalisation, no weight to tune, and a document ranked highly by *both* systems rises to
the top. If one retriever is clearly stronger on your data, weight it:

```
score = w_dense / (k + rank_dense) + w_bm25 / (k + rank_bm25)
```

and tune the weights on your labelled set. Equal weights are only sensible when both systems are
about equally good — which the table above shows is often untrue.

---

## 4. The retrieval funnel

```
   millions of chunks
        |  cheap:  BM25 and/or vector search  ->  top 20-50
   shortlist
        |  expensive:  a cross-encoder reads the query WITH each candidate
   top 3-5   ->  into the prompt
```

Ordinary vector search uses a **bi-encoder**: the query and document are turned into vectors
*separately*, so it can never notice that one contains the exact code the other asked for. A
**cross-encoder** reads both together and is far more accurate — and far too slow to run over a
whole corpus, which is exactly why it runs over the shortlist.

Off-the-shelf rerankers: **Cohere Rerank**, **BGE reranker**, **Jina reranker**, or any
cross-encoder from `sentence-transformers`. Adding one is usually the single highest-value
improvement available after fixing chunking.

---

## 5. The rest of the toolbox

| Technique | What it does | When it helps |
|---|---|---|
| **Query rewriting** | Turn a follow-up ("what about the second one?") into a standalone question | Essential for chat; retrieval on a bare follow-up is meaningless |
| **Multi-query** | Generate 3–5 phrasings, retrieve for each, fuse | Cheap, reliably helps recall |
| **[HyDE](https://arxiv.org/abs/2212.10496)** | Write a hypothetical *answer*, embed that, search with it | Answers look more like documents than questions do |
| **Metadata filters** | Narrow by date, source, tenant, permissions before searching | Correctness and security, not just quality |
| **Sentence-window** | Embed small units for precision, return the surrounding paragraph | Precision without losing context |
| **Parent-document** | Search child chunks, hand the model the parent section | Same idea, larger granularity |
| **[ColBERT](https://arxiv.org/abs/2004.12832)** | Late interaction: match at token level, not one vector per document | Higher quality than a bi-encoder, cheaper than a cross-encoder |

Add them **one at a time and measure each**. Every one costs latency, and on your data some will do
nothing at all.

---

## 6. Tuning top-k

How many chunks should go in the prompt?

- **Too few** and the answer may not be there.
- **Too many** and you pay for tokens, add distracting material, and hit
  [Lost in the Middle](https://arxiv.org/abs/2307.03172) — models attend less reliably to material
  buried in a long context.

3–5 chunks after reranking is the common landing spot. Retrieve more (20–50) as a shortlist for the
reranker, then cut hard. Measure **context recall** (did the answer make it in?) and **context
precision** (how much of what we sent was useful?) separately — see
[05 · Evaluation](../../05-evaluation/).

---

## 7. Debugging bad answers

When a RAG answer is wrong, check in this order:

1. **Was the right chunk retrieved at all?** If not, nothing downstream can save you. Fix chunking
   or retrieval.
2. **Was it retrieved but ranked below your top-k cutoff?** Add reranking, or raise k.
3. **Was it in the prompt and the model still got it wrong?** Now — and only now — it is a
   generation problem: strengthen the grounding instruction or use a better model.

Most teams spend their time on step 3 when the problem is step 1. **Always print what was
retrieved.**

---

## 8. Examples

```bash
pip install numpy
pip install sentence-transformers      # optional; the tables above used it
python examples/hybrid_search.py
```

Implements BM25, dense retrieval, RRF fusion and a reranker from scratch, then reports recall@1 and
recall@3 by query type plus a per-query rank table showing exactly where each method fails.

---

## 9. Exercises

1. **Read the rank table.** Find the two queries where dense retrieval ranks the answer 5th or 6th.
   What do those documents have in common?
2. **Explain the drop.** Hybrid RRF scores *below* dense alone at recall@1. Using the rank table,
   explain mechanically how fusing rank 1 with rank 14 produces rank 3.
3. **Weight the fusion.** Modify RRF to weight dense at 0.7 and BM25 at 0.3. Does overall recall@1
   beat either method alone now? What does that tell you about equal weights?
4. **Add a query type.** Write three queries of a kind not represented (for example, a question
   requiring a date filter). Which method handles them?
5. **Kill the reranker.** Remove the exact-term bonus from `rerank`. How much of its advantage came
   from that one line, and what does that say about what a real cross-encoder provides?

---

## 10. Projects to build and test

### Beginner — Add BM25 to your RAG
Take `mini_rag.py` from the track root and add keyword search plus RRF fusion.

**How to test it:** build 15 questions, 5 of which contain an exact identifier. Report recall before
and after — and report it per query type, not just overall.

### Intermediate — Add a real reranker
Put a cross-encoder from `sentence-transformers` in front of your top-20.

**How to test it:** measure recall@3 and added latency. Decide, with numbers, whether the latency is
worth it for your use case.

### Advanced — Retrieval evaluation harness
Build a labelled set of (query, correct-chunk) pairs and a script that scores every retrieval
configuration you can assemble.

**How to test it:** publish a table across at least six configurations. The winning combination
should beat the naive baseline substantially — and if it does not, that is a finding worth writing
down too.

---

## 11. Resources

- [BM25 explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-variables) — Elastic. The clearest walkthrough of the formula and its parameters.
- [Hybrid search explained](https://weaviate.io/blog/hybrid-search-explained) — Weaviate, including fusion methods.
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic. Contextual BM25 plus embeddings plus reranking, with measured failure-rate reductions.
- [ColBERT](https://arxiv.org/abs/2004.12832) — late interaction, between bi-encoders and cross-encoders.
- [HyDE](https://arxiv.org/abs/2212.10496) — retrieving with a hypothetical answer.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — why more retrieved chunks is not always better.

---

**Previous:** [03 · Vector Stores](../03-vector-stores/) · **Next:** [05 · Agentic RAG](../05-agentic-rag/)
