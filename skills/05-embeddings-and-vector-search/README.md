# 05 · Embeddings & Vector Search

**Level:** Intermediate · **Time:** ~4 hours

---

## 1. What you will be able to do

Take a folder of documents and build search over them that finds the right passage even when the
query shares no words with it — then measure how good that search actually is, and improve it.

---

## 2. Prerequisites

- Skill: [01 · First LLM App](../01-first-llm-app/)
- Concepts: [06 · Embeddings](../../01-foundations/06-embeddings/)
- `pip install sentence-transformers numpy`

---

## 3. The concept, briefly

Embed text into vectors positioned so that semantically similar passages sit close together. Search
becomes: embed the query, find the nearest vectors.

The part nobody warns you about: **retrieval quality is dominated by chunking, not by the embedding
model.** People spend days picking a model off a leaderboard and ten minutes on how they split
their documents, then wonder why results are poor. Invert that.

The other thing worth knowing early: **hybrid search almost always beats pure vector search.**
Embeddings are excellent at meaning and bad at exact tokens — order numbers, error codes, product
SKUs, names. BM25 keyword search is the opposite. Combine them and you get both.

---

## 4. Build it

1. **Load and chunk.** Start with fixed-size chunks (about 500 tokens, 15% overlap). Keep metadata
   with every chunk: source file, position, section heading. You need it for citations later.

2. **Embed.** Use `sentence-transformers` with `all-MiniLM-L6-v2` — small, fast, runs on a laptop,
   good enough to learn on. Set `normalize_embeddings=True`.

3. **Search with NumPy.** Under ~100K chunks you do not need a vector database. A matrix multiply
   is genuinely fine, and doing it manually makes clear that a vector DB is an index, not magic.

4. **Build an evaluation set.** 20 questions, each with the chunk id that actually answers it. This
   is the work everyone skips and everyone regrets skipping. Without it, every later step is guesswork.

5. **Measure the baseline.** recall@1, recall@5, MRR. Write the numbers down.

6. **Sweep chunk size.** 200, 500, 1000 tokens. Re-embed, re-measure. The curve is usually not what
   you expect and is specific to your documents.

7. **Sweep overlap.** 0%, 10%, 25%. Construct at least one question whose answer straddles a
   boundary and check which settings find it.

8. **Try semantic chunking.** Split on headings and paragraphs instead of a character count.
   Re-measure. On structured documents this often beats every fixed-size configuration.

9. **Add BM25.** `rank_bm25` gives you keyword search in a few lines. Measure it alone.

10. **Fuse them.** Reciprocal Rank Fusion: `score = Σ 1/(k + rank_i)` with k≈60. Measure the hybrid.
    It should beat both — confirm that on your data rather than taking my word for it.

11. **Add a reranker.** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores your top 20 down to the
    best 5. Slower, usually a solid accuracy gain. Decide whether the latency is worth it.

---

## 5. Checkpoint

- [ ] A table comparing at least six configurations on recall@5 for the same question set
- [ ] Your best configuration beats the naive baseline by a margin you can state
- [ ] You can explain *why* your winning chunk size won, in terms of your documents
- [ ] Hybrid search beats vector-only on a query containing an exact identifier
- [ ] A query about something genuinely absent returns low scores rather than confident junk
- [ ] Retrieval over 10K chunks returns in under a second
- [ ] Every result carries the metadata needed to cite it

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| No evaluation set | You cannot tell whether changes help | Build 20 labelled pairs first |
| Embedding whole documents | One vector meaning everything and nothing | Chunk |
| Zero overlap | Answers straddling a boundary are unfindable | 10–20% overlap |
| Chunks longer than the model's max input | Silent truncation; you never see an error | Check the model's limit |
| Dropping metadata | You retrieve a good passage and cannot cite it | Store it alongside every chunk |
| Vector-only search | Exact codes and names are unfindable | Hybrid |
| Comparing vectors across models | Meaningless numbers | Re-embed everything when you switch |
| Reaching for a vector DB at 500 documents | Infrastructure you do not need | NumPy until it hurts |
| Forgetting to normalise | Cosine and dot product disagree | `normalize_embeddings=True` |

---

## 7. Going further

- **Next skill:** [06 · RAG Pipeline](../06-rag-pipeline/) — retrieval is half of it; you have the half
- Try query expansion: rewrite the user's question into several before retrieving
- Try HyDE: generate a hypothetical answer, embed *that*, and search with it
- Compare 3 embedding models on your own data instead of trusting the leaderboard average

**Links:** [Sentence-Transformers](https://sbert.net/) ·
[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ·
[rank_bm25](https://github.com/dorianbrown/rank_bm25) ·
[Chroma](https://www.trychroma.com/)
