# 06 · Embeddings

Meaning as geometry. Embeddings turn text into vectors positioned so that *similar things sit close
together* — which turns "find me something relevant" into "find me a nearby point", a problem
computers are extremely good at.

---

## 1. What this is

An embedding is a dense vector of floating-point numbers (typically 384–3072 dimensions)
representing a piece of text.

```
"a dog chasing a ball"   -> [ 0.021, -0.114,  0.339, ... ]
"a puppy playing fetch"  -> [ 0.019, -0.108,  0.341, ... ]   <- nearly the same direction
"quarterly tax filing"   -> [-0.402,  0.277, -0.089, ... ]   <- somewhere else entirely
```

Crucially, closeness is **semantic**, not lexical. "a puppy playing fetch" shares almost no words
with "a dog chasing a ball", but lands right next to it. Keyword search cannot do that.

### Measuring closeness

**Cosine similarity** — the angle between two vectors — is the standard measure:

```
cos(a, b) = (a · b) / (|a| · |b|)      ranges from -1 to 1
```

It ignores magnitude and compares direction only, which is what you want: a long document and a
short query about the same topic should match.

> Most modern embedding APIs return **normalised** vectors (length 1). When they do, cosine
> similarity and dot product are identical, and dot product is faster. Check your provider's docs.

### Token embeddings vs sentence embeddings

Two different things share the name:

- **Token embeddings** — the lookup table inside a transformer, one vector per vocabulary token,
  learned during pretraining. This is what feeds into [attention](../01-attention-is-all-you-need/).
- **Sentence/document embeddings** — one vector for a whole passage, produced by a dedicated
  embedding model. This is what powers search and RAG.

When someone says "embeddings" in an applied context, they almost always mean the second.

### The famous arithmetic

`king - man + woman ≈ queen`

From word2vec, and still the best intuition pump for why this works: directions in the space encode
relationships. Modern contextual embeddings are far more capable — the same word gets different
vectors in "river bank" and "bank loan" — but the geometric intuition carries over.

---

## 2. Why it matters

Embeddings are the retrieval half of [RAG](../../06-rag/), and RAG is how you give a model
knowledge it was never trained on. Beyond that: semantic search, deduplication, clustering,
classification, recommendation, anomaly detection, and dynamic few-shot example selection.

They are also cheap. Embedding a document costs a small fraction of generating text about it.

---

## 3. Core practicalities

### Chunking

You embed *chunks*, not whole documents — a 50-page PDF averaged into one vector means everything
and therefore nothing.

- Typical chunk: 200–800 tokens, with 10–20% overlap so ideas straddling a boundary survive.
- Split on semantic boundaries (headings, paragraphs) rather than a fixed character count where you
  can.
- Keep metadata (source, page, section) attached — you need it for citations.

Chunking strategy affects retrieval quality more than the choice of embedding model. Most people
get this backwards.

### Choosing a model

| Consideration | Guidance |
|---|---|
| **Dimensions** | Higher is not automatically better; it costs storage and search time. |
| **Max input** | Chunks longer than this are silently truncated — a common, invisible bug. |
| **Domain** | A general model may be weak on legal, medical or code. Check the benchmark for *your* domain. |
| **Symmetric vs asymmetric** | Query→document search often needs a model trained for it, sometimes with a required prefix like `"query: "`. |
| **Self-host vs API** | `sentence-transformers` runs locally and free; APIs are easier and usually stronger. |

Check the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — but weight the tasks
that resemble yours, not the average.

### Vector search

Comparing a query against 10 million vectors one at a time is too slow, so vector databases use
**approximate nearest neighbour** indexes (HNSW, IVF), trading a little recall for enormous speed.
Options: FAISS, Chroma, Qdrant, pgvector, Pinecone, Weaviate. For under ~100K vectors, plain NumPy
is genuinely fine — do not reach for infrastructure you do not need.

### Where embeddings fail

- **Negation.** "The drug is effective" and "The drug is not effective" embed close together.
- **Exact identifiers.** Order numbers, error codes, SKUs. Use keyword search for those — or
  better, **hybrid search** (BM25 + vectors), which reliably beats either alone.
- **Numeric comparison.** "Under 500 rupees" is not a geometric relationship.
- **Cross-model comparison.** Vectors from two different models are meaningless to compare. Change
  your embedding model and you must re-embed everything.

---

## 4. Examples

```bash
pip install sentence-transformers numpy
```

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # small, fast, runs on a laptop

docs = [
    "The cat sat on the mat.",
    "A feline rested upon the rug.",
    "Quarterly revenue rose 12% year on year.",
    "Our Q3 earnings beat expectations.",
]
vectors = model.encode(docs, normalize_embeddings=True)

# Normalised vectors -> dot product IS cosine similarity
similarity = vectors @ vectors.T
print(np.round(similarity, 2))
```

You should see two clear blocks: sentences 0–1 close to each other, 2–3 close to each other, and
near-zero similarity across the blocks. That is semantic clustering, with no keywords in common.

---

## 5. Exercises

1. **Confirm the clustering.** Run the snippet above. Then add a sentence that is lexically similar
   but semantically unrelated ("The cat sat on the board of directors"). Where does it land?
2. **Break it with negation.** Embed "this movie was great" and "this movie was not great". Report
   the similarity. Would your retrieval system tell them apart?
3. **Chunk size sweep.** Take a long article. Chunk at 100, 400 and 1000 tokens. Ask the same five
   questions of each and judge which retrieves better. There is no universally right answer —
   find yours.
4. **Overlap matters.** Chunk with 0% and 20% overlap. Construct a question whose answer straddles
   a boundary. Which setup finds it?
5. **Hybrid beats both.** Build keyword-only (BM25), vector-only, and combined retrieval. Test with
   a query containing an exact product code *and* a fuzzy description. Which wins, and why?
6. **Reproduce the arithmetic.** With word-level vectors, compute `king - man + woman` and find the
   nearest neighbour. Then try it with a modern sentence-embedding model. Explain the difference.

---

## 6. Projects to build and test

### Beginner — Semantic search over your own notes
Embed a folder of markdown files, then answer natural-language queries with the top 5 chunks and
their similarity scores.

**How to test it:** write 10 questions whose answers you know are in the notes. Measure recall@5.
Then ask a question that is definitely *not* covered — does it return low scores, or confidently
return junk?

### Beginner — Duplicate detector
Given a list of support tickets or product descriptions, cluster near-duplicates above a similarity
threshold.

**How to test it:** plant known duplicates with rewritten wording. Tune the threshold and record
precision/recall. Notice how sensitive the result is to that one number.

### Intermediate — Embedding model bake-off
Compare 3–4 embedding models on *your* data for retrieval quality, speed and storage cost.

**How to test it:** build a small labelled set of (query, correct-chunk) pairs. Report recall@1 and
recall@5 per model with a cost column. Publish the table.

### Advanced — Hybrid search engine
BM25 plus vector search, fused with Reciprocal Rank Fusion, then reranked with a cross-encoder.

**How to test it:** evaluate all four configurations (keyword, vector, hybrid, hybrid+rerank) on
the same query set. Each stage should improve recall — quantify by how much, and decide whether the
reranker's latency is worth its gain. Leads straight into [06 · RAG](../../06-rag/).

---

## 7. Resources

- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) — Jay Alammar. The clearest introduction to why any of this works.
- [Sentence-Transformers documentation](https://sbert.net/) — the practical library; excellent docs.
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — how embedding models actually compare.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — the paper that made good sentence embeddings practical.
- [Efficient Estimation of Word Representations (word2vec)](https://arxiv.org/abs/1301.3781) — where the geometry idea started.
- [FAISS](https://github.com/facebookresearch/faiss) · [Chroma](https://www.trychroma.com/) · [pgvector](https://github.com/pgvector/pgvector) — vector search, from raw to batteries-included.
- [Embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings) — Gemini's embedding API, free tier included.

---

**Previous:** [05 · Tokenization](../05-tokenization/) ·
**Next:** [07 · Transformer Architecture](../07-transformer-architecture/)
