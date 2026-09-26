# 03 · Vector Stores

> **In one sentence:** a vector store keeps your embeddings, and finds the nearest ones to a query
> fast — over millions of vectors, with metadata filters, without loading everything into memory.

This module covers what they actually do, and **which provider to choose**.

---

## 1. What a vector store holds

```
  id     vector (384-3072 floats)          text                      metadata
  ----   ------------------------------    ----------------------    --------------------------
  c_001  [0.021, -0.114, 0.339, ...]       "Annual leave is 24..."   {source: handbook/leave.md,
                                                                       section: Leave,
                                                                       updated: 2026-03-01,
                                                                       tenant: acme}
```

Four things, and the last two matter more than beginners expect: you need the **text** to put in
the prompt, and the **metadata** to cite, filter and enforce permissions.

---

## 2. Search: exact, then approximate

**Brute force** compares the query against every vector. It is exact, trivial to implement, and
perfectly fine up to roughly 100,000 vectors — in
[`vector_index_demo.py`](examples/vector_index_demo.py) it searches 50,000 vectors in about 9 ms.

Past that, you use an **approximate nearest neighbour (ANN)** index, which trades a little recall
for a lot of speed. The script builds an IVF index (cluster the vectors, then search only the
nearest clusters):

| Clusters searched | Time/query | Speed-up | Recall@10 |
|---|---|---|---|
| 1 | 2.3 ms | 5.1× | 66% |
| 2 | 3.3 ms | 3.5× | 94% |
| 4 | 4.7 ms | 2.5× | **100%** |
| 16 | 17.5 ms | 0.7× | 100% |
| 32 | 34.2 ms | 0.3× | 100% |

Two things to take from that table. **Approximate really is approximate** — searching one cluster
misses a third of the true neighbours. And **more probing is not free**: past a point you are
searching most of the corpus again, with index overhead on top, and it becomes *slower* than brute
force. Every vector database exposes this dial (`nprobe` in FAISS, `ef_search` in HNSW).

### The index types you will meet

| Index | How it works | Trade-off |
|---|---|---|
| **Flat** | brute force, no index | exact, slow at scale |
| **IVF** | cluster, search nearest clusters | fast, tunable recall, needs training |
| **HNSW** | navigable small-world graph | the common default; fast and accurate, memory-hungry |
| **PQ / quantized** | compress vectors to fewer bytes | big memory savings, some accuracy loss |
| **DiskANN / disk-based** | keep most of the index on SSD | far cheaper for huge corpora, higher latency |

---

## 3. Metadata filtering — the feature that decides real projects

"Only documents from 2026", "only this customer's files", "only what this user may see." The order
of operations matters:

| Approach | What happens |
|---|---|
| **Post-filter** (search, then filter) | Simple, and silently returns fewer results than you asked for — sometimes none |
| **Pre-filter** (filter, then search) | Correct, but the index has to support it efficiently |

In the demo, a post-filter asked for 10 results and returned **3**. When you compare vector
databases, filtered-search quality is one of the genuine differentiators — and for multi-tenant or
permissioned data, it is a correctness requirement, not an optimisation.

---

## 4. What it costs at scale

Vectors alone, before the index, the text or the metadata:

| Vectors | 384 dims (fp32) | 1536 dims (fp32) | 1536 dims (int8) |
|---|---|---|---|
| 100,000 | 0.2 GB | 0.6 GB | 0.2 GB |
| 1,000,000 | 1.5 GB | 6.1 GB | 1.5 GB |
| 10,000,000 | 15.4 GB | 61.4 GB | 15.4 GB |
| 100,000,000 | 153.6 GB | 614.4 GB | 153.6 GB |

Smaller embedding dimensions are not a detail, and quantizing vectors is how large deployments stay
affordable. This table is also the honest reason managed services exist: at 10M+ vectors, memory,
sharding, replication and filtered search stop being something most teams want to run themselves.

---

## 5. The providers

### Libraries — run inside your process, no server

| Tool | Notes |
|---|---|
| **numpy** | Seriously. Under ~100k vectors a matrix multiply is enough, and you keep full control |
| **[FAISS](https://github.com/facebookresearch/faiss)** | Meta's library; the reference implementation for IVF, HNSW and PQ. No persistence layer or filtering to speak of |
| **[Chroma](https://www.trychroma.com/)** | Embedded, minimal setup, great for prototypes and small production |
| **[LanceDB](https://lancedb.com/)** | Embedded, built on the open Lance columnar format; strong for multimodal data |
| **[pgvector](https://github.com/pgvector/pgvector)** | A Postgres extension (HNSW and IVFFlat). If you already run Postgres, start here |

### Dedicated vector databases — you run them, or pay someone to

| Product | Best for |
|---|---|
| **[Qdrant](https://qdrant.tech/)** | Fast, open source, good filtering; popular self-hosted default |
| **[Weaviate](https://weaviate.io/)** | Built-in hybrid search and embedding modules |
| **[Milvus](https://milvus.io/)** / **[Zilliz Cloud](https://zilliz.com/)** | Billions of vectors; powerful, needs engineering attention |
| **[Vespa](https://vespa.ai/)** | Search engine plus ranking and ML inference; heavyweight and very capable |
| **[Pinecone](https://www.pinecone.io/)** | Fully managed, zero ops; the easy answer if you would rather not operate one |
| **[turbopuffer](https://turbopuffer.com/)** | Built on object storage; aimed at large corpora at low cost |

### Search engines and databases that added vectors

| Product | Why you'd choose it |
|---|---|
| **Elasticsearch / OpenSearch** | You already run it, and you want BM25 and vectors together |
| **Redis** | Already in your stack, low-latency, in-memory |
| **MongoDB Atlas Vector Search** | Your documents already live in Mongo |
| **Azure AI Search / Vertex AI Vector Search / Amazon OpenSearch** | You are committed to that cloud |

### How to choose

```
Under ~100k vectors?              -> numpy or FAISS, in your own process
Already running Postgres?         -> pgvector (one database, SQL filters, transactions)
Prototyping?                      -> Chroma or LanceDB
Self-hosting at scale?            -> Qdrant, Weaviate, or Milvus
Want zero operations?             -> Pinecone, Zilliz Cloud, or a managed Qdrant/Weaviate
Need BM25 and vectors together?   -> Weaviate, Vespa, or Elasticsearch
```

**An honest note:** for most projects this choice matters far less than chunking and retrieval
quality. Teams routinely spend a week comparing vector databases and then lose more recall to a bad
chunk size than any of them would have cost. Pick the one that fits your stack, and spend the
saved time on [01 · Chunking](../01-chunking/) and [04 · Retrieval](../04-retrieval/).

---

## 6. Questions to ask before committing

- **Filtered search:** pre-filter or post-filter? How does recall behave with a narrow filter?
- **Updates and deletes:** can you update a document in place, or must you rebuild?
- **Hybrid search:** is BM25 built in, or will you run a second system?
- **Multi-tenancy:** can one index serve many customers with hard isolation?
- **Persistence and backup:** what happens when the process restarts?
- **Cost model:** priced per vector, per query, per GB, or per hour? Model it at 10× your current size.
- **Migration:** if you change embedding model you must re-embed everything — how painful is a full reindex?

---

## 7. Examples

```bash
pip install numpy
python examples/vector_index_demo.py       # a few seconds
```

Builds a 50,000-vector corpus, searches it brute force, builds an IVF index from scratch, measures
the recall/speed trade-off, demonstrates pre- vs post-filtering, and prints the memory table above.

---

## 8. Exercises

1. **Find your crossover.** Change `N_VECTORS` to 5,000 and 500,000. At what size does brute force
   stop being acceptable for a 100 ms latency budget?
2. **Tune the dial.** For a system needing 95% recall, which `n_probe` would you choose from the
   table, and what does it cost in latency?
3. **Break the filter.** Make the metadata filter much narrower (say 1% of documents). What does
   post-filtering return now? Rewrite it as a pre-filter.
4. **Do the memory maths.** 5 million chunks, 1536-dimension vectors, fp32. How much RAM for the
   vectors? What if you switch to a 768-dimension model? To int8?
5. **Pick a provider.** For each: a personal notes app (10k chunks); a startup's support bot (2M
   chunks, small team); a bank's document search (500M chunks, strict isolation). Which store, and
   what would you ask the vendor first?

---

## 9. Projects to build and test

### Beginner — Swap the store
Take `mini_rag.py` from the track root and move its storage to Chroma or pgvector.

**How to test it:** identical answers to the numpy version for the same questions, plus persistence
— restart the process and confirm you do not re-embed.

### Intermediate — Implement HNSW
Build a small hierarchical navigable small-world index yourself.

**How to test it:** measure recall@10 against brute force and plot it against `ef_search`. You
should reproduce the same shape of curve as the IVF table above.

### Advanced — Benchmark three stores on your data
Index the same corpus in three stores and measure build time, query latency (p50/p95), recall,
filtered-search behaviour and memory.

**How to test it:** publish the table with your corpus size and filter selectivity stated. A
benchmark without those two numbers is not reproducible.

---

## 10. Resources

- [Best Vector Databases: comparison guide](https://www.firecrawl.dev/blog/best-vector-databases) — a current, practical comparison.
- [Vector Database Comparison: Pinecone vs Weaviate vs Qdrant vs pgvector](https://weekonelabs.com/blog/vector-database-comparison-2026) — opinionated, with the trade-offs stated plainly.
- [FAISS](https://github.com/facebookresearch/faiss) — the library most of the concepts here come from.
- [pgvector](https://github.com/pgvector/pgvector) — HNSW and IVFFlat inside Postgres.
- [Efficient and robust approximate nearest neighbor search using HNSW](https://arxiv.org/abs/1603.09320) — the HNSW paper, and a readable one.

---

**Previous:** [02 · Embeddings](../02-embeddings/) · **Next:** [04 · Retrieval](../04-retrieval/)
