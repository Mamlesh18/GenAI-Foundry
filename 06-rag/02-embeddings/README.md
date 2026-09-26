# 02 · Embeddings

> **In one sentence:** an embedding turns a piece of text into a list of numbers positioned so that
> texts with similar meanings end up close together — which lets a computer find "the passage that
> answers this" without matching a single word.

---

## 1. From words to vectors

Start with something you can read. Give every word in the vocabulary a dimension and count:

```
vocabulary: ['cat', 'mat', 'quarterly', 'revenue', 'rose', 'rug', 'sat']

"the cat sat on the mat"    ->  [1 1 0 0 0 0 1]
"a cat sat on a rug"        ->  [1 0 0 0 0 1 1]
"quarterly revenue rose"    ->  [0 0 1 1 1 0 0]
```

A real embedding has 384–3,072 dimensions that mean nothing individually, but the principle is
identical: **text becomes a point in space.**

### Comparing two vectors

**Cosine similarity** measures the angle between them, from −1 to 1:

```
cosine(a, b) = (a · b) / (|a| · |b|)
```

Ignoring length is deliberate — a long document and a short question about the same topic should
still match. And if you **normalise** every vector to length 1 first, cosine similarity is just the
dot product, which is why vector databases store normalised vectors and use a plain multiply.

---

## 2. Why counting words is not enough

From [`embedding_basics.py`](examples/embedding_basics.py), the same pairs scored two ways:

| Pair | Bag of words | Real embeddings |
|---|---|---|
| "a dog chasing a ball" / "a puppy playing fetch" | **0.000** | **0.493** |
| "how do I reset my password" / "I forgot my login credentials" | 0.365 | 0.677 |
| "the meeting is on Tuesday" / "the meeting is **not** on Tuesday" | 0.816 | 0.880 |
| "annual leave policy" / "quarterly revenue report" | 0.000 | 0.271 |

Word counting says a dog chasing a ball and a puppy playing fetch are **completely unrelated**,
because they share no words. A learned embedding scores them 0.49 — it has learned that dogs and
puppies, chasing and fetching, belong together. **That is the entire reason RAG uses embeddings
instead of keyword search.**

Now look at the third row. Both methods score the Tuesday pair highly, and the second sentence means
the opposite of the first. **Embeddings are weak at negation** — a real limitation, not a quirk of
this small model.

---

## 3. Choosing a model

| Option | Dimensions | Notes |
|---|---|---|
| **all-MiniLM-L6-v2** (open) | 384 | Tiny, fast, CPU-friendly. A good baseline and what the examples here use |
| **BGE / E5 / GTE / Nomic** (open) | 768–1024 | Stronger open models; check the licence for commercial use |
| **Qwen3-Embedding** (open) | up to 4096 | Among the strongest open models on public leaderboards |
| **OpenAI text-embedding-3** (API) | 1536 / 3072 | Cheap, easy, solid general quality |
| **Cohere Embed** (API) | 1024+ | Strong multilingual; compression-friendly variants |
| **Voyage** (API) | varies | Domain-specific variants (code, finance, legal) |
| **Gemini Embedding** (API) | varies | Competitive; tops public leaderboards at times |

**How to actually choose:**

1. Start with a small open model. It costs nothing and is usually good enough to build with.
2. Check the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — but weight the
   task types that resemble yours (retrieval, your language, your domain), not the headline average.
3. **Test two or three on your own documents and questions.** Retrieval quality on your data is the
   only benchmark that decides anything.

**Dimensions are a cost, not a score.** A 3,072-dimension vector needs 4× the storage and search
time of a 768-dimension one. Many modern models let you truncate vectors with little quality loss —
check before paying for width you do not need.

---

## 4. Things that will bite you

| Problem | Why | What to do |
|---|---|---|
| **Vectors from different models are incomparable** | The numbers mean different things | Changing model = re-embed everything. Store the model name and version with the vectors |
| **Exact identifiers embed badly** | `ERR-4471` and `ERR-4472` are nearly the same vector | Use keyword search for codes and names — [hybrid search](../04-retrieval/) |
| **Negation is barely represented** | See the Tuesday row above | Do not rely on embeddings for "must not" / "excluding" distinctions |
| **Asymmetry** | A short question and a long passage are different kinds of text | Some models want a `query:` / `passage:` prefix. Read the model card — using it wrong quietly costs recall |
| **Long inputs get truncated** | Every model has a maximum input length | Check it; chunks longer than the limit are silently cut |
| **Multilingual is not free** | Many models are English-first | Pick a multilingual model if your corpus or users are not English |

---

## 5. Where embeddings are used besides RAG

Semantic search, deduplication, clustering, classification, recommendation, anomaly detection, and
picking few-shot examples dynamically ([04 · Prompting Techniques](../../01-foundations/04-prompting-techniques/)).
The vector is a general-purpose handle on meaning.

---

## 6. Examples

```bash
pip install numpy
pip install sentence-transformers      # optional, but section 4 needs it
python examples/embedding_basics.py
```

Builds bag-of-words vectors you can read, computes cosine similarity by hand, shows where lexical
matching fails, then runs the same pairs through a real model for comparison.

---

## 7. Exercises

1. **Read a vector.** Run section 1 and check one vector against the vocabulary by hand. Which
   dimension is which word?
2. **Predict then measure.** Before running section 4, guess the embedding similarity for each pair.
   How close were you? Which one surprised you?
3. **Find the negation failure.** Write three sentence pairs that mean opposite things but share
   most words. How does the model score them? What does that imply for a policy document full of
   "must" and "must not"?
4. **Break it with identifiers.** Embed `ERR-4471` and `ERR-4472` in context and compare. Now do the
   same with two product names differing by one character.
5. **Compare two models.** Run the same pairs through `all-MiniLM-L6-v2` and a larger model. Do the
   rankings change, or only the absolute numbers? Which matters for retrieval?

---

## 8. Projects to build and test

### Beginner — Semantic search over your notes
Embed a folder of markdown files and answer queries with the top 5 passages and scores.

**How to test it:** write 10 questions whose answers you know. Measure recall@5. Then ask something
definitely not covered — does it return low scores, or confidently return junk?

### Intermediate — Embedding model bake-off
Compare three models on your own corpus for retrieval quality, speed and storage.

**How to test it:** build (query, correct-chunk) pairs, report recall@1 and recall@5 per model with
a cost column, and state which you would ship and why.

### Advanced — Dimension reduction study
Take one model and truncate its vectors to 25%, 50% and 75% of full width.

**How to test it:** plot recall against storage. Find the point where quality starts to fall —
usually much further down than people expect, which is a real saving at scale.

---

## 9. Resources

- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) — Jay Alammar. The clearest explanation of why meaning-as-geometry works at all.
- [Sentence-Transformers documentation](https://sbert.net/) — the library used here; excellent docs.
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — how embedding models compare, by task.
- [Best embedding models compared](https://www.buildmvpfast.com/blog/best-embedding-model-comparison-voyage-openai-cohere-2026) — a practical look at the commercial options.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — the paper that made good sentence embeddings practical.
- [Gemini Embedding](https://arxiv.org/abs/2503.07891) — a current example of how these models are built.

**Related in this repo**
- [01-foundations/06 · Embeddings](../../01-foundations/06-embeddings/) — the same idea from the model's side
- [04 · Retrieval](../04-retrieval/) — what to do when embeddings alone are not enough

---

**Previous:** [01 · Chunking](../01-chunking/) · **Next:** [03 · Vector Stores](../03-vector-stores/)
