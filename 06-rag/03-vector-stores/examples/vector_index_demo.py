"""
What a vector database actually does -- built small enough to read.

Once you have vectors, you need to find the nearest ones to a query, fast,
over millions of them, with filters. That is the entire job of a vector store.

This script shows the three things every one of them does:

  1. BRUTE FORCE search -- exact, simple, and too slow at scale
  2. An APPROXIMATE index (IVF: cluster, then search a few clusters)
     -- the speed/recall trade-off that all of them make
  3. METADATA FILTERING -- and why "filter then search" vs "search then filter"
     is a question you will have to answer

    pip install numpy
    python vector_index_demo.py
"""

import time

import numpy as np

rng = np.random.default_rng(0)

N_VECTORS = 50_000
DIMENSIONS = 384          # the size of all-MiniLM-L6-v2 vectors
N_CLUSTERS = 100
TOP_K = 10


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalise(matrix):
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Data: real embeddings are clustered by topic, not uniformly random
# ---------------------------------------------------------------------------

def make_corpus():
    topic_centres = normalise(rng.normal(size=(40, DIMENSIONS)))
    topics = rng.integers(0, 40, size=N_VECTORS)
    vectors = topic_centres[topics] + 0.05 * rng.normal(size=(N_VECTORS, DIMENSIONS))
    # Metadata, exactly as you would store beside each vector
    metadata = [{"id": i, "topic": int(topics[i]),
                 "year": int(rng.integers(2020, 2027)),
                 "source": f"doc_{i // 10}.md"} for i in range(N_VECTORS)]
    return normalise(vectors), metadata


# ---------------------------------------------------------------------------
# 1. Brute force
# ---------------------------------------------------------------------------

def brute_force(vectors, query, k=TOP_K):
    scores = vectors @ query            # unit vectors -> dot product IS cosine
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


# ---------------------------------------------------------------------------
# 2. An approximate index, IVF style
# ---------------------------------------------------------------------------

class IVFIndex:
    """Cluster the vectors once; at query time search only the nearest clusters.

    This is what FAISS's IVF does, and the same bargain HNSW makes differently:
    give up a little recall for a large speed-up.
    """

    def __init__(self, vectors, n_clusters=N_CLUSTERS, iterations=8):
        self.vectors = vectors
        sample = vectors[rng.choice(len(vectors), size=min(5000, len(vectors)), replace=False)]
        centroids = sample[rng.choice(len(sample), size=n_clusters, replace=False)].copy()

        for _ in range(iterations):                      # plain k-means
            assignments = np.argmax(sample @ centroids.T, axis=1)
            for c in range(n_clusters):
                members = sample[assignments == c]
                if len(members):
                    centroids[c] = members.mean(axis=0)
            centroids = normalise(centroids)

        self.centroids = centroids
        owner = np.argmax(vectors @ centroids.T, axis=1)
        self.lists = [np.where(owner == c)[0] for c in range(n_clusters)]

    def search(self, query, k=TOP_K, n_probe=4):
        nearest_clusters = np.argsort(-(self.centroids @ query))[:n_probe]
        candidates = np.concatenate([self.lists[c] for c in nearest_clusters])
        if len(candidates) == 0:
            return np.array([], dtype=int)
        scores = self.vectors[candidates] @ query
        top = np.argsort(-scores)[:k]
        return candidates[top]


def recall(approximate, exact):
    return len(set(approximate.tolist()) & set(exact.tolist())) / len(exact)


# ---------------------------------------------------------------------------

def main():
    print(f"  building a corpus of {N_VECTORS:,} vectors x {DIMENSIONS} dimensions...")
    vectors, metadata = make_corpus()
    queries = normalise(vectors[rng.choice(N_VECTORS, size=50, replace=False)]
                        + 0.03 * rng.normal(size=(50, DIMENSIONS)))

    section("1. BRUTE FORCE -- exact, and the baseline for everything else")
    start = time.perf_counter()
    exact_results = [brute_force(vectors, q) for q in queries]
    brute_ms = (time.perf_counter() - start) / len(queries) * 1000
    print(f"  compares the query against all {N_VECTORS:,} vectors, every time")
    print(f"  time per query : {brute_ms:.1f} ms")
    print(f"  recall         : 100% by definition -- this IS the right answer")
    print(
        "\n  Perfectly fine up to roughly 100,000 vectors. Beyond that the cost grows\n"
        "  linearly with your corpus, and every query pays it."
    )

    section("2. AN APPROXIMATE INDEX -- trading recall for speed")
    start = time.perf_counter()
    index = IVFIndex(vectors)
    build_s = time.perf_counter() - start
    print(f"  built {N_CLUSTERS} clusters in {build_s:.1f}s (done once, not per query)\n")
    print(f"  {'clusters searched':>18} {'time/query':>12} {'speed-up':>10} {'recall@10':>11}")
    print("  " + "-" * 56)
    for n_probe in (1, 2, 4, 8, 16, 32):
        start = time.perf_counter()
        approx = [index.search(q, n_probe=n_probe) for q in queries]
        ms = (time.perf_counter() - start) / len(queries) * 1000
        mean_recall = sum(recall(a, e) for a, e in zip(approx, exact_results)) / len(queries)
        print(f"  {n_probe:>18} {ms:>11.1f}ms {brute_ms / ms:>9.1f}x {mean_recall:>10.0%}")

    print(
        "\n  Searching 1 cluster out of 100 is fastest and misses a third of the true\n"
        "  neighbours. A few clusters recover almost all of it while still being\n"
        "  several times faster than brute force. Push it far enough and you are\n"
        "  searching most of the corpus again -- with the index overhead on top, so it\n"
        "  ends up SLOWER than brute force. The sweet spot is a handful of clusters.\n\n"
        "  That dial -- nprobe in FAISS, ef_search in HNSW -- exists in every vector\n"
        "  database, and setting it is choosing how much recall you will give up.\n"
        "  (This corpus is only 50,000 vectors, so brute force is still quick. The\n"
        "  bigger the corpus, the further right that sweet spot moves.)\n\n"
        "  Two consequences people forget:\n"
        "    - approximate means APPROXIMATE. Your search can miss the best match.\n"
        "    - building the index costs time and must be redone as data grows."
    )

    section("3. METADATA FILTERING -- 'only 2026 documents'")
    query = queries[0]
    wanted_year = 2026

    exact_top = brute_force(vectors, query, k=TOP_K)
    post_filtered = [i for i in exact_top if metadata[i]["year"] == wanted_year]

    mask = np.array([m["year"] == wanted_year for m in metadata])
    subset = np.where(mask)[0]
    scores = vectors[subset] @ query
    pre_filtered = subset[np.argsort(-scores)[:TOP_K]]

    print(f"  corpus                       : {N_VECTORS:,} vectors")
    print(f"  match the filter (year=2026) : {mask.sum():,}\n")
    print(f"  search first, then filter -> {len(post_filtered)} results (asked for {TOP_K})")
    print(f"  filter first, then search -> {len(pre_filtered)} results")
    print(
        "\n  Searching first and filtering afterwards is easy and silently returns\n"
        "  fewer results than asked for -- sometimes none, if the filter is narrow.\n"
        "  Filtering first is correct but needs the index to support it, which is\n"
        "  exactly what vector databases spend engineering effort on. When you compare\n"
        "  products, 'filtered search' quality is one of the real differentiators."
    )

    section("4. WHAT THIS COSTS AT SCALE")
    print(f"  {'vectors':>12} {'384 dims':>12} {'1536 dims':>12} {'1536 dims, int8':>18}")
    print("  " + "-" * 60)
    for n in (100_000, 1_000_000, 10_000_000, 100_000_000):
        small = n * 384 * 4 / 1e9
        large = n * 1536 * 4 / 1e9
        quantized = n * 1536 * 1 / 1e9
        print(f"  {n:>12,} {small:>10.1f} GB {large:>10.1f} GB {quantized:>16.1f} GB")
    print(
        "\n  Vectors alone, before the index structure, the original text, or metadata.\n"
        "  Two lessons: smaller embedding dimensions are not a detail, and quantizing\n"
        "  vectors (int8, binary) is how large deployments stay affordable.\n\n"
        "  This table is the honest reason managed vector databases exist: at 10M+\n"
        "  vectors, memory, sharding, replication and filtered search stop being\n"
        "  something you want to maintain yourself."
    )

    section("SO WHICH STORE SHOULD I USE?")
    print(
        "  Under ~100k vectors     numpy or FAISS in your own process. Really.\n"
        "  Already using Postgres  pgvector -- SQL filters, joins, one database\n"
        "  Prototyping             Chroma -- runs in-process, minimal setup\n"
        "  Self-hosted at scale    Qdrant, Milvus, Weaviate\n"
        "  Managed, no ops         Pinecone, Zilliz, Weaviate Cloud\n"
        "  Already using Elastic   its own vector search, alongside BM25\n\n"
        "  See the README for the full comparison. The honest summary: for most\n"
        "  projects this choice matters far less than chunking and retrieval quality."
    )


if __name__ == "__main__":
    main()
