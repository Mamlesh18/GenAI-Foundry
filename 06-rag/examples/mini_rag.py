"""
A complete RAG system in one file -- chunk, embed, store, retrieve, answer.

RAG (Retrieval-Augmented Generation) is often described as if it were an
architecture. It is not. It is four steps:

    1. CHUNK     cut your documents into pieces
    2. EMBED     turn each piece into a vector
    3. RETRIEVE  for a question, find the nearest pieces
    4. GENERATE  put those pieces in the prompt and tell the model to answer
                 ONLY from them

That is the whole idea. Everything else in this track -- better chunking, hybrid
search, reranking, agents -- is an improvement to one of those four steps.

This file does all four against a small fictional company handbook, prints the
actual prompt it builds, and shows the two behaviours that matter: answers with
citations, and refusing to answer when the documents do not say.

    pip install numpy                       (sentence-transformers optional)
    python mini_rag.py
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# The knowledge base: documents RAG will ground its answers in
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "handbook/leave.md": """
Annual leave policy. Every full-time employee receives 24 days of paid annual
leave per year. Leave accrues monthly and unused days may be carried into the
next year, up to a maximum of 5 days. Requests must be approved by your manager
at least two weeks in advance. Sick leave is separate and does not count against
annual leave.
""",
    "handbook/expenses.md": """
Expense policy. Travel expenses must be submitted within 30 days of the trip.
Meals are reimbursed up to 40 pounds per day. Any single expense above 250
pounds requires prior written approval from a director. Receipts are required
for every claim, without exception.
""",
    "handbook/equipment.md": """
Equipment policy. New joiners receive a laptop and a monitor. Hardware refresh
happens every three years. Requests for additional equipment go through the IT
portal and are approved by your manager. Lost or damaged equipment must be
reported to IT within 24 hours.
""",
    "handbook/remote.md": """
Remote work policy. Employees may work remotely up to three days per week.
Fully remote arrangements require director approval and are reviewed every six
months. Core collaboration hours are 10:00 to 16:00 in your local time zone.
""",
}


# ---------------------------------------------------------------------------
# Step 1: CHUNK
# ---------------------------------------------------------------------------

def chunk_document(text, source, chunk_words=40, overlap_words=10):
    """Split into overlapping windows of words.

    Overlap matters: without it, an answer that straddles a boundary is in no
    single chunk, and retrieval can never return it whole. See ../01-chunking/.
    """
    words = text.split()
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, max(1, len(words)), step):
        window = words[start:start + chunk_words]
        if not window:
            break
        chunks.append({
            "text": " ".join(window),
            "source": source,           # keep metadata -- you need it to cite
            "position": len(chunks),
        })
        if start + chunk_words >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# Step 2: EMBED
# ---------------------------------------------------------------------------

class Embedder:
    """Uses a real embedding model when available, and a simple word-overlap
    vector otherwise, so this file runs anywhere."""

    def __init__(self):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.kind = "sentence-transformers / all-MiniLM-L6-v2 (real embeddings)"
        except Exception:
            self.kind = "bag-of-words fallback (no model available)"
            self.vocabulary = {}

    def fit(self, texts):
        if self.model is None:
            words = {w for t in texts for w in tokenize(t)}
            self.vocabulary = {w: i for i, w in enumerate(sorted(words))}

    def encode(self, texts):
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True)
        vectors = np.zeros((len(texts), max(1, len(self.vocabulary))))
        for row, text in enumerate(texts):
            for word in tokenize(text):
                if word in self.vocabulary:
                    vectors[row, self.vocabulary[word]] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


# Interrogatives and filler are removed so that "how many days of annual leave"
# reduces to the words that actually carry the question: day, annual, leave.
STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "on",
             "at", "by", "or", "be", "may", "must", "your", "per", "up", "from",
             "what", "when", "where", "which", "who", "how", "many", "much", "often",
             "do", "does", "did", "i", "my", "me", "can", "get", "there", "company"}


def stem(word):
    """Crude suffix stripping so 'refreshed' matches 'refresh'."""
    for suffix in ("ing", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# Step 3: RETRIEVE  (a "vector store" in its simplest possible form)
# ---------------------------------------------------------------------------

class VectorStore:
    """A matrix of vectors plus the chunks they came from. A real vector
    database adds persistence, filtering and an approximate index -- see
    ../03-vector-stores/ -- but this is the core."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.vectors = None

    def add(self, chunks):
        self.chunks.extend(chunks)
        texts = [c["text"] for c in self.chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)

    def search(self, query, k=3):
        query_vector = self.embedder.encode([query])[0]
        scores = self.vectors @ query_vector          # cosine: vectors are normalised
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]


# ---------------------------------------------------------------------------
# Step 4: GENERATE
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: NOT_FOUND.
Cite the source number for every claim.

<context>
{context}
</context>

Question: {question}
Answer:"""


def build_prompt(question, retrieved):
    context = "\n".join(
        f"[{i + 1}] ({chunk['source']}) {chunk['text']}"
        for i, (chunk, _) in enumerate(retrieved)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def mock_llm(question, retrieved, relevance_floor=0.15, min_shared_words=2):
    """Stands in for a real model call.

    It does what a well-prompted LLM does: answer from the context, cite the
    source, and REFUSE when the context does not actually support an answer.

    Its rule for "supported" is deliberately crude -- the best sentence must
    share at least `min_shared_words` content words with the question. A real
    model judges sufficiency far better than this. What matters here is that
    the refusal path exists and is exercised: notice that the retriever still
    returns the leave policy for a question about PARENTAL leave, because it is
    the nearest text available. Nearest is not the same as sufficient, and that
    gap is why the prompt must permit NOT_FOUND.
    """
    if not retrieved or retrieved[0][1] < relevance_floor:
        return "NOT_FOUND"

    question_words = set(tokenize(question))
    best_sentence, best_index, best_overlap = None, 0, 0
    for index, (chunk, _) in enumerate(retrieved):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            overlap = len(question_words & set(tokenize(sentence)))
            if overlap > best_overlap:
                best_sentence, best_index, best_overlap = sentence, index, overlap

    if best_sentence is None or best_overlap < min_shared_words:
        return "NOT_FOUND"
    return f"{best_sentence.strip()} [{best_index + 1}]"


# ---------------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    embedder = Embedder()
    store = VectorStore(embedder)

    section("BUILDING THE INDEX  (steps 1 and 2, done once, offline)")
    all_chunks = []
    for source, text in DOCUMENTS.items():
        all_chunks.extend(chunk_document(text, source))
    store.add(all_chunks)
    print(f"  documents : {len(DOCUMENTS)}")
    print(f"  chunks    : {len(all_chunks)}  (40 words each, 10 words of overlap)")
    print(f"  embedder  : {embedder.kind}")
    print(f"  vectors   : {store.vectors.shape}  (chunks x dimensions)")
    print(
        "\n  This part runs when documents change, not when a question arrives.\n"
        "  Everything below happens per question, in milliseconds."
    )

    section("ANSWERING A QUESTION  (steps 3 and 4)")
    question = "How many days of annual leave do I get?"
    retrieved = store.search(question, k=3)

    print(f"  question: {question}\n")
    print("  retrieved chunks, best first:")
    for rank, (chunk, score) in enumerate(retrieved, 1):
        print(f"    {rank}. score {score:.3f}  {chunk['source']}")
        print(f"       {chunk['text'][:88]}...")

    print("\n  the prompt actually sent to the model:")
    print("  " + "-" * 74)
    for line in build_prompt(question, retrieved).splitlines():
        print(f"  | {line}")
    print("  " + "-" * 74)
    print(f"\n  answer: {mock_llm(question, retrieved)}")
    print(
        "\n  That prompt IS retrieval-augmented generation. The model was never\n"
        "  trained on this handbook; it is reading it, right there in the context."
    )

    section("MORE QUESTIONS -- including one the handbook cannot answer")
    questions = [
        "How much can I claim for meals per day?",
        "How often is hardware refreshed?",
        "How many days per week can I work remotely?",
        "What is the parental leave allowance for new fathers?",   # not in any document
    ]
    for q in questions:
        hits = store.search(q, k=3)
        answer = mock_llm(q, hits)
        print(f"\n  Q: {q}")
        print(f"  A: {answer}")
        if answer == "NOT_FOUND":
            print(f"     (nearest chunk was {hits[0][0]['source']} at {hits[0][1]:.3f} --")
            print("      close in topic, but it does not answer the question)")
        else:
            cited = int(answer.rsplit("[", 1)[1].rstrip("]"))
            print(f"     source: {hits[cited - 1][0]['source']}")

    section("WHAT THIS BUYS YOU, AND WHAT COMES NEXT")
    print(
        "  The model answered questions about documents it has never seen, cited\n"
        "  where each answer came from, and declined the one it could not support.\n"
        "  Those three properties are why RAG is the default way to put an LLM on\n"
        "  top of private or changing data.\n\n"
        "  Each step has a module of its own, because each can be done much better:\n"
        "    01-chunking       where you cut decides what can be found\n"
        "    02-embeddings     what 'similar' means, and what it misses\n"
        "    03-vector-stores  storing millions of vectors and searching fast\n"
        "    04-retrieval      hybrid search and reranking, the biggest quality win\n"
        "    05-agentic-rag    letting the model decide what to search for, and retry"
    )


if __name__ == "__main__":
    main()
