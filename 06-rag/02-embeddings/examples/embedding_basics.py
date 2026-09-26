"""
What a vector actually is, and why you need a LEARNED one.

An embedding turns text into a list of numbers positioned so that similar
meanings sit close together. This script builds the idea from the bottom:

  1. a vector you can read: counting words
  2. cosine similarity, and why vectors are normalised
  3. where word counting fails -- paraphrases share no words
  4. the same pairs through a real embedding model (if installed)
  5. what embeddings still cannot do

    pip install numpy                sentence-transformers optional but
    python embedding_basics.py       strongly recommended for section 4
"""

import re

import numpy as np

STOPWORDS = {"a", "an", "the", "is", "was", "are", "were", "of", "to", "in", "on",
             "and", "for", "with", "at", "by", "it", "this", "that", "from", "as"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# 1 + 2. Vectors you can read, and cosine similarity
# ---------------------------------------------------------------------------

def bag_of_words(texts):
    """The simplest possible 'embedding': one dimension per word."""
    vocabulary = sorted({w for t in texts for w in tokenize(t)})
    index = {w: i for i, w in enumerate(vocabulary)}
    vectors = np.zeros((len(texts), len(vocabulary)))
    for row, text in enumerate(texts):
        for word in tokenize(text):
            vectors[row, index[word]] += 1
    return vectors, vocabulary


def normalise(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def demo_vectors():
    section("1. A VECTOR YOU CAN READ")
    texts = ["the cat sat on the mat", "a cat sat on a rug", "quarterly revenue rose"]
    vectors, vocabulary = bag_of_words(texts)

    print(f"  vocabulary ({len(vocabulary)} words): {vocabulary}\n")
    for text, vector in zip(texts, vectors):
        print(f"  {text:<28} {vector.astype(int)}")
    print(
        "\n  One dimension per word, the value is how often it appears. A real\n"
        "  embedding has 384 to 3,072 dimensions that mean nothing individually --\n"
        "  but the principle is identical: text becomes a point in space."
    )

    section("2. COSINE SIMILARITY -- comparing directions, not lengths")
    unit = normalise(vectors)
    print("  cosine(a, b) = dot product of the normalised vectors, from -1 to 1\n")
    pairs = [(0, 1, "cat/mat vs cat/rug"), (0, 2, "cat/mat vs revenue")]
    for i, j, label in pairs:
        print(f"  {label:<28} {float(unit[i] @ unit[j]):.3f}")
    print(
        "\n  Normalising first is why vector databases can use a plain dot product:\n"
        "  for unit vectors, dot product IS cosine similarity, and it is faster.\n"
        "  Ignoring length also means a long document and a short query about the\n"
        "  same topic still match."
    )


# ---------------------------------------------------------------------------
# 3. Where counting words fails
# ---------------------------------------------------------------------------

PAIRS = [
    ("a dog chasing a ball", "a puppy playing fetch", "same meaning, no shared words"),
    ("how do I reset my password", "I forgot my login credentials", "same intent, different words"),
    ("the meeting is on Tuesday", "the meeting is not on Tuesday", "opposite meaning, near-identical words"),
    ("annual leave policy", "quarterly revenue report", "unrelated"),
]


def demo_lexical_failure():
    section("3. WHY COUNTING WORDS IS NOT ENOUGH")
    texts = [t for pair in PAIRS for t in pair[:2]]
    vectors, _ = bag_of_words(texts)
    unit = normalise(vectors)

    print(f"  {'pair':<58} {'similarity':>10}")
    print("  " + "-" * 72)
    for n, (a, b, note) in enumerate(PAIRS):
        score = float(unit[2 * n] @ unit[2 * n + 1])
        print(f"  {a + '  /  ' + b:<58} {score:>10.3f}")
        print(f"    {note}")
    print(
        "\n  Word counting says 'a dog chasing a ball' and 'a puppy playing fetch'\n"
        "  are completely unrelated, because they share no words. And it says the\n"
        "  two opposite sentences about Tuesday are nearly identical, because they\n"
        "  share almost all of them.\n\n"
        "  Fixing the first problem is exactly what learned embeddings are for."
    )


# ---------------------------------------------------------------------------
# 4. Real embeddings
# ---------------------------------------------------------------------------

def demo_real_embeddings():
    section("4. THE SAME PAIRS, THROUGH A REAL EMBEDDING MODEL")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers is not installed, so this section is skipped.\n")
        print("    pip install sentence-transformers\n")
        print("  It downloads a ~90 MB model on first use and runs on a CPU.")
        return None

    print("  loading all-MiniLM-L6-v2 (small, fast, 384 dimensions)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [t for pair in PAIRS for t in pair[:2]]
    vectors = model.encode(texts, normalize_embeddings=True)

    print(f"\n  {'pair':<58} {'bag of words':>13} {'embeddings':>11}")
    print("  " + "-" * 84)
    bow = normalise(bag_of_words(texts)[0])
    for n, (a, b, _) in enumerate(PAIRS):
        lexical = float(bow[2 * n] @ bow[2 * n + 1])
        semantic = float(vectors[2 * n] @ vectors[2 * n + 1])
        print(f"  {a + '  /  ' + b:<58} {lexical:>13.3f} {semantic:>11.3f}")

    print(
        "\n  The paraphrases jump from ~0 to a high score: the model has learned that\n"
        "  dogs and puppies, chasing and fetching, belong together. That is the whole\n"
        "  reason RAG uses embeddings instead of keyword search.\n\n"
        "  But look at the Tuesday pair. The model ALSO scores that pair highly,\n"
        "  because 'not' barely changes the sentence's overall meaning as encoded.\n"
        "  Embeddings are weak at negation -- a limitation worth remembering."
    )
    return vectors


# ---------------------------------------------------------------------------
# 5. Practical properties
# ---------------------------------------------------------------------------

def demo_practicalities():
    section("5. THINGS THAT WILL BITE YOU")
    print(
        "  VECTORS FROM DIFFERENT MODELS ARE NOT COMPARABLE.\n"
        "    The numbers mean different things. Change your embedding model and you\n"
        "    must re-embed every document. Plan for it: store the model name and\n"
        "    version alongside the vectors.\n\n"
        "  EXACT IDENTIFIERS DO NOT EMBED WELL.\n"
        "    'ERR-4471' and 'ERR-4472' are near-identical as vectors and completely\n"
        "    different in meaning. Use keyword search for codes, SKUs and names --\n"
        "    that is why hybrid search exists (../04-retrieval/).\n\n"
        "  NEGATION IS BARELY REPRESENTED.\n"
        "    Section 4 shows it: 'is' and 'is not' land close together.\n\n"
        "  LONGER IS NOT BETTER.\n"
        "    A 3,072-dimension vector costs 4x the storage and search time of a\n"
        "    768-dimension one. Check whether the quality gain is worth it for you;\n"
        "    many models now support shortening vectors with little loss.\n\n"
        "  ASYMMETRY MATTERS.\n"
        "    A short question and a long passage are different kinds of text. Some\n"
        "    models expect a prefix like 'query: ' vs 'passage: '. Read the model\n"
        "    card -- using it wrong quietly costs you retrieval quality."
    )

    section("CHOOSING A MODEL")
    print(
        f"  {'option':<34} {'why':<44}\n  " + "-" * 78)
    rows = [
        ("all-MiniLM-L6-v2 (open, 384d)", "tiny, fast, runs on a laptop; good baseline"),
        ("BGE / E5 / GTE / Nomic (open)", "stronger, still self-hosted; check licence"),
        ("OpenAI text-embedding-3", "cheap, easy, solid general quality"),
        ("Cohere embed / Voyage", "strong multilingual and domain options"),
        ("Gemini embedding", "competitive, tops public leaderboards"),
    ]
    for name, why in rows:
        print(f"  {name:<34} {why:<44}")
    print(
        "\n  Check the MTEB leaderboard, but weight the tasks that resemble yours --\n"
        "  the average score across all tasks is rarely the number you care about.\n"
        "  Then test two or three on YOUR documents and questions. Retrieval quality\n"
        "  on your data is the only benchmark that decides anything."
    )


def main():
    demo_vectors()
    demo_lexical_failure()
    demo_real_embeddings()
    demo_practicalities()


if __name__ == "__main__":
    main()
