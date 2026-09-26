"""
Where you cut the document decides what can ever be found.

Chunking is the least glamorous part of RAG and the one that most often decides
whether it works. The rule is simple: RETRIEVAL RETURNS CHUNKS. If the answer to
a question is split across two chunks, no retriever, reranker or model can put
it back together.

This script measures that directly. It takes a document, a set of questions
whose answers live in known spans of text, and asks of each strategy:

    for how many questions does SOME SINGLE CHUNK contain the whole answer?

No dependencies.   python chunking_strategies.py
"""

import re

# ---------------------------------------------------------------------------
# A document with structure -- headings and paragraphs, like real ones
# ---------------------------------------------------------------------------

DOCUMENT = """# Employee Handbook

## Annual Leave
Every full-time employee receives 24 days of paid annual leave per year. Leave
accrues monthly. Unused days may be carried into the next year, up to a maximum
of 5 days. Requests must be approved by your manager at least two weeks in
advance.

## Expenses
Travel expenses must be submitted within 30 days of the trip. Meals are
reimbursed up to 40 pounds per day. Any single expense above 250 pounds requires
prior written approval from a director.

## Equipment
New joiners receive a laptop and a monitor. Hardware refresh happens every three
years. Lost or damaged equipment must be reported to IT within 24 hours.

## Remote Work
Employees may work remotely up to three days per week. Fully remote arrangements
require director approval and are reviewed every six months. Core collaboration
hours are 10:00 to 16:00 in your local time zone.
"""

# Each question's answer is exactly this span of the document. A chunk is only
# useful if it contains the span COMPLETE.
QUESTIONS = [
    ("How many leave days do I get?",
     "24 days of paid annual leave per year"),
    ("How many unused days can I carry over?",
     "carried into the next year, up to a maximum of 5 days"),
    ("How much notice for a leave request?",
     "approved by your manager at least two weeks in advance"),
    ("What is the meal allowance?",
     "reimbursed up to 40 pounds per day"),
    ("When do I need director approval for an expense?",
     "above 250 pounds requires prior written approval from a director"),
    ("How often is hardware refreshed?",
     "Hardware refresh happens every three years"),
    ("How fast must I report damaged equipment?",
     "reported to IT within 24 hours"),
    ("How many remote days per week?",
     "remotely up to three days per week"),
]


def normalise(text):
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def fixed_size(text, size_words, overlap_words=0):
    words = normalise(text).split()
    step = max(1, size_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size_words >= len(words):
            break
    return chunks


def by_sentence(text, sentences_per_chunk=3):
    sentences = re.split(r"(?<=[.!?])\s+", normalise(text))
    return [" ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, len(sentences), sentences_per_chunk)]


def split_sections(text):
    """Return (section title, section body) pairs, using the markdown headings."""
    sections = []
    for part in re.split(r"\n(?=## )", text):
        lines = [line for line in part.strip().splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("## "):
            sections.append((lines[0][3:].strip(), normalise(" ".join(lines[1:]))))
        else:                                   # the document title block
            sections.append((lines[0].lstrip("# ").strip(), normalise(" ".join(lines[1:]))))
    return [(title, body) for title, body in sections if body]


def by_structure(text):
    """Split on markdown headings: each section becomes one chunk. The document
    already tells you where the topic boundaries are -- use them."""
    return [body for _, body in split_sections(text)]


def by_structure_with_titles(text):
    """Structure-aware, plus the document and section title prepended to every
    chunk. This is the cheap version of 'contextual retrieval': a chunk that
    says what it is about is far easier to retrieve."""
    doc_title = text.strip().splitlines()[0].lstrip("# ").strip()
    return [f"[{doc_title} > {title}] {body}" for title, body in split_sections(text)]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def answer_intact_rate(chunks):
    """Share of questions whose answer span sits COMPLETE inside some chunk."""
    normalised = [normalise(c) for c in chunks]
    intact, broken = 0, []
    for question, span in QUESTIONS:
        span_norm = normalise(span)
        if any(span_norm in c for c in normalised):
            intact += 1
        else:
            broken.append(question)
    return intact / len(QUESTIONS), broken


def stats(chunks):
    sizes = [len(c.split()) for c in chunks]
    return len(chunks), sum(sizes) / len(sizes), max(sizes)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    strategies = [
        ("fixed 20 words, no overlap", fixed_size(DOCUMENT, 20)),
        ("fixed 20 words, 50% overlap", fixed_size(DOCUMENT, 20, 10)),
        ("fixed 60 words, no overlap", fixed_size(DOCUMENT, 60)),
        ("fixed 60 words, 25% overlap", fixed_size(DOCUMENT, 60, 15)),
        ("whole document as one chunk", fixed_size(DOCUMENT, 10_000)),
        ("3 sentences per chunk", by_sentence(DOCUMENT, 3)),
        ("by section heading", by_structure(DOCUMENT)),
        ("by section + titles prepended", by_structure_with_titles(DOCUMENT)),
    ]

    section("1. CAN THE ANSWER EVEN BE FOUND?")
    print("  'answer intact' = some single chunk contains the whole answer span.")
    print("  If it is 0, no retriever can ever return that answer complete.\n")
    print(f"  {'strategy':<32} {'chunks':>7} {'avg words':>10} {'answers intact':>15}")
    print("  " + "-" * 70)
    results = {}
    for name, chunks in strategies:
        count, avg, _ = stats(chunks)
        rate, broken = answer_intact_rate(chunks)
        results[name] = (rate, broken)
        print(f"  {name:<32} {count:>7} {avg:>10.0f} {rate:>14.0%}")

    section("2. WHAT SMALL CHUNKS LOSE")
    rate, broken = results["fixed 20 words, no overlap"]
    print(f"  fixed 20 words, no overlap: {rate:.0%} of answers survive intact.")
    print("  Questions whose answer was cut in half:")
    for question in broken:
        print(f"    - {question}")
    overlap_rate, overlap_broken = results["fixed 20 words, 50% overlap"]
    print(f"\n  Add 50% overlap and it rises to {overlap_rate:.0%}"
          f" ({len(broken) - len(overlap_broken)} answers recovered).")
    print(
        "\n  Overlap is insurance against cutting through the middle of an answer.\n"
        "  It costs storage and some duplicate retrieval, and it is almost always\n"
        "  worth it. 10-20% of the chunk size is the usual choice."
    )

    section("3. WHAT LARGE CHUNKS LOSE")
    big_count, big_avg, _ = stats(fixed_size(DOCUMENT, 10_000))
    print(f"  the whole document as a single chunk: {big_count} chunk of {big_avg:.0f} words,"
          f" {results['whole document as one chunk'][0]:.0%} answers intact.")
    print(
        "\n  Every answer survives -- so why not always use huge chunks? Because\n"
        "  retrieval quality collapses in a different way:\n\n"
        "    - one vector has to represent many topics, so it matches everything\n"
        "      weakly and nothing strongly\n"
        "    - you spend context (and money) on text the question did not need\n"
        "    - the model has more irrelevant material to be distracted by\n\n"
        "  Small chunks retrieve precisely but lose context; large chunks keep\n"
        "  context but retrieve vaguely. That trade-off is the whole game."
    )

    section("4. USE THE DOCUMENT'S OWN STRUCTURE")
    print("  by section heading:")
    for chunk in by_structure(DOCUMENT)[:3]:
        print(f"    - {chunk[:72]}...")
    print(
        "\n  Documents tell you where the topic boundaries are: headings, paragraphs,\n"
        "  list items, code blocks, table rows. Splitting there keeps each chunk about\n"
        "  ONE thing, which is exactly what an embedding represents well.\n\n"
        "  And prepending the document and section title costs almost nothing:\n"
        f"    {by_structure_with_titles(DOCUMENT)[1][:76]}...\n\n"
        "  Now a chunk that says 'up to 40 pounds per day' also says it is about\n"
        "  Expenses. Anthropic's 'contextual retrieval' does a richer version of this\n"
        "  -- an LLM writes a sentence of context for every chunk -- and reports a 35%\n"
        "  reduction in retrieval failures from that alone, 49% combined with BM25."
    )

    section("PRACTICAL DEFAULTS")
    print(
        "  - Start at roughly 500 tokens per chunk with 10-20% overlap. Published\n"
        "    benchmarks keep finding this hard to beat.\n"
        "  - Split on structure (headings, paragraphs) before falling back to size.\n"
        "  - Keep metadata on every chunk: source, section, date. You need it to\n"
        "    cite, and to filter (see ../03-vector-stores/).\n"
        "  - Prepend the document and section title to each chunk.\n"
        "  - Then MEASURE on your own questions. Chunking is the cheapest parameter\n"
        "    to change and the one with the largest effect on retrieval quality."
    )


if __name__ == "__main__":
    main()
