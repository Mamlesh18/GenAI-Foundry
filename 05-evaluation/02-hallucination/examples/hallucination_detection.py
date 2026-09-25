"""
Catching hallucinations without knowing the truth.

A hallucination is text the model states confidently that is not supported by
anything -- not by the facts, not by the documents you gave it. The hard part
is that the model sounds identical whether it knows or is inventing.

Two detectors that work without an oracle, both implemented here:

  1. SELF-CONSISTENCY (the SelfCheckGPT idea)
     Ask the same question several times. If the model knows, the answers agree.
     If it is inventing, it invents something different each time.

  2. GROUNDEDNESS (for RAG)
     Check every claim in the answer against the retrieved documents. Anything
     not supported by the context is flagged, whether or not it is true.

Both are simulated with a mock model whose knowledge we control, so we can
score the detectors against the truth -- which you cannot do in production, and
which is exactly why detectors matter.

No dependencies.   python hallucination_detection.py
"""

import random
import re

# ---------------------------------------------------------------------------
# A mock model that knows some things and not others
# ---------------------------------------------------------------------------

KNOWN = {
    "What is the capital of France?": "Paris",
    "Who wrote Hamlet?": "William Shakespeare",
    "What is the chemical symbol for gold?": "Au",
    "How many continents are there?": "Seven",
    "What is the largest planet?": "Jupiter",
}

# Questions the model was never trained on. It will not say "I don't know" --
# untrained models almost never do. It produces a confident guess instead.
UNKNOWN = {
    "What was the Q3 2031 revenue of Initech?": ["$4.2 million", "$18.7 million",
                                                 "$960,000", "$22 million", "$7.1 million"],
    "Who won the 2087 Nobel Prize in Physics?": ["Dr. Elena Vasquez", "Prof. Hiroshi Tanaka",
                                                 "Dr. Amara Okafor", "Prof. Liu Wei", "Dr. Sofia Rossi"],
    "What is the population of Zerkalo, Uzbekistan?": ["about 42,000", "roughly 8,500",
                                                      "around 120,000", "some 63,400", "nearly 15,000"],
    "When was the Treaty of Kelvingrove signed?": ["1842", "1908", "1763", "1891", "1955"],
    "What is the melting point of unobtainium?": ["3,400 C", "1,250 C", "2,780 C",
                                                  "890 C", "4,100 C"],
}


class MockLLM:
    """Answers confidently either way -- that is the whole problem."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def ask(self, question):
        if question in KNOWN:
            return KNOWN[question]           # it knows: same answer every time
        return self.rng.choice(UNKNOWN[question])   # it guesses: different each time


# ---------------------------------------------------------------------------
# Detector 1: self-consistency
# ---------------------------------------------------------------------------

def consistency_score(model, question, samples=5):
    """Ask repeatedly; return the share of answers that agree with the most
    common one. 1.0 = perfectly consistent, 0.2 = five different answers."""
    answers = [model.ask(question) for _ in range(samples)]
    counts = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values()) / samples, answers


def demo_self_consistency():
    print("=" * 78)
    print("1. SELF-CONSISTENCY -- ask five times, see if the story holds")
    print("=" * 78)
    model = MockLLM(seed=1)

    print(f"  {'question':<46} {'agreement':>10}  sampled answers")
    print("  " + "-" * 94)
    rows = []
    for question in list(KNOWN)[:3] + list(UNKNOWN)[:3]:
        score, answers = consistency_score(model, question)
        truly_known = question in KNOWN
        rows.append((question, score, truly_known))
        distinct = ", ".join(dict.fromkeys(answers))
        shown = distinct if len(distinct) <= 42 else distinct[:39] + "..."
        print(f"  {question[:44]:<46} {score:>9.0%}  {shown}")

    print(
        "\n  The model never hesitates. But when it knows, five samples agree; when\n"
        "  it is inventing, they contradict each other. That contradiction is a\n"
        "  signal you can compute WITHOUT knowing the right answer."
    )
    return rows


def demo_threshold():
    print("\n" + "=" * 78)
    print("2. TURNING THE SIGNAL INTO A DETECTOR -- and picking a threshold")
    print("=" * 78)
    model = MockLLM(seed=2)
    questions = [(q, True) for q in KNOWN] + [(q, False) for q in UNKNOWN]
    scored = [(consistency_score(model, q)[0], known) for q, known in questions]

    print("  'Flag as hallucination if agreement < threshold.'\n")
    print(f"  {'threshold':>10} {'flagged':>8} {'precision':>10} {'recall':>8} {'missed':>8}")
    print("  " + "-" * 50)
    for threshold in (0.4, 0.6, 0.8, 1.0):
        flagged = [(score, known) for score, known in scored if score < threshold]
        true_positives = sum(1 for _, known in flagged if not known)
        precision = true_positives / len(flagged) if flagged else 1.0
        total_hallucinations = sum(1 for _, known in scored if not known)
        recall = true_positives / total_hallucinations
        print(f"  {threshold:>10.1f} {len(flagged):>8} {precision:>9.0%} {recall:>7.0%}"
              f" {total_hallucinations - true_positives:>8}")

    print(
        "\n  A low threshold flags little and misses hallucinations; a high one flags\n"
        "  everything, including correct answers. Where you sit depends on the cost of\n"
        "  each mistake -- a medical assistant should over-flag, a brainstorming tool\n"
        "  should not.\n\n"
        "  HONEST LIMITS of this method:\n"
        "  - It costs N times more (N samples per answer).\n"
        "  - It detects UNCERTAINTY, not falsehood. A model that is confidently and\n"
        "    consistently wrong -- a common misconception, say -- sails through.\n"
        "  - Real implementations compare MEANING, not exact strings: 'about 42,000'\n"
        "    and '42k' should count as agreement. That is what semantic entropy does."
    )


# ---------------------------------------------------------------------------
# Detector 2: groundedness against retrieved context
# ---------------------------------------------------------------------------

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "and",
             "for", "with", "at", "by", "it", "its", "this", "that", "from", "as", "be"}


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9%.,$]+", text.lower()) if w not in STOPWORDS}


def claim_is_grounded(claim, context, threshold=0.75):
    """Crude support check: are most of the claim's content words in the context?"""
    claim_words = content_words(claim)
    if not claim_words:
        return True
    overlap = len(claim_words & content_words(context)) / len(claim_words)
    return overlap >= threshold


def demo_groundedness():
    print("\n" + "=" * 78)
    print("3. GROUNDEDNESS -- did the answer come from the documents?")
    print("=" * 78)

    context = (
        "Initech reported revenue of 4.2 million dollars in Q3 2031. "
        "The company employs 62 people and is based in Austin."
    )
    answers = {
        "grounded": "Initech reported revenue of 4.2 million dollars in Q3 2031.",
        "partly invented": "Initech reported revenue of 4.2 million dollars in Q3 2031. "
                           "The company was founded by Peter Gibbons in 2019.",
        "fully invented": "Initech earned 18.7 million dollars and employs 400 people.",
        "honest refusal": "NOT_FOUND",
    }

    print(f"  context: {context}\n")
    for label, answer in answers.items():
        if answer == "NOT_FOUND":
            print(f"  {label:<16} -> refused (NOT_FOUND), nothing to check")
            continue
        # Split on sentence ends only -- a full stop followed by a capital --
        # so that numbers like "4.2 million" survive intact.
        claims = [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer) if c.strip()]
        results = [(c, claim_is_grounded(c, context)) for c in claims]
        unsupported = [c for c, ok in results if not ok]
        print(f"  {label:<16} -> {len(results) - len(unsupported)}/{len(results)} claims supported")
        for claim in unsupported:
            print(f"                     UNSUPPORTED: {claim[:60]}")

    print(
        "\n  Every claim is checked against the retrieved text, so this works even when\n"
        "  you do not know the true answer. It is the standard metric for RAG, where\n"
        "  it is called faithfulness or groundedness.\n\n"
        "  HONEST LIMITS of word overlap: it cannot read. 'The drug is effective' and\n"
        "  'The drug is not effective' share every content word, so this checker calls\n"
        "  the second one grounded:"
    )
    ctx = "The drug is effective in adults."
    print(f"     context : {ctx}")
    print(f"     claim   : The drug is not effective in adults.")
    print(f"     verdict : {'grounded' if claim_is_grounded('The drug is not effective in adults.', ctx) else 'unsupported'}"
          "   <- WRONG")
    print(
        "\n  Production systems use a model to judge entailment instead of counting\n"
        "  words (Ragas, TruLens, DeepEval all do this). The structure above is right;\n"
        "  the comparison function needs to understand language."
    )


def demo_prevention():
    print("\n" + "=" * 78)
    print("4. THE CHEAPEST FIX IS NOT A DETECTOR")
    print("=" * 78)
    print(
        "  Detection happens after the model has already invented something. Two\n"
        "  prompt-level changes prevent much of it in the first place:\n\n"
        "    1. AN ESCAPE HATCH\n"
        "       'If the context does not contain the answer, reply NOT_FOUND.'\n"
        "       Without permission to decline, a model under pressure to answer will\n"
        "       invent. This single line removes a large share of RAG hallucinations.\n\n"
        "    2. CITATIONS\n"
        "       'Cite the source id for every claim.' Forces the answer to point at\n"
        "       evidence, and lets a human (or a script) check it.\n\n"
        "  Then measure: what fraction of unanswerable questions does the system\n"
        "  correctly refuse? That number belongs in your eval set alongside accuracy."
    )


def main():
    demo_self_consistency()
    demo_threshold()
    demo_groundedness()
    demo_prevention()


if __name__ == "__main__":
    main()
