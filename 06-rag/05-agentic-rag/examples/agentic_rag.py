"""
Agentic RAG: letting the model decide what to search for, and search again.

Ordinary RAG is a straight line -- embed the question, retrieve once, answer.
That works when the answer sits in one chunk. It fails when:

    the question needs TWO facts that live in different documents
    the question as typed is not a good search query
    the first search comes back with nothing useful

Agentic RAG wraps retrieval in a LOOP and gives the model control of it:

    ROUTE     does this even need retrieval?
    PLAN      what should I search for? (decompose multi-step questions)
    RETRIEVE  search
    GRADE     is this evidence good enough?
    -> if not, reformulate and search again (up to a limit)
    ANSWER    write the answer with citations, or refuse

This file implements that loop against a small corpus, prints the full trace of
the agent's reasoning, and measures agentic against naive RAG on three kinds of
question -- including what the extra intelligence COSTS.

No dependencies.   python agentic_rag.py
"""

import re

# ---------------------------------------------------------------------------
# A corpus where some answers require two hops
# ---------------------------------------------------------------------------

CORPUS = [
    "The billing service is owned by Team Aurora, which also maintains the invoicing pipeline.",
    "Team Aurora is managed by Priya Nair, who reports to the VP of Engineering.",
    "The search service is owned by Team Borealis and handles all query traffic.",
    "Team Borealis is managed by Daniel Okoro, based in the Lisbon office.",
    "Priya Nair joined the company in 2019 and previously led the payments team.",
    "Daniel Okoro joined in 2021 after working on distributed databases.",
    "The invoicing pipeline runs nightly and produces PDF invoices for all customers.",
    "Production incidents are declared in the incidents channel and require a written postmortem.",
    "Postmortems must be published within five working days of an incident being resolved.",
    "The on-call rotation is weekly and is published a month in advance.",
    "Expense claims above 250 pounds require director approval before submission.",
    "The Lisbon office opened in 2022 and hosts the search and platform teams.",
]

# (question, documents needed, the document the ANSWER should come from, kind)
QUESTIONS = [
    ("Who manages the team that owns the billing service?", [0, 1], 1, "multi-hop"),
    ("Who manages the team that owns the search service?", [2, 3], 3, "multi-hop"),
    ("When did the manager of Team Aurora join the company?", [1, 4], 4, "multi-hop"),
    ("How long do we have to publish a postmortem?", [8], 8, "single-hop"),
    ("How often is the on-call rotation?", [9], 9, "single-hop"),
    ("What expense amount requires director approval?", [10], 10, "single-hop"),
    ("What is the company's holiday policy?", [], None, "unanswerable"),
    ("Which vendor supplies our office coffee?", [], None, "unanswerable"),
]

STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "on",
             "at", "by", "or", "be", "who", "what", "when", "how", "which", "that",
             "do", "does", "we", "our", "long", "have", "did"}

# Words shared by so many documents that matching on them means nothing.
GENERIC = {"team", "service", "manager", "managed", "manages", "owns", "owned",
           "company", "office", "joined"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def retrieve(query, k=2):
    """A plain keyword retriever. The point of this file is the LOOP around
    retrieval, not the retriever itself -- see ../../04-retrieval/ for that."""
    query_words = set(tokenize(query))
    scored = []
    for i, document in enumerate(CORPUS):
        overlap = len(query_words & set(tokenize(document)))
        if overlap:
            scored.append((overlap / max(1, len(query_words)), i))
    scored.sort(reverse=True)
    return [(i, score) for score, i in scored[:k]]


# ---------------------------------------------------------------------------
# The mock model. Every decision below is one an LLM would make from a prompt.
# ---------------------------------------------------------------------------

class MockLLM:
    """Deterministic stand-ins for four prompts you would really write."""

    @staticmethod
    def needs_retrieval(question):
        """ROUTER: 'Can you answer from general knowledge, or should we search?'"""
        return not re.match(r"^(hi|hello|thanks|thank you)\b", question.strip().lower())

    @staticmethod
    def grade(query, document):
        """GRADER: 'Does this document actually help with this query?'

        Requires a shared word that is DISTINCTIVE -- every document mentions
        teams and services, so matching on those proves nothing.
        """
        shared = set(tokenize(query)) & set(tokenize(document))
        return any(word not in GENERIC and len(word) > 3 for word in shared)

    @staticmethod
    def find_missing_link(question, evidence):
        """PLANNER: after a hop, what single fact is still missing?

        A real prompt: 'Given the question and what you have found so far, what
        do you still need? Reply with a search query, or DONE.'
        """
        question_lower = question.lower()
        entities = set()
        for _, document in evidence:
            entities.update(re.findall(r"Team \w+|[A-Z][a-z]+ [A-Z][a-z]+", document))

        known = {e.lower() for e in entities}
        if "manage" in question_lower:
            # Need a person for a team we have now identified.
            if any(e.startswith("Team") for e in entities) and not any(
                    re.search(r"managed by", doc) for _, doc in evidence):
                team = sorted(e for e in entities if e.startswith("Team"))[0]
                return f"{team} managed by"
        if "join" in question_lower:
            people = sorted(e for e in entities if not e.startswith("Team"))
            for person in people:
                if person.lower() not in question_lower and not any(
                        re.search(rf"{person} joined", doc) for _, doc in evidence):
                    return f"{person} joined"
        return None

    @staticmethod
    def answer(evidence, focus_query):
        """GENERATOR: answer from the evidence only, and cite it.

        `focus_query` is what we were LAST looking for -- on a multi-hop question
        that is the second hop, which is where the answer lives.
        """
        if not evidence:
            return "NOT_FOUND"
        focus = set(tokenize(focus_query))
        best = max(evidence, key=lambda item: len(focus & set(tokenize(item[1]))))
        if not (focus & set(tokenize(best[1]))):
            return "NOT_FOUND"
        citations = ", ".join(f"[doc {i}]" for i, _ in evidence)
        return f"{best[1]} {citations}"


# ---------------------------------------------------------------------------
# Naive RAG: retrieve once, answer
# ---------------------------------------------------------------------------

def naive_rag(question, k=2):
    hits = retrieve(question, k=k)
    evidence = [(i, CORPUS[i]) for i, _ in hits]
    return {"answer": MockLLM.answer(evidence, question),
            "used": [i for i, _ in hits], "retrievals": 1}


# ---------------------------------------------------------------------------
# Agentic RAG: route, plan, retrieve, grade, retry, answer
# ---------------------------------------------------------------------------

MAX_STEPS = 3


def agentic_rag(question, trace=False):
    steps, evidence, retrievals = [], [], 0

    if not MockLLM.needs_retrieval(question):
        return {"answer": "(answered without retrieval)", "used": [], "retrievals": 0}

    query = question
    for step in range(1, MAX_STEPS + 1):
        hits = retrieve(query, k=2)
        retrievals += 1
        steps.append(f'RETRIEVE #{step}: "{query}" -> docs {[i for i, _ in hits] or "none"}')

        for i, _ in hits:
            if MockLLM.grade(query, CORPUS[i]):
                steps.append(f"  GRADE doc {i}: relevant -- keep")
                if (i, CORPUS[i]) not in evidence:
                    evidence.append((i, CORPUS[i]))
            else:
                steps.append(f"  GRADE doc {i}: nothing distinctive in common -- discard")

        if not evidence:
            steps.append("  no usable evidence -- stop and refuse")
            break

        next_query = MockLLM.find_missing_link(question, evidence)
        if next_query is None:
            steps.append("  PLAN: evidence is sufficient -> answer")
            break
        steps.append(f'  PLAN: still missing a link -> search "{next_query}"')
        query = next_query

    answer = MockLLM.answer(evidence, query)
    steps.append(f"ANSWER: {answer}")
    if trace:
        for line in steps:
            print(f"    {line}")
    return {"answer": answer, "used": [i for i, _ in evidence], "retrievals": retrievals}


# ---------------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def is_correct(result, answer_doc):
    """Answerable: the answer must actually quote the right document.
    Unanswerable: refusing IS the correct behaviour."""
    if answer_doc is None:
        return result["answer"] == "NOT_FOUND"
    return result["answer"].startswith(CORPUS[answer_doc])


def main():
    section("1. WHERE ORDINARY RAG BREAKS")
    question, needed, answer_doc, _ = QUESTIONS[0]
    print(f"  question: {question}")
    print(f"  (it needs BOTH of these, and only the second one answers it)\n")
    for i in needed:
        print(f"    [doc {i}] {CORPUS[i]}")

    naive = naive_rag(question)
    print(f"\n  NAIVE RAG searches once, using the question as the query:")
    print(f"    retrieved: docs {naive['used']}")
    print(f"    answer   : {naive['answer']}")
    print(f"    correct  : {is_correct(naive, answer_doc)}")
    print(
        "\n  It finds the billing service document, because that is what the question\n"
        "  mentions. But the manager's name is in a DIFFERENT document, which the\n"
        "  question never names -- so one search can never reach it."
    )

    section("2. THE AGENT LOOP, STEP BY STEP")
    print(f"  question: {question}\n")
    agentic = agentic_rag(question, trace=True)
    print(f"\n    correct: {is_correct(agentic, answer_doc)}")
    print(
        "\n  The second search does not appear anywhere in the question. The agent\n"
        "  WROTE it, after learning from the first hop that the billing service\n"
        "  belongs to Team Aurora. That is the whole idea: retrieval as a loop the\n"
        "  model steers, rather than a single fixed step."
    )

    section("3. DOES IT ACTUALLY HELP? -- and what it costs")
    print(f"  {'question type':<16} {'naive correct':>14} {'agentic correct':>17}"
          f" {'naive searches':>15} {'agentic searches':>17}")
    print("  " + "-" * 82)
    for kind in ("single-hop", "multi-hop", "unanswerable"):
        subset = [(q, doc) for q, _, doc, k in QUESTIONS if k == kind]
        naive_results = [naive_rag(q) for q, _ in subset]
        agentic_results = [agentic_rag(q) for q, _ in subset]
        naive_ok = sum(is_correct(r, doc) for r, (_, doc) in zip(naive_results, subset))
        agentic_ok = sum(is_correct(r, doc) for r, (_, doc) in zip(agentic_results, subset))
        naive_calls = sum(r["retrievals"] for r in naive_results) / len(subset)
        agentic_calls = sum(r["retrievals"] for r in agentic_results) / len(subset)
        print(f"  {kind:<16} {f'{naive_ok}/{len(subset)}':>14} {f'{agentic_ok}/{len(subset)}':>17}"
              f" {naive_calls:>15.1f} {agentic_calls:>17.1f}")

    print(
        "\n  Agentic retrieval wins exactly where the question needs more than one\n"
        "  search, and it costs more everywhere. On single-hop questions it buys\n"
        "  nothing while still spending calls on grading and planning.\n\n"
        "  Note the unanswerable row: the GRADER is what produces the refusal. It\n"
        "  throws away documents that merely look similar, leaving no evidence -- so\n"
        "  the generator has nothing to write from and says NOT_FOUND."
    )

    section("4. THE PATTERNS, AND WHAT THEY ARE CALLED")
    print(
        "  QUERY REWRITING     turn the question into a better search query\n"
        "  DECOMPOSITION       split a multi-part question into sub-questions\n"
        "  MULTI-HOP           use what hop 1 found to write hop 2's query (above)\n"
        "  GRADING             judge each retrieved chunk before trusting it\n"
        "  SELF-CORRECTION     if the evidence is weak, search again or say so\n"
        "                      (CRAG adds a web-search fallback for exactly this)\n"
        "  SELF-REFLECTION     critique the draft answer against the evidence\n"
        "                      (Self-RAG trains the model to do this with special tokens)\n"
        "  ROUTING             choose a source: docs, SQL, the web, or no retrieval\n"
        "  TOOL USE            retrieval is one tool among several (see ../../07-agents/)"
    )

    section("5. BEFORE YOU BUILD ONE")
    print(
        "  - CAP THE LOOP. MAX_STEPS here is 3. An uncapped agent is an uncapped\n"
        "    bill, and a loop that may never return.\n"
        "  - LOG THE TRACE. Every query the agent wrote, every document it kept or\n"
        "    discarded. You cannot debug what you cannot see.\n"
        "  - LATENCY MULTIPLIES. Three searches plus three model calls is several\n"
        "    seconds before the user sees a single token.\n"
        "  - FIX RETRIEVAL FIRST. An agent searching a badly chunked corpus just\n"
        "    fails more expensively. Chunking, hybrid search and reranking come first.\n"
        "  - MEASURE AGAINST THE SIMPLE VERSION. Section 3 is that experiment. If the\n"
        "    straight pipeline matches it on your questions, ship the straight one."
    )


if __name__ == "__main__":
    main()
