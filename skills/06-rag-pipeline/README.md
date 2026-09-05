# 06 · RAG Pipeline

**Level:** Intermediate · **Time:** ~6 hours

---

## 1. What you will be able to do

Build a system that answers questions over your own documents, cites its sources, and says "I don't
know" instead of inventing an answer — and measure all three of those properties.

---

## 2. Prerequisites

- Skills: [02 · Prompt Engineering](../02-prompt-engineering/),
  [05 · Embeddings & Vector Search](../05-embeddings-and-vector-search/)
- Concepts: [06 · RAG](../../06-rag/)

---

## 3. The concept, briefly

Retrieval-Augmented Generation: find relevant passages, put them in the prompt, and instruct the
model to answer only from them.

```
question -> [retrieve top-k chunks] -> [build grounded prompt] -> [generate] -> answer + citations
```

It solves the three problems a bare model cannot: knowledge after the training cutoff, private data
it never saw, and unverifiable claims.

**RAG is retrieval plus a prompt.** That is genuinely all it is — and it means almost every RAG
failure is a *retrieval* failure wearing a generation costume. If the right chunk was not
retrieved, no prompt can save you. Debug retrieval first, always.

The generation prompt has three load-bearing lines:

- *"Answer using only the context below."* — grounding
- *"If the context does not contain the answer, reply NOT_FOUND."* — the escape hatch, which
  prevents an enormous share of hallucinations on its own
- *"Cite the source id for every claim."* — verifiability

---

## 4. Build it

1. **Reuse your retriever from skill 05.** Hybrid, with metadata. Do not rebuild it.

2. **Write the grounded prompt.** Context wrapped in delimiters with visible source ids, the
   question, and the three lines above. Number the chunks `[1]`, `[2]`, ... so citations are cheap.

3. **Generate an answer.** Low temperature — you want faithfulness, not creativity.

4. **Test the escape hatch.** Ask something definitely not in your documents. If it invents an
   answer, your prompt is not strong enough. Fix it before going further; this is the single most
   important behaviour in the system.

5. **Add citations and verify them.** Parse the `[n]` markers, map back to sources, and display
   them. Then check whether the cited chunk *actually supports* the claim — models cite
   plausibly-but-wrongly more often than you would like.

6. **Build an evaluation set.** 30 questions across four categories: answerable from one chunk,
   answerable only by combining chunks, not answerable at all, and ambiguous.

7. **Measure four things separately.**
   - *Retrieval*: was the right chunk in the top k?
   - *Faithfulness*: is every claim supported by the retrieved context?
   - *Answer correctness*: is it actually right?
   - *Refusal accuracy*: does it decline exactly when it should?

   Measuring them separately is what lets you fix the right component.

8. **Handle the multi-chunk case.** Questions needing two documents usually fail first. Increase k,
   try query decomposition, and re-measure.

9. **Add conversation.** Follow-up questions ("what about the second one?") break retrieval,
   because the follow-up alone is not a searchable query. Rewrite it into a standalone question
   using the history *before* retrieving.

10. **Show your work in the UI.** Display the retrieved chunks alongside the answer. Users trust a
    system they can check, and you will debug ten times faster.

---

## 5. Checkpoint

Against your 30-question set:

- [ ] Retrieval recall@5 above 85%
- [ ] Every answer carries citations that resolve to real chunks
- [ ] Spot-checking 10 citations, each actually supports the claim it is attached to
- [ ] It returns NOT_FOUND for **every** unanswerable question — no exceptions
- [ ] Multi-chunk questions succeed at a rate you have measured and can state
- [ ] A follow-up question in conversation retrieves correctly
- [ ] You can point at any wrong answer and say whether retrieval or generation caused it
- [ ] End-to-end latency is under 3 seconds

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| No escape hatch | Confident, fluent, wrong answers | Explicit NOT_FOUND instruction, tested |
| Debugging generation when retrieval failed | Endless prompt tweaking, no improvement | Always check what was retrieved first |
| Evaluating end-to-end only | You know it is broken, not where | Measure retrieval and generation separately |
| Citations never verified | It cites the wrong chunk and nobody notices | Spot-check, and build a faithfulness metric |
| No unanswerable questions in the test set | You never discover it cannot refuse | At least a quarter should be unanswerable |
| k too large | Noise crowds out the answer; "lost in the middle" | Tune k; rerank instead of widening |
| Raw follow-ups sent to retrieval | Conversational RAG silently degrades | Rewrite into a standalone query first |
| Retrieved content treated as trusted | Prompt injection via your own documents | Delimit it, and label it as untrusted data |

---

## 7. Going further

- **Next skills:** [07 · Building Agents](../07-building-agents/), [08 · Evaluation](../08-evaluation-and-testing/)
- **Agentic RAG:** let the model decide whether and what to retrieve, and to retrieve again after
  reading
- **Query decomposition:** split complex questions into sub-questions, retrieve for each
- **Contextual retrieval:** prepend a short document-level summary to each chunk before embedding
- **GraphRAG:** build an entity graph for questions that need to connect facts across many documents

**Links:** [RAG paper](https://arxiv.org/abs/2005.11401) ·
[Ragas](https://docs.ragas.io/) — RAG evaluation metrics ·
[Anthropic on contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)
