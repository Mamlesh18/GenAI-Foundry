# 02 · Hallucination

> **In one sentence:** a hallucination is output the model states confidently that is not supported
> by the facts or by the documents you gave it — and the hard part is that it reads exactly like
> output that *is* supported.

---

## 1. Why models hallucinate

It is not a bug that will be patched. It follows from how the model works
([01-foundations/02](../../01-foundations/02-llms/)):

| Cause | Explanation |
|---|---|
| **The objective** | It is trained to produce *plausible* continuations, not *true* ones. Nothing in next-token prediction represents truth. |
| **No "I don't know" by default** | Unless trained or instructed to decline, the most plausible continuation of a question is an answer — any answer. |
| **Knowledge cutoff** | It cannot know what happened after training, but it can still write a confident sentence about it. |
| **Fluency is uniform** | The model's confidence in its wording is unrelated to its confidence in the facts. |
| **Retrieval failures** | In RAG, if the right document was not retrieved, the model fills the gap itself. |

---

## 2. Two kinds, and why the distinction matters

| Kind | Definition | How to check |
|---|---|---|
| **Factual** | Contradicts the real world: invented citations, wrong dates, fake people | Needs an external source of truth |
| **Faithfulness** (also called groundedness) | Contradicts or goes beyond the context *you supplied* | Checkable against the context — **no oracle needed** |

The second is the practical one. In a RAG system you have the retrieved documents, so you can check
every claim against them mechanically. That is why faithfulness is the standard RAG metric.

---

## 3. How to detect it

| Method | Idea | Cost | Needs |
|---|---|---|---|
| **Self-consistency** ([SelfCheckGPT](https://arxiv.org/abs/2303.08896)) | Ask several times; inconsistent answers mean the model is inventing | N× generation | nothing external |
| **Semantic entropy** | Same idea, but cluster answers by *meaning* rather than exact text | N× generation | an entailment model |
| **Groundedness / NLI check** | Test whether the context entails each claim | 1 extra call per claim | the retrieved context |
| **Citation verification** | Require source ids, then check the cited passage supports the claim | cheap | citations in the output |
| **Fact decomposition** ([FActScore](https://arxiv.org/abs/2305.14251)) | Split long text into atomic facts, verify each | expensive | a knowledge source |
| **Uncertainty from logprobs** | Low token probability suggests low confidence | free | logprob access |

### What the script shows

[`hallucination_detection.py`](examples/hallucination_detection.py) implements the first and third
against a mock model whose knowledge we control.

**Self-consistency.** Ask five times and look at the agreement:

| Question | Agreement | What it means |
|---|---|---|
| "What is the capital of France?" | 100% | it knows |
| "Who wrote Hamlet?" | 100% | it knows |
| "What was the Q3 2031 revenue of Initech?" | 40% | five samples, several different numbers |
| "What is the population of Zerkalo, Uzbekistan?" | 60% | inventing |

Turning that into a detector means choosing a threshold, and the threshold is a trade-off:

| Flag if agreement < | Flagged | Precision | Recall |
|---|---|---|---|
| 0.4 | 0 | — | 0% |
| 0.6 | 3 | 100% | 60% |
| 0.8 | 5 | 100% | 100% |

**Honest limits:** it costs N times more, and it detects **uncertainty, not falsehood**. A model
that is confidently and consistently wrong — repeating a common misconception — passes cleanly.

**Groundedness.** Check each claim against the retrieved context:

| Answer | Result |
|---|---|
| Grounded answer | 1/1 claims supported |
| Grounded + one invented sentence | 1/2 supported — the invented one is flagged |
| Fully invented | 0/1 supported |
| `NOT_FOUND` refusal | nothing to check |

The script's checker counts word overlap, and then shows its own failure: given the context
*"The drug is effective in adults"*, it marks the claim *"The drug is **not** effective in adults"*
as **grounded**, because the words all match. Negation, quantities and attribution need a model that
understands entailment — which is what Ragas, TruLens and DeepEval use. The structure is right; the
comparison function has to be smarter than string matching.

---

## 4. Benchmarks for hallucination

| Benchmark | Measures |
|---|---|
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | Whether models repeat common human misconceptions |
| [HaluEval](https://arxiv.org/abs/2305.11747) | Whether models can recognise hallucinated content |
| [FActScore](https://arxiv.org/abs/2305.14251) | Share of atomic facts in long-form text that are supported |

Useful for comparing models. For your own system, measure faithfulness and refusal accuracy on your
own data instead.

---

## 5. Reducing it — cheapest first

1. **Give it the facts.** [RAG](../../06-rag/) replaces recall with reading. The single biggest win.
2. **Give it permission to decline.** *"If the context does not contain the answer, reply
   NOT_FOUND."* Without an escape hatch, a model under pressure to answer invents one.
3. **Demand citations.** *"Cite the source id for every claim."* Makes claims checkable — by a
   human and by a script.
4. **Lower the temperature** for factual tasks.
5. **Constrain the output.** Structured output with validation makes some inventions impossible.
6. **Verify before showing.** Run a groundedness check and either flag or regenerate.
7. **Escalate to a human** where the cost of being wrong is high.

Note the ordering: **prevention beats detection**. Detection happens after the model has already
invented something.

---

## 6. Measuring it in a real product

Put these four numbers in your eval set ([module 03](../03-model-evaluation/)):

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Is every claim supported by the retrieved context? |
| **Refusal accuracy** | Does it say NOT_FOUND exactly when it should? |
| **Citation validity** | Do the cited sources exist, and do they support the claim? |
| **Answer correctness** | Is it actually right, on the cases where you know the answer? |

Include **unanswerable questions** — at least a quarter of your set. A system that never refuses has
not been tested for the failure that matters most.

---

## 7. Examples

```bash
python examples/hallucination_detection.py     # instant, no dependencies
```

Implements self-consistency detection with a threshold sweep, groundedness checking for RAG, and
demonstrates the failure mode of each.

---

## 8. Exercises

1. **Read the signal.** In the script's output, why does the model answer "Paris" five times out of
   five, but give different revenue figures each time? What is being measured?
2. **Pick a threshold.** For a medical assistant and for a brainstorming tool, which agreement
   threshold would you choose, and what does each choice cost you?
3. **Break the detector.** Add a question to `KNOWN` whose answer is *wrong* (a misconception). Does
   self-consistency flag it? What does that tell you about what the method really measures?
4. **Fix the negation bug.** Improve `claim_is_grounded` so the negation example is caught. Then
   find an input that defeats your improved version too.
5. **Test the escape hatch.** With any API, ask 10 questions that a provided context cannot answer,
   with and without the NOT_FOUND instruction. Count the inventions.
6. **Design the eval.** Write 20 test cases for a document Q&A system: 10 answerable, 5 unanswerable,
   5 answerable only by combining two documents. What is your pass criterion for each group?

---

## 9. Projects to build and test

### Beginner — A refusal test set
Build 30 questions against a small document set: answerable, unanswerable, and ambiguous.

**How to test it:** measure refusal accuracy before and after adding the escape hatch to the prompt.
Report both numbers — the improvement is usually large and is the cheapest win available.

### Intermediate — A groundedness checker
Use an LLM to judge whether each claim in an answer is entailed by the context, and validate the
judge against your own hand labels.

**How to test it:** hand-label 50 claims, then report your checker's precision and recall. Confirm
it catches the negation case that word overlap misses.

### Advanced — Self-consistency in production
Wrap a real model so that high-stakes answers are sampled n times, clustered by meaning, and flagged
when they disagree.

**How to test it:** measure detection rate against a labelled set, plus the added cost and latency.
Then decide, with numbers, whether it is worth running on every request or only on some — that
trade-off is the actual engineering decision.

---

## 10. Resources

**Start here**
- [What are AI hallucination evaluations? Metrics and methods](https://www.braintrust.dev/articles/ai-hallucination-evaluations-metrics-methods-2026) — Braintrust; practical overview.
- [How to Detect Hallucinations in Your LLM Applications](https://www.getmaxim.ai/articles/how-to-detect-hallucinations-in-your-llm-applications/) — detection methods compared.
- [Ragas documentation](https://docs.ragas.io/) — faithfulness and related RAG metrics, ready to use.

**Go deeper**
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) — Nature. Clustering answers by meaning rather than wording.
- [awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) — a maintained reading list.

**Papers**
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) — the sampling-and-comparing method implemented here
- [TruthfulQA](https://arxiv.org/abs/2109.07958) · [HaluEval](https://arxiv.org/abs/2305.11747) · [FActScore](https://arxiv.org/abs/2305.14251)

**Related in this repo**
- [06 · RAG](../../06-rag/) — grounding answers in retrieved documents
- [skills/03 · Structured Outputs](../../skills/03-structured-outputs/) — validation that makes some failures impossible

---

**Previous:** [01 · Benchmarks](../01-benchmarks/) · **Next:** [03 · Model Evaluation](../03-model-evaluation/)
