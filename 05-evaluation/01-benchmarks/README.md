# 01 · Benchmarks

> **In one sentence:** a benchmark is a fixed set of questions with known answers, used to compare
> models — useful for shortlisting, and far less meaningful than the numbers suggest.

This module is as much about **reading benchmark claims sceptically** as about running them.

---

## 1. The ones you will see quoted

| Benchmark | Tests | Format | Watch out for |
|---|---|---|---|
| **MMLU** | General knowledge, 57 subjects | 4-option multiple choice | The most contaminated major benchmark; scores are saturated |
| **[MMLU-Pro](https://arxiv.org/abs/2406.01574)** | Same idea, harder | 10 options, reasoning-heavy | Lower baseline (10%), so numbers look lower by design |
| **[GPQA](https://arxiv.org/abs/2311.12022)** | Graduate biology/physics/chemistry | Multiple choice | "Google-proof": non-expert humans score ~34% even with web access |
| **GSM8K / AIME** | Maths word problems / competition maths | Free-form answer | GSM8K is largely saturated; AIME is the current stand-in |
| **HumanEval / MBPP** | Writing small functions | Code, run against tests | Tiny and saturated; not representative of real software work |
| **[SWE-bench](https://arxiv.org/abs/2310.06770)** | Fixing real GitHub issues | Agentic, patch must pass tests | Much closer to real work; "Verified" is the human-checked subset |
| **[IFEval](https://arxiv.org/abs/2311.07911)** | Following instructions exactly | Programmatically checkable instructions | Measures compliance, not intelligence — and that is the point |
| **BBH** | Hard reasoning tasks | Mixed | Older; partly saturated |
| **[LiveBench](https://arxiv.org/abs/2406.19314)** | Broad capability | Refreshed regularly | Built specifically to resist contamination |
| **[LiveCodeBench](https://arxiv.org/abs/2403.07974)** | Coding | Problems filtered by date | Filter by date after a model's cutoff for a fair read |
| **[LMArena](https://lmarena.ai)** | Human preference | Pairwise votes, ranked | Measures what people *like*, which rewards style as well as substance |

**How to use this table:** pick two or three that resemble your task, treat them as a shortlist
filter, and then build your own eval set ([module 03](../03-model-evaluation/)).

---

## 2. How scoring actually works

| Task shape | Scoring method |
|---|---|
| Multiple choice | Either generate an answer and parse the letter, or compare the model's likelihood of each option. **These give different scores for the same model.** |
| Short factual answer | Exact match, or normalised match (lowercase, strip punctuation and articles) |
| Code | Run the generated code against unit tests — `pass@k` is "did any of k attempts pass?" |
| Open-ended text | An LLM judge or human raters; see [module 03](../03-model-evaluation/) |

---

## 3. Four ways benchmark numbers lie

All four are demonstrated, with numbers, by
[`benchmark_harness.py`](examples/benchmark_harness.py).

### The model was right; your parser said it was wrong

Four mock models, all with **exactly 80% true ability**, differing only in how they write the
answer:

| The model writes... | Strict parser | Robust parser |
|---|---|---|
| `B` | 80.0% | 80.0% |
| `The answer is B.` | **0.0%** | 80.0% |
| `**B**` | **0.0%** | 80.0% |
| a different format each time | 16.8% | 81.8% |

A brittle answer extractor reports **0%** for a model that answered 80% correctly. This is a real
reason published scores for the same model disagree between labs. Always log the unparsed rate.

### The gap is smaller than the noise

Model A truly scores 75%, model B truly scores 78%:

| Questions | A measured | B measured | p-value | Verdict |
|---|---|---|---|---|
| 50 | 82.0% | 76.0% | 0.461 | cannot tell — **A looks better** |
| 200 | 77.0% | 76.0% | 0.814 | cannot tell |
| 1,000 | 75.3% | 77.4% | 0.269 | cannot tell |
| 5,000 | 75.2% | 76.7% | 0.092 | cannot tell |
| 20,000 | 75.4% | 77.5% | 0.000 | B really is ahead |

On 50 questions the genuinely weaker model *won*. A 3-point true difference needs thousands of
questions to establish. Treat "model X beats Y by 1.2 points" as noise unless you see an interval.

### The model has seen the test

| Model | Public benchmark | Fresh questions | Gap |
|---|---|---|---|
| Honest (60% true ability) | 61.5% | 58.5% | +3.0% |
| Contaminated (30% true ability) | **100.0%** | **35.5%** | **+64.5%** |

**Contamination** means benchmark questions ended up in the training data, so the model recalls
rather than reasons. The weaker model looks perfect. The tell is a large gap between an old public
benchmark and fresh questions of similar difficulty — which is exactly why LiveBench and
LiveCodeBench refresh their questions, and why some benchmarks keep a private holdout.

### The setup moved the score

Number of few-shot examples, the chat template, whether chain-of-thought was allowed, temperature,
and how many attempts were averaged all move results by several points. Two labs can run "MMLU" and
report different numbers for the same model without either lying.

---

## 4. Reading a benchmark claim

A checklist for the next model announcement you read:

- [ ] **Which variant?** MMLU or MMLU-Pro? SWE-bench or SWE-bench Verified? GPQA or GPQA Diamond?
- [ ] **What is the random baseline**, and how far above it is this?
- [ ] **Is there an interval or error bar?** If not, assume small gaps are noise.
- [ ] **Was it few-shot or zero-shot**, with or without chain-of-thought?
- [ ] **Could the test be in the training data?** What is the model's cutoff date?
- [ ] **Who ran it?** A lab reporting its own model chooses the settings.
- [ ] **Does this benchmark resemble my task at all?**

---

## 5. What benchmarks cannot tell you

They cannot tell you whether the model works on **your** documents, in **your** tone, at **your**
latency budget, with **your** prompt. A model that tops every leaderboard can still be wrong for
you — too slow, too expensive, or weak in your language or domain.

Use benchmarks to narrow four candidates to two. Then measure the two on your own task.

---

## 6. Examples

```bash
python examples/benchmark_harness.py     # instant, no dependencies
```

[`benchmark_harness.py`](examples/benchmark_harness.py) builds a small multiple-choice benchmark and
demonstrates all four failure modes above against mock models whose true ability is known.

---

## 7. Exercises

1. **Guess the baseline.** For 4-option, 10-option and free-form-answer benchmarks, what score does
   a model that knows nothing get? Why does that make MMLU-Pro scores look lower than MMLU?
2. **Break the parser.** Add a new response format to `FORMATS` (e.g. `"I'd go with B"`). Does
   `robust_parse` still handle it? Fix it, and notice how easily a harness accumulates special cases.
3. **Find your own sample size.** Change the true accuracies to 0.70 and 0.72. How many questions
   are needed before the p-value drops below 0.05?
4. **Spot the contamination.** You see a model score 89% on MMLU and 41% on MMLU-Pro. List two
   innocent explanations and one worrying one. What would you check?
5. **Read a real model card.** Open a recent model announcement and run it through the section 4
   checklist. How many boxes can you actually tick?

---

## 8. Projects to build and test

### Beginner — Run a real benchmark
Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to score a small
model on one task.

**How to test it:** compare your number against the published one for the same model. If it differs,
find out why — few-shot count, template, or answer parsing. That investigation is the lesson.

### Intermediate — A contamination check
Take a public benchmark, write 30 fresh questions in the same style and difficulty, and compare a
model's score on both.

**How to test it:** have someone else confirm your new questions are comparable in difficulty —
otherwise a gap proves nothing.

### Advanced — A private benchmark for your domain
Build 200 questions from your own domain that no public dataset contains, with a documented scoring
method.

**How to test it:** score three models and report accuracy with confidence intervals. Then check
whether the ranking matches public leaderboards — and explain any difference.

---

## 9. Resources

**Start here**
- [LLM Benchmarks Compared: MMLU, HumanEval, GSM8K and More](https://www.lxt.ai/blog/llm-benchmarks/) — a plain-language tour.
- [LLM Benchmark Methodology: contamination and reading leaderboards](https://www.digitalapplied.com/blog/llm-benchmark-methodology-2026-contamination-leaderboard-guide) — how to read the numbers critically.

**Tools and leaderboards**
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — run standard benchmarks yourself
- [HELM](https://crfm.stanford.edu/helm/) — Stanford's holistic evaluation
- [Inspect](https://inspect.aisi.org.uk/) — UK AI Security Institute's framework, 200+ benchmarks
- [LMArena](https://lmarena.ai) — human preference rankings

**Papers**
- [MMLU-Pro](https://arxiv.org/abs/2406.01574) · [GPQA](https://arxiv.org/abs/2311.12022) · [IFEval](https://arxiv.org/abs/2311.07911) · [SWE-bench](https://arxiv.org/abs/2310.06770)
- [LiveBench](https://arxiv.org/abs/2406.19314) · [LiveCodeBench](https://arxiv.org/abs/2403.07974) — contamination-resistant by design
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) — ranking models from pairwise human votes

---

**Track:** [05 · Evaluation](../) · **Next:** [02 · Hallucination](../02-hallucination/)
