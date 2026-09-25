# 03 · Model Evaluation

> **In one sentence:** this is the eval system for *your* product — the set of test cases, checks
> and statistics that let you say "version B is better than version A" and be right.

Benchmarks compare models in general. This module is about the only comparison that pays: does your
system, on your task, get better or worse when you change something?

**If you read one module in this track, read this one.**

---

## 1. Start with error analysis, not metrics

The instinct is to pick metrics first. The better move, from
[Hamel Husain's evals guide](https://hamel.dev/blog/posts/evals/), is:

```
  1. LOOK AT DATA        read 50-100 real outputs, by hand. no tooling, no scores.
        |
  2. NAME THE FAILURES   group what you saw: "invents policy details",
        |                "ignores the word 'not'", "too formal for chat"
        |
  3. WRITE TEST CASES    turn each failure mode into concrete examples
        |
  4. CHOOSE CHECKS       for each: can code decide? a reference answer? a judge?
        |
  5. MEASURE AND ITERATE now you have a baseline, and changes are measurable
```

Metrics chosen before looking at data measure what you imagined, not what is broken. You cannot
skip step 1, and it stays valuable forever — schedule it weekly.

---

## 2. The eval set is the asset

| Property | Why |
|---|---|
| **Covers real failures** | Built from production traces, not imagination |
| **Mixed difficulty** | Typical, edge, adversarial, and should-refuse cases |
| **Binary criteria** | "Did it cite a real source?" beats "rate this 1–10" |
| **Stable** | The same set across versions, so comparisons are paired |
| **Growing** | Every new production failure is added |
| **Version controlled** | It is code; review changes to it like code |

Aim for 50 cases to start and a few hundred for go/no-go decisions — section 5 shows why.

---

## 3. Deterministic checks first

Cheap, instant, perfectly reliable, and they catch more real defects than people expect:

| Output | valid JSON | required keys | no PII leaked |
|---|---|---|---|
| `{"sentiment": "positive", "confidence": 0.9}` | PASS | PASS | PASS |
| `The sentiment is positive.` | FAIL | FAIL | PASS |
| `{"sentiment": "positive"}` | PASS | FAIL | PASS |
| `{..., "note": "contact bob@example.com"}` | PASS | PASS | FAIL |

Only reach for a judge when code genuinely cannot decide.

---

## 4. LLM-as-a-judge, done properly

For tone, helpfulness and faithfulness you need a model to score. It works — but only with
precautions, and [`eval_harness.py`](examples/eval_harness.py) measures why.

### Direct scoring or pairwise?

| Approach | Ask | Good for |
|---|---|---|
| **Direct scoring** | "Does this answer satisfy the rubric? pass/fail" | objective criteria, regression testing |
| **Pairwise** | "Which of these two is better?" | subjective quality; humans and judges are far more consistent at comparing than at scoring |

### The biases are large

The script's mock judge carries the biases measured in the literature. The same 400 comparisons,
judged twice with only the **order** changed:

- A shown **first** → A wins **69%**
- A shown **second** → A wins **23%**
- **46% of verdicts flipped** purely because of position

| Bias | What it is | Fix |
|---|---|---|
| **Position** | Prefers whichever answer it reads first | Judge both orders; disagreement = tie |
| **Verbosity** | Prefers longer answers regardless of content | Report average length beside every win rate; say "do not prefer longer answers" in the rubric |
| **Self-preference** | Models favour their own style | Never use the same model as judge and candidate |

### Validate the judge against humans

Agreement with hand labels, before and after debiasing:

| Judge setup | Pairs decided | Agreement | Cohen's kappa |
|---|---|---|---|
| One order only (naive) | 100% | 62% | 0.33 |
| **Both orders; ties = abstain** | 51% | **92%** | **0.83** |

Judging both ways makes the judge abstain on the half of pairs where its verdict flipped — and on
the pairs it still decides, agreement jumps from 62% to 92%. You trade coverage for
trustworthiness, and the abstentions are useful: they are the close calls, worth a human's time.

**Kappa** corrects for agreeing by chance (a coin flip scores ~50% raw agreement and kappa 0).
Rough reading: below 0.4 weak, 0.4–0.6 moderate, above 0.6 good. Label 30–50 examples yourself and
measure this **before** letting a judge make decisions.

---

## 5. Is the difference real?

System B is genuinely better. Can you tell, at each eval set size?

| Eval set | B win rate | 95% CI | Conclusion |
|---|---|---|---|
| 20 | 55% | [35%, 75%] | cannot tell |
| 60 | 53% | [42%, 67%] | cannot tell |
| 200 | 58% | [52%, 66%] | B is better |
| 1,000 | 61% | [58%, 64%] | B is better |

With 20 examples almost any result is consistent with the systems being equal. That is the
arithmetic behind "we tried a few prompts and B seemed better".

**Practical rules:** use the same cases across versions (paired comparison is far more sensitive),
aim for a few hundred cases for a real decision, run each case more than once if your temperature is
above 0, and report the interval next to the number every time.

---

## 6. The pipeline

```
ON EVERY CHANGE (minutes, in CI)
  1. deterministic checks on all cases
  2. reference-based scoring where answers are known
  3. LLM judge, both orders, for the rest
  4. compare against the stored baseline, with intervals
  5. fail the build on regression beyond your threshold

WEEKLY (an hour, by hand)
  6. read real production failures -- error analysis
  7. add them to the golden set

MONTHLY
  8. re-check judge-versus-human agreement; a model update can move the judge
```

### Once it is live

| Signal | What it tells you |
|---|---|
| Thumbs up/down, regenerations, edits | Real quality, cheaply — regenerations are an underrated failure signal |
| A/B tests | Whether a change helps real users, not just your eval set |
| Guardrail trips | How often outputs fail validation in production |
| Cost and latency per request | The dimensions people forget until the bill arrives |
| Drift | The same eval set re-run weekly; provider models change under you |

---

## 7. Anti-patterns

| Anti-pattern | Why it hurts |
|---|---|
| Testing on the examples used to write the prompt | You measure memorisation of your own test |
| Scores out of 10 | Neither humans nor judges are consistent; use binary criteria |
| Judge never validated | You optimise towards the judge's biases |
| Judging in one order only | 46% of verdicts can flip on position alone |
| Same model judging its own output | Self-preference inflates the score |
| Eval set never grows | You stop catching new failure modes |
| One run per case | Non-determinism becomes a "result" |
| Quality tracked, cost ignored | You ship a 3× bill for a 2% gain |

---

## 8. Examples

```bash
python examples/eval_harness.py     # instant, no dependencies
```

[`eval_harness.py`](examples/eval_harness.py) builds the whole thing in miniature: deterministic
checks, a mock judge with measurable position and verbosity bias, judge validation via Cohen's
kappa, and bootstrap confidence intervals for an A/B comparison.

---

## 9. Exercises

1. **Do the error analysis.** Take any LLM feature you have used and write down five distinct ways
   it fails. Group them. Which would a metric have caught, and which needed a human to notice?
2. **Turn failures into tests.** Convert three of those into concrete test cases with binary pass
   criteria.
3. **Measure your own consistency.** Rate 20 outputs 1–10, wait a day, rate them again. How often do
   you agree with yourself? Now do the same with pass/fail.
4. **Fix the judge.** In `eval_harness.py`, set `verbosity_bias=0` and rerun. How much of the
   judge's disagreement with humans came from length alone?
5. **Find your sample size.** Change B's true quality so the systems are nearly equal. How large
   must the eval set be before the difference is detectable? What does that tell you about
   chasing 1% improvements?
6. **Build the gate.** Write a script that fails with exit code 1 when accuracy drops more than 5%
   below a stored baseline, and wire it into a pre-commit hook or CI job.

---

## 10. Projects to build and test

### Beginner — Golden dataset and runner
50 cases for a real task, with binary criteria, and one command that scores everything and prints a
table.

**How to test it:** run it twice with no changes — the score should be stable (or you have a
determinism problem worth knowing about). Then change the prompt and confirm the score moves.

### Intermediate — A validated judge
Write a judge for one dimension, hand-label 50 examples, and iterate on the rubric until kappa
exceeds 0.6.

**How to test it:** report kappa after each rubric revision. Also run each pair in both orders and
report how often the judge contradicts itself — if that is high, the rubric is ambiguous.

### Advanced — Full eval pipeline in CI
Version-controlled prompts, an eval set, deterministic + judge scoring, baseline comparison with
intervals, and a build that fails on regression.

**How to test it:** deliberately degrade a prompt (remove the format constraint) and confirm the
build fails with a useful message. Confirm an unrelated refactor passes. Then add a production
failure to the set and watch the baseline update.

---

## 11. Resources

**Start here**
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — Hamel Husain. The three-level framework and the case for error analysis. Read it twice.
- [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/) — Eugene Yan. When to use direct scoring vs pairwise, and how to validate.
- [Evals, error analysis, and better prompts](https://www.lennysnewsletter.com/p/evals-error-analysis-and-better-prompts) — a practical interview-format walkthrough.

**Go deeper**
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — the foundational paper, including the bias measurements.
- [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926) — position bias, measured and mitigated.
- [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge](https://arxiv.org/abs/2410.02736) · [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/abs/2410.21819)

**Tools**
- [promptfoo](https://promptfoo.dev/) — compare prompts and models, run in CI
- [DeepEval](https://github.com/confident-ai/deepeval) — pytest-style assertions for LLM output
- [Ragas](https://docs.ragas.io/) — RAG metrics · [Inspect](https://inspect.aisi.org.uk/) — full evaluation framework

---

**Previous:** [02 · Hallucination](../02-hallucination/) · **Track:** [05 · Evaluation](../) ·
**Next track:** [06 · RAG](../../06-rag/)
