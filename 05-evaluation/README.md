# 05 · Evaluation

> **In one sentence:** evaluation is how you find out whether your system works — and, more often,
> whether the change you just made improved it or quietly broke something.

Everything before this track makes a model produce output. This track is about the only question
that decides whether any of it was worth doing: **is it actually any good?**

It is also the part teams most often skip, and the one that separates a demo from a product.

---

## 1. Why this is hard

Ordinary software has tests: given this input, expect exactly this output. LLMs break that in three
ways:

| Problem | What it means |
|---|---|
| **No single right answer** | Twenty different summaries can all be good. You cannot compare against one gold string. |
| **Non-determinism** | The same prompt gives different output each run, so `assert output == expected` fails constantly. |
| **Quality is multi-dimensional** | An answer can be accurate but rude, fluent but invented, correct but 10× too slow. |

So evaluation moves from *exact matching* to **scoring properties** and **comparing distributions** —
and that requires statistics, a labelled dataset, and a habit of looking at real outputs.

---

## 2. The three questions, and the three modules

| Question | Module | What it gives you |
|---|---|---|
| "Which model should I start with?" | [01 · Benchmarks](01-benchmarks/) | Public scores — useful for shortlisting, easy to over-trust |
| "Is it making things up?" | [02 · Hallucination](02-hallucination/) | Ways to detect and reduce unsupported output |
| "Is my system getting better?" | [03 · Model Evaluation](03-model-evaluation/) | An eval set and pipeline for **your** task — the one that actually matters |

**If you only do one, do 03.** A benchmark measures a benchmark; your users experience your task.

---

## 3. The evaluation ladder

Climb only as far as you need. Each rung costs more and is less reliable than the one above it.

```
  cheap, totally reliable
       |
  1. DETERMINISTIC CHECKS      valid JSON? required keys? forbidden content?
       |                       code decides. free, instant, never flaky.
       |
  2. REFERENCE-BASED SCORING   compare against a known answer: exact match,
       |                       numeric match, embedding similarity, pass@k for code
       |
  3. LLM-AS-JUDGE              a model scores quality against a rubric.
       |                       needed for tone, helpfulness, faithfulness.
       |                       MUST be validated against human labels first.
       |
  4. HUMAN REVIEW              the ground truth. slow, expensive, irreplaceable.
       |
  expensive, but the standard everything else is measured against
```

Most teams jump straight to rung 3 and skip rung 1. A surprising share of production failures are
caught by "is this valid JSON with the required keys", which costs nothing.

---

## 4. The vocabulary

| Term | Meaning |
|---|---|
| **Accuracy** | Share of answers correct. Only meaningful next to the random baseline. |
| **Precision / recall** | Of the things you flagged, how many were right (precision); of the things you should have flagged, how many did you catch (recall). |
| **Exact match / F1** | String-level scoring for short answers. |
| **pass@k** | For code: does at least one of k attempts pass the tests? |
| **Win rate** | In A-vs-B comparisons, how often B is preferred. |
| **Elo / Bradley-Terry** | Turning many pairwise votes into a single ranking — how [LMArena](https://lmarena.ai) ranks models. |
| **Cohen's kappa** | Agreement between two raters, corrected for agreement by chance. Use it to check a judge against humans. |
| **Faithfulness / groundedness** | Is every claim supported by the retrieved context? The key RAG metric. |
| **Refusal accuracy** | Does it say "I don't know" exactly when it should? |
| **Confidence interval** | The range the true score plausibly lies in. Report it, always. |

---

## 5. The rules that matter most

1. **Measure before you change.** Without a baseline, "better" is an opinion.
2. **Look at your data by hand.** Read 50 real outputs before designing any metric. Error analysis
   finds problems no metric was written to catch.
3. **Prefer binary pass/fail** to a 1–10 quality score. "Did it cite a real source?" is answerable;
   "how good is this, out of 10?" is not, and neither humans nor judges are consistent at it.
4. **Validate your judge.** An LLM judge you have not compared against human labels is an opinion
   of unknown quality, and you will optimise towards its biases.
5. **Report intervals.** A 3-point difference on 100 examples is noise — the
   [benchmark script](01-benchmarks/examples/benchmark_harness.py) shows exactly how much.
6. **Keep the eval set and grow it.** Every production failure becomes a test case. Models,
   prompts and frameworks change; a labelled dataset of your real failures keeps its value.
7. **Track cost and latency alongside quality.** A change that improves accuracy 2% and triples the
   bill is not obviously an improvement.

---

## 6. Common mistakes

| Mistake | Why it hurts |
|---|---|
| Evaluating on the examples you used to write the prompt | You measure memorisation of your own test |
| Judging by a handful of hand-picked prompts | The sample is too small to mean anything |
| One run per test case | Non-determinism turns noise into a "result" |
| Only easy cases in the eval set | Everything scores 100% and you learn nothing |
| Trusting a public leaderboard for your task | Benchmarks are contaminated and rarely resemble your workload |
| Same model as judge and candidate | Models favour their own style |
| Never re-checking the judge | A model or prompt update silently shifts it |
| Measuring quality but not cost or latency | You ship a regression in a dimension you weren't watching |

---

## 7. The tools

| Tool | For |
|---|---|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Running standard academic benchmarks; the backend of Hugging Face's Open LLM Leaderboard |
| [HELM](https://crfm.stanford.edu/helm/) | Stanford's broad, multi-metric evaluation suite |
| [Inspect](https://inspect.aisi.org.uk/) | Evaluation framework from the UK AI Security Institute; 200+ built-in benchmarks, agents and scorers |
| [Ragas](https://docs.ragas.io/) | RAG-specific metrics: faithfulness, answer relevance, context precision/recall |
| [DeepEval](https://github.com/confident-ai/deepeval) | Pytest-style LLM testing for application code |
| [promptfoo](https://promptfoo.dev/) | Prompt/model comparison plus red-teaming and CI integration |
| [LMArena](https://lmarena.ai) | Human pairwise preference rankings across models |

---

## 8. Examples

Three scripts, no dependencies and no API keys — each simulates models whose true quality is known,
so you can see how badly (or well) the measurement recovers it:

```bash
python 01-benchmarks/examples/benchmark_harness.py
python 02-hallucination/examples/hallucination_detection.py
python 03-model-evaluation/examples/eval_harness.py
```

| Script | Shows |
|---|---|
| [`benchmark_harness.py`](01-benchmarks/examples/benchmark_harness.py) | Answer parsing, sample size and significance, and a contaminated model scoring 100% on a public set and 35% on fresh questions |
| [`hallucination_detection.py`](02-hallucination/examples/hallucination_detection.py) | Self-consistency detection, threshold trade-offs, RAG groundedness — and where each method fails |
| [`eval_harness.py`](03-model-evaluation/examples/eval_harness.py) | Deterministic checks, judge position bias (46% of verdicts flip), validating a judge with Cohen's kappa, and bootstrap significance |

---

## 9. Exercises

1. **Rank the ladder.** For each of these, say which rung of section 3 you would use: "is the output
   valid JSON", "is the summary faithful to the article", "did it solve the maths problem", "is the
   tone appropriate for a complaint".
2. **Find the baseline.** A model scores 55% on a 4-option benchmark and 55% on a 10-option one.
   Which result is more impressive, and by how much?
3. **Design an eval set.** For a customer-support assistant, list 10 test cases covering: typical,
   edge, adversarial, and should-refuse. That mix matters more than the count.
4. **Catch yourself.** Take a prompt you have written and list three ways you could convince
   yourself it works when it does not.

## 10. Projects

- **Beginner — A 50-case eval set.** Build one for a task you care about, with binary pass/fail
  criteria and one deterministic check per case. *Test:* run it against two different models and
  report the gap with a confidence interval.
- **Intermediate — A judge you can trust.** Write an LLM judge for one quality dimension, label 50
  examples by hand, and measure agreement. *Test:* report kappa before and after improving the
  rubric; if kappa stays below 0.4, the rubric is the problem.
- **Advanced — Evaluation in CI.** Wire your eval set into a build that fails on regression.
  *Test:* deliberately degrade a prompt and confirm the build fails; confirm a harmless refactor
  does not.

---

## 11. Resources

**Start here**
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — Hamel Husain. The most practical article on this subject; the three-level framework (unit tests, human & model eval, A/B) is the backbone of module 03.
- [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/) — Eugene Yan. A survey of ~24 papers on LLM judges, with a decision tree.

**Go deeper**
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — the paper that established both the method and its biases.
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)
- [HELM](https://crfm.stanford.edu/helm/) · [Inspect](https://inspect.aisi.org.uk/) · [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

---

**Previous track:** [04 · Serving](../04-serving/) · **Next track:** [06 · RAG](../06-rag/)
