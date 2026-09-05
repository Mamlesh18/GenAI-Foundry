# 08 · Evaluation & Testing

**Level:** Intermediate · **Time:** ~5 hours

---

## 1. What you will be able to do

Answer the question "did that change make it better?" with evidence instead of an opinion — for
systems whose output is different every time you run them.

Do this skill earlier than feels natural. Everything before it is guesswork without it.

---

## 2. Prerequisites

- Skills: [02 · Prompt Engineering](../02-prompt-engineering/), and something worth evaluating
- Concepts: [05 · Evaluation](../../05-evaluation/)

---

## 3. The concept, briefly

Traditional tests assert exact outputs. LLM outputs are non-deterministic, so that fails
immediately. You need to assert on **properties** and **aggregate scores** instead.

Three tiers, cheapest first — use all three:

| Tier | What it checks | Cost | Reliability |
|---|---|---|---|
| **Deterministic** | Valid JSON, required keys, length bounds, contains a required term, no PII | free, instant | total |
| **Reference-based** | Match against a known correct answer — exact, fuzzy, or embedding similarity | cheap | high, needs labels |
| **LLM-as-judge** | A model scores quality against a rubric | slow, costs money | moderate, needs its own validation |

Most people jump to LLM-as-judge. Start at the top: a shocking share of production failures are
caught by "is this valid JSON with the required keys", which costs nothing.

**On LLM-as-judge specifically:** it is useful and it is not free of bias. Judges prefer longer
answers, prefer their own model family's style, and are sensitive to option order. Before trusting
one, check its agreement with your own labels on 30 examples. If it agrees less than ~80% of the
time, fix the rubric before you use the judge to make decisions.

---

## 4. Build it

1. **Build a golden dataset.** 50+ cases: typical, edge, adversarial, and should-refuse. Store it as
   version-controlled JSON. This is the asset; everything else is a script.

2. **Add deterministic checks.** Schema validity, required fields, length bounds, forbidden content,
   latency ceiling. Run them on everything. They will catch more than you expect.

3. **Add reference-based scoring** where you have known answers. Exact match for classification,
   embedding similarity for free text.

4. **Build the runner.** One command: run all cases, score them, output a report with per-category
   breakdowns. If running the evaluation is any harder than one command, you will stop doing it.

5. **Record a baseline.** Commit the results. Every future claim of improvement is measured against
   this.

6. **Add an LLM judge** for what the first two tiers cannot capture — helpfulness, tone,
   faithfulness to context. Give it a specific rubric with a defined scale, not "rate this 1–10".

7. **Validate the judge.** Score 30 examples yourself. Compare. Below ~80% agreement, the rubric is
   the problem, not the judge.

8. **Measure variance.** Run the same case 5 times. High variance means your metric is noise and any
   "improvement" under that threshold is imaginary.

9. **Wire it into CI.** Run on every prompt or model change. Fail the build on regression beyond a
   threshold. Prompts are code; treat them that way.

10. **Add production monitoring.** Log inputs, outputs, latency, cost and user feedback. Sample real
    traffic weekly and feed the interesting failures back into the golden dataset. Your test set
    should grow every month.

---

## 5. Checkpoint

- [ ] A golden dataset of 50+ cases in version control
- [ ] One command runs everything and prints a scored report
- [ ] Deterministic, reference-based and judge scores all present
- [ ] Judge agreement with your own labels measured and above 80%
- [ ] Run-to-run variance measured, so you know what counts as a real change
- [ ] A deliberate regression (degrade a prompt) is caught automatically
- [ ] CI blocks a merge on regression
- [ ] You can say "version B is X% better on Y, at Z% of the cost" with real numbers

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Evaluating by vibes | Months of work with no idea whether it improved | Build the dataset first |
| Asserting exact strings | Tests fail constantly for no reason | Assert properties |
| Only easy cases | Everything scores 100%, nothing is learned | Include adversarial and should-refuse cases |
| Trusting an unvalidated judge | You optimise toward the judge's biases | Check agreement with your labels |
| One run per case | Noise mistaken for signal | 3–5 runs, look at variance |
| Test set never grows | You stop catching new failure modes | Feed production failures back in |
| No cost or latency tracking | Quality improves, the bill quietly triples | Track all three together |
| Judge and system are the same model | It rates its own style highly | Use a different model to judge |

---

## 7. Going further

- **Next skill:** [09 · Shipping to Production](../09-shipping-to-production/)
- Add A/B testing on live traffic, and learn how long you must run it to reach significance
- Try [Ragas](https://docs.ragas.io/) for RAG-specific metrics, [DeepEval](https://github.com/confident-ai/deepeval) for general ones
- Read about pairwise comparison — humans and judges are far more consistent at "which is better"
  than at "score this out of 10"

**Links:** [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) ·
[Ragas](https://docs.ragas.io/) ·
[HELM](https://crfm.stanford.edu/helm/) ·
[LMSYS Chatbot Arena](https://lmarena.ai/)
