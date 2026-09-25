"""
A benchmark harness in miniature -- and the four ways benchmark numbers lie.

Running a benchmark looks trivial: ask questions, count correct answers. This
script builds that, then demonstrates why two labs can run "the same" benchmark
on the same model and publish different scores:

  1. ANSWER PARSING   the model was right; your regex said it was wrong
  2. SAMPLE SIZE      a 3-point gap on 100 questions is usually noise
  3. CONTAMINATION    the model memorised the test set
  4. PROMPT FORMAT    changing the template moves the score

Everything is simulated with mock models, so it runs instantly and the "true"
ability of each model is known -- which is what lets us show measurement error.

No dependencies.   python benchmark_harness.py
"""

import math
import random
import re

# ---------------------------------------------------------------------------
# A tiny multiple-choice benchmark (the shape of MMLU, GPQA, ARC, ...)
# ---------------------------------------------------------------------------

LETTERS = "ABCD"


def make_dataset(n, seed):
    """Synthetic questions. Content does not matter -- only that there is one
    correct option out of four, as in every multiple-choice benchmark."""
    rng = random.Random(seed)
    return [{"id": f"s{seed}-q{i}",
             "question": f"Synthetic question {i} (set {seed})",
             "answer": rng.choice(LETTERS)}
            for i in range(n)]


# ---------------------------------------------------------------------------
# Mock models. We KNOW their true ability, so we can see how badly we measure it.
# ---------------------------------------------------------------------------

# Real models answer in wildly different shapes. A benchmark harness has to
# turn all of these into a single letter.
FORMATS = [
    "{letter}",
    "({letter})",
    "The answer is {letter}.",
    "**{letter}**",
    "Answer: {letter}",
    "I think the correct option is {letter}, because ...",
]


class MockModel:
    def __init__(self, name, true_accuracy, format_style, memorised=(), seed=0):
        self.name = name
        self.true_accuracy = true_accuracy
        self.format_style = format_style
        self.memorised = set(memorised)      # ids whose answers it has seen
        self.rng = random.Random(seed)

    def answer(self, item):
        if item["id"] in self.memorised:     # contamination: it just recalls
            letter = item["answer"]
        elif self.rng.random() < self.true_accuracy:
            letter = item["answer"]
        else:                                 # wrong, but plausibly wrong
            letter = self.rng.choice([c for c in LETTERS if c != item["answer"]])
        template = (self.rng.choice(FORMATS) if self.format_style == "varied"
                    else self.format_style)
        return template.format(letter=letter)


# ---------------------------------------------------------------------------
# Two parsers: one brittle, one careful
# ---------------------------------------------------------------------------

def strict_parse(response):
    """Accept only a bare letter. This is roughly what a first draft looks like."""
    text = response.strip()
    return text if text in LETTERS else None


def robust_parse(response):
    """Look for a standalone A-D, allowing the usual decorations."""
    patterns = [
        r"\banswer\s*(?:is)?\s*[:\-]?\s*\(?([A-D])\)?",   # "Answer: B", "answer is B"
        r"\*\*\(?([A-D])\)?\*\*",                          # "**B**"
        r"^\s*\(?([A-D])\)?[\.\)]?\s*$",                   # "B", "(B)", "B."
        r"\boption\s+\(?([A-D])\)?",                       # "option B"
        r"\b([A-D])\b",                                    # last resort
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def evaluate(model, dataset, parser):
    correct = unparsed = 0
    for item in dataset:
        predicted = parser(model.answer(item))
        if predicted is None:
            unparsed += 1                    # counted as wrong, like real harnesses
        elif predicted == item["answer"]:
            correct += 1
    return correct / len(dataset), unparsed


# ---------------------------------------------------------------------------
# Statistics: how much of a gap is real?
# ---------------------------------------------------------------------------

def wilson_interval(correct, n, z=1.96):
    """95% confidence interval for an accuracy. Better than the textbook
    normal approximation at small n and near 0 or 1."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def two_proportion_p(correct_a, n_a, correct_b, n_b):
    """Two-sided p-value for 'these two accuracies are the same'.

    Overlapping confidence intervals are NOT the right test -- two intervals
    can overlap while the difference is still significant. This compares the
    difference directly.
    """
    p_a, p_b = correct_a / n_a, correct_b / n_b
    pooled = (correct_a + correct_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = abs(p_a - p_b) / se
    return math.erfc(z / math.sqrt(2))


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------

def main():
    dataset = make_dataset(400, seed=1)

    section("1. THE BASIC LOOP -- and the random baseline")
    strong = MockModel("strong", 0.80, "{letter}", seed=10)
    accuracy, _ = evaluate(strong, dataset, robust_parse)
    print(f"  4 options, {len(dataset)} questions.\n")
    print(f"  random guessing scores ~25%  (1 in 4)")
    print(f"  our 'strong' model scores    {accuracy:.1%}")
    print(
        "\n  ALWAYS compare against the random baseline. A model scoring 30% on a\n"
        "  4-option benchmark has learned almost nothing, even though 30% sounds\n"
        "  like partial credit. On a 10-option benchmark the baseline is 10%, which\n"
        "  is one reason MMLU-Pro (10 options) reports lower numbers than MMLU (4)."
    )

    section("2. ANSWER PARSING -- the same model, two scores")
    print("  Every model here has the SAME true ability (80%). Only the way it")
    print("  writes its answer differs.\n")
    print(f"  {'model writes...':<42} {'strict parser':>14} {'robust parser':>14}")
    print("  " + "-" * 72)
    for label, style in (("just the letter: 'B'", "{letter}"),
                         ("'The answer is B.'", "The answer is {letter}."),
                         ("'**B**'", "**{letter}**"),
                         ("a different format each time", "varied")):
        strict_acc, strict_bad = evaluate(
            MockModel("m", 0.80, style, seed=11), dataset, strict_parse)
        robust_acc, robust_bad = evaluate(
            MockModel("m", 0.80, style, seed=11), dataset, robust_parse)
        print(f"  {label:<42} {strict_acc:>13.1%} {robust_acc:>13.1%}")

    print(
        "\n  A brittle parser reports 0% for a model that got 80% of the answers\n"
        "  right. This is not hypothetical: differences in answer extraction and\n"
        "  prompt templates are a common reason published scores for the same model\n"
        "  disagree. When you see a benchmark number, ask HOW the answer was parsed.\n"
        "  Always log the unparsed rate -- here the strict parser failed on"
        f" {strict_bad}/{len(dataset)}\n  responses for the varied-format model."
    )

    section("3. SAMPLE SIZE -- is that 3-point gap real?")
    print("  Model A truly scores 75%, model B truly scores 78%. We measure both")
    print("  on benchmarks of different sizes and check whether we can tell.\n")
    print(f"  {'questions':>10} {'A':>8} {'B':>8} {'A 95% CI':>16} {'B 95% CI':>16} {'p-value':>9}   verdict")
    print("  " + "-" * 92)
    for n in (50, 200, 1000, 5000, 20000):
        subset = make_dataset(n, seed=2)
        a_acc, _ = evaluate(MockModel("A", 0.75, "{letter}", seed=20), subset, robust_parse)
        b_acc, _ = evaluate(MockModel("B", 0.78, "{letter}", seed=21), subset, robust_parse)
        a_correct, b_correct = round(a_acc * n), round(b_acc * n)
        a_lo, a_hi = wilson_interval(a_correct, n)
        b_lo, b_hi = wilson_interval(b_correct, n)
        p = two_proportion_p(a_correct, n, b_correct, n)
        verdict = "B really is ahead" if p < 0.05 else "cannot tell"
        print(f"  {n:>10} {a_acc:>7.1%} {b_acc:>7.1%} "
              f"{f'[{a_lo:.1%}, {a_hi:.1%}]':>16} {f'[{b_lo:.1%}, {b_hi:.1%}]':>16}"
              f" {p:>9.3f}   {verdict}")

    print(
        "\n  A 3-point true difference needs thousands of questions before it can be\n"
        "  told apart from luck -- and on a 50-question set, model A can easily look\n"
        "  BETTER than the model that is actually stronger.\n\n"
        "  At 5,000 questions the measured gap is still only half the true one and\n"
        "  the p-value says 'not yet'. Note also that two intervals can overlap while\n"
        "  the difference between them IS significant -- eyeballing error bars is not\n"
        "  the test. Compare the difference directly.\n\n"
        "  This is why 'model X beats model Y by 1.2 points' on a leaderboard usually\n"
        "  means nothing at all."
    )

    section("4. CONTAMINATION -- the model has seen the test")
    public = make_dataset(200, seed=3)          # published benchmark
    holdout = make_dataset(200, seed=4)         # fresh questions, same difficulty

    honest = MockModel("honest", 0.60, "{letter}", seed=30)
    cheat = MockModel("contaminated", 0.30, "{letter}",
                      memorised=[item["id"] for item in public], seed=31)

    print(f"  {'model':<16} {'public benchmark':>18} {'fresh questions':>18}   gap")
    print("  " + "-" * 62)
    for model in (honest, cheat):
        pub, _ = evaluate(model, public, robust_parse)
        new, _ = evaluate(model, holdout, robust_parse)
        print(f"  {model.name:<16} {pub:>17.1%} {new:>17.1%}   {pub - new:+.1%}")

    print(
        "\n  The contaminated model is WORSE than the honest one (30% true ability vs\n"
        "  60%) yet looks perfect on the public set, because its questions were in\n"
        "  the training data. The tell is the gap between a public benchmark and\n"
        "  fresh questions of the same difficulty.\n\n"
        "  This is why benchmarks like LiveBench and LiveCodeBench refresh their\n"
        "  questions, why some keep a private holdout, and why your own private\n"
        "  eval set is worth more to you than any public leaderboard."
    )

    section("WHAT TO TAKE AWAY")
    print(
        "  - Compare against the random baseline before being impressed.\n"
        "  - Publish (and read) confidence intervals, not bare point scores.\n"
        "  - Log parse failures; a brittle parser silently invents a bad model.\n"
        "  - Assume public benchmarks are contaminated; keep a private test set.\n"
        "  - A benchmark measures a benchmark. What matters is YOUR task --\n"
        "    see ../../03-model-evaluation/."
    )


if __name__ == "__main__":
    main()
