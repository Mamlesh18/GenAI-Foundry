"""
"Is version B better than version A?" -- how to answer that honestly.

This is the question every LLM project asks weekly, and the one most teams
answer by trying three prompts and forming an impression. This script builds
the machinery to answer it with evidence:

  1. a golden dataset and deterministic checks (free, instant, catch most bugs)
  2. an LLM judge for what code cannot check -- WITH its biases measured
  3. agreement with human labels (Cohen's kappa), because an unvalidated judge
     is just a second opinion you have not checked
  4. a bootstrap confidence interval, because a 5-point win on 60 examples is
     usually noise

The systems and the judge are simulated, so their TRUE quality is known and we
can see how well the measurement recovers it.

No dependencies.   python eval_harness.py
"""

import json
import random
import re

SEED = 7


# ---------------------------------------------------------------------------
# 1. Deterministic checks -- always start here
# ---------------------------------------------------------------------------

def check_valid_json(output):
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def check_has_keys(output, keys=("sentiment", "confidence")):
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False
    return all(k in data for k in keys)


def check_no_pii(output):
    return not re.search(r"\b\d{3}-\d{2}-\d{4}\b|\b[\w.]+@[\w.]+\.\w+\b", output or "")


DETERMINISTIC_CHECKS = {
    "valid JSON": check_valid_json,
    "required keys": check_has_keys,
    "no PII leaked": check_no_pii,
}


def demo_deterministic():
    print("=" * 78)
    print("1. DETERMINISTIC CHECKS -- free, instant, and catch the worst bugs")
    print("=" * 78)
    outputs = {
        "good": '{"sentiment": "positive", "confidence": 0.9}',
        "prose, not JSON": "The sentiment is positive.",
        "missing a key": '{"sentiment": "positive"}',
        "leaks an email": '{"sentiment": "positive", "confidence": 0.9, '
                          '"note": "contact bob@example.com"}',
    }
    print(f"  {'output':<20} " + " ".join(f"{name:>14}" for name in DETERMINISTIC_CHECKS))
    print("  " + "-" * 72)
    for label, output in outputs.items():
        marks = " ".join(f"{('PASS' if check(output) else 'FAIL'):>14}"
                         for check in DETERMINISTIC_CHECKS.values())
        print(f"  {label:<20} {marks}")
    print(
        "\n  Run these on every output, on every change. They cost nothing, they never\n"
        "  disagree with themselves, and in most projects they catch more real defects\n"
        "  than any clever quality score. Only reach for a judge when code cannot decide."
    )


# ---------------------------------------------------------------------------
# 2. A simulated A/B comparison with a biased judge
# ---------------------------------------------------------------------------

class Response:
    def __init__(self, system, quality, length):
        self.system = system
        self.quality = quality          # hidden truth, unknown in real life
        self.length = length


def make_eval_set(n, seed=SEED):
    """n prompts, each answered by system A and system B. B is genuinely a
    little better; A tends to write longer answers."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        a = Response("A", rng.gauss(0.50, 0.15), rng.randint(150, 400))
        b = Response("B", rng.gauss(0.58, 0.15), rng.randint(80, 200))
        items.append({"id": i, "A": a, "B": b})
    return items


def human_label(item, rng):
    """A careful human, who is not perfect either."""
    gap = item["B"].quality - item["A"].quality
    if rng.random() < 0.05:                       # occasional slip
        return rng.choice(["A", "B"])
    return "B" if gap > 0 else "A"


class MockJudge:
    """An LLM judge with the biases measured in the literature."""

    def __init__(self, position_bias=0.15, verbosity_bias=0.20, noise=0.10, seed=SEED):
        self.position_bias = position_bias      # prefers whatever it reads first
        self.verbosity_bias = verbosity_bias    # prefers the longer answer
        self.noise = noise
        self.rng = random.Random(seed)

    def compare(self, first, second):
        """Given two responses IN ORDER, return the winner's system name."""
        score = second.quality - first.quality            # the real difference
        score -= self.position_bias                        # tilt towards `first`
        length_gap = (second.length - first.length) / 400
        score += self.verbosity_bias * length_gap          # tilt towards longer
        score += self.rng.gauss(0, self.noise)
        return second.system if score > 0 else first.system


def demo_judge_bias():
    print("\n" + "=" * 78)
    print("2. AN LLM JUDGE, AND THE BIASES IT ARRIVES WITH")
    print("=" * 78)
    items = make_eval_set(400)
    judge = MockJudge()

    a_first = sum(1 for it in items if judge.compare(it["A"], it["B"]) == "A")
    b_first = sum(1 for it in items if judge.compare(it["B"], it["A"]) == "A")
    n = len(items)

    print(f"  The same {n} comparisons, judged twice -- only the ORDER changes:\n")
    print(f"    A shown first  -> A wins {a_first / n:.0%} of the time")
    print(f"    A shown second -> A wins {b_first / n:.0%} of the time")
    print(f"    swing from position alone: {abs(a_first - b_first) / n:.0%}")

    flips = sum(1 for it in items
                if judge.compare(it["A"], it["B"]) != judge.compare(it["B"], it["A"]))
    print(f"\n    verdict changed on {flips / n:.0%} of pairs when the order was swapped")
    print(
        "\n  POSITION BIAS. The fix is simple and non-negotiable: judge every pair\n"
        "  BOTH ways and average. If the two runs disagree, the pair is a tie.\n\n"
        "  VERBOSITY BIAS is in this judge too -- it rewards longer answers\n"
        "  regardless of content. Since system A writes longer answers here, the\n"
        "  judge flatters A even though B is genuinely better. Watch for it by\n"
        "  reporting average response length next to every win rate."
    )


def judge_both_ways(judge, item):
    """Debiased comparison: run both orders, tie if they disagree."""
    first = judge.compare(item["A"], item["B"])
    second = judge.compare(item["B"], item["A"])
    return first if first == second else "tie"


def cohens_kappa(labels_a, labels_b):
    """Agreement corrected for the agreement you would get by chance."""
    n = len(labels_a)
    observed = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    categories = set(labels_a) | set(labels_b)
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0, observed


def demo_validate_judge():
    print("\n" + "=" * 78)
    print("3. DOES THE JUDGE AGREE WITH A HUMAN?")
    print("=" * 78)
    items = make_eval_set(300)
    rng = random.Random(99)
    humans = [human_label(it, rng) for it in items]
    judge = MockJudge()

    naive = [judge.compare(it["A"], it["B"]) for it in items]
    debiased = [judge_both_ways(judge, it) for it in items]

    k_naive, agree_naive = cohens_kappa(humans, naive)
    # A tie means the judge abstained, so score it only on the pairs it decided
    # -- and report how many those were.
    decided = [(h, j) for h, j in zip(humans, debiased) if j != "tie"]
    k_decided, agree_decided = cohens_kappa([h for h, _ in decided], [j for _, j in decided])
    coverage = len(decided) / len(items)

    print(f"  {'judge setup':<32} {'decided':>9} {'agreement':>11} {'kappa':>8}")
    print("  " + "-" * 64)
    print(f"  {'one order only (naive)':<32} {1.0:>8.0%} {agree_naive:>10.0%} {k_naive:>8.2f}")
    print(f"  {'both orders; ties = abstain':<32} {coverage:>8.0%} {agree_decided:>10.0%}"
          f" {k_decided:>8.2f}")
    print(
        f"\n  Judging both ways makes the judge abstain on {1 - coverage:.0%} of pairs -- exactly\n"
        "  those where its verdict flipped when the order changed. On the pairs it\n"
        "  still calls, it agrees with the human more often and its kappa is higher.\n"
        "  You trade coverage for trustworthiness, and the abstentions are useful in\n"
        "  themselves: they are the genuinely close calls, worth a human's time."
    )
    print(
        "\n  Kappa corrects for agreeing by luck. On a two-way choice, guessing gets\n"
        "  you ~50% raw agreement and kappa 0. Rough reading: below 0.4 is weak,\n"
        "  0.4-0.6 moderate, above 0.6 good.\n\n"
        "  THE RULE: label 30-50 examples by hand and measure this BEFORE you trust a\n"
        "  judge to make decisions. An unvalidated judge is an opinion with a\n"
        "  confidence interval of unknown width -- and you will optimise towards its\n"
        "  biases rather than towards quality."
    )


# ---------------------------------------------------------------------------
# 4. Is the difference real?
# ---------------------------------------------------------------------------

def bootstrap_ci(wins, n, iterations=4000, seed=SEED):
    """Resample the eval set to get a confidence interval on the win rate."""
    rng = random.Random(seed)
    outcomes = [1] * wins + [0] * (n - wins)
    rates = []
    for _ in range(iterations):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    return rates[int(0.025 * iterations)], rates[int(0.975 * iterations)]


def demo_significance():
    print("\n" + "=" * 78)
    print("4. IS THE DIFFERENCE REAL? -- eval set size decides")
    print("=" * 78)
    print("  System B is truly better. We compare on eval sets of different sizes")
    print("  and ask whether B's win rate is distinguishable from a coin flip.\n")
    print(f"  {'eval set':>9} {'B win rate':>12} {'95% CI':>20}   conclusion")
    print("  " + "-" * 68)

    rng = random.Random(123)
    for n in (20, 60, 200, 1000):
        items = make_eval_set(n, seed=n)
        wins = sum(1 for it in items if human_label(it, rng) == "B")
        lo, hi = bootstrap_ci(wins, n)
        conclusion = "B is better" if lo > 0.5 else "cannot tell yet"
        print(f"  {n:>9} {wins / n:>11.0%} {f'[{lo:.0%}, {hi:.0%}]':>20}   {conclusion}")

    print(
        "\n  With 20 examples the interval is enormous -- almost any result is\n"
        "  consistent with the two systems being equal. This is the arithmetic behind\n"
        "  'we tried a few prompts and B seemed better'.\n\n"
        "  Practical guidance: aim for a few hundred examples for a go/no-go decision,\n"
        "  keep the SAME set across versions so comparisons are paired, and report the\n"
        "  interval next to the number every time."
    )


def demo_pipeline():
    print("\n" + "=" * 78)
    print("PUTTING IT TOGETHER -- what a real eval pipeline looks like")
    print("=" * 78)
    print(
        "  On every change:\n"
        "    1. deterministic checks on all cases      (seconds, free)\n"
        "    2. reference-based scoring where answers are known\n"
        "    3. LLM judge, both orders, for the rest   (slow, costs money)\n"
        "    4. compare against the stored baseline, with intervals\n"
        "    5. fail the build on a regression beyond your threshold\n\n"
        "  Weekly:\n"
        "    6. read actual production failures by hand -- error analysis beats\n"
        "       any metric for finding out what is really wrong\n"
        "    7. add those failures to the golden set, so it grows over time\n\n"
        "  Monthly:\n"
        "    8. re-check judge-versus-human agreement; a model or prompt change can\n"
        "       silently move the judge.\n\n"
        "  The eval set is the asset. Prompts, models and frameworks will all change;\n"
        "  a good labelled dataset of your real failures keeps its value."
    )


def main():
    demo_deterministic()
    demo_judge_bias()
    demo_validate_judge()
    demo_significance()
    demo_pipeline()


if __name__ == "__main__":
    main()
