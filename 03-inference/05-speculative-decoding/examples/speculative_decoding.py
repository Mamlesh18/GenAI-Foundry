"""
Speculative decoding, from scratch -- and proof that it is LOSSLESS.

A big model is slow because each token needs a full forward pass. A small model
is fast but worse. Speculative decoding gets the big model's exact output at
closer to the small model's speed:

  1. DRAFT   the small model guesses the next k tokens (cheap)
  2. VERIFY  the big model checks all k guesses in ONE forward pass
  3. ACCEPT  guesses one by one; at the first rejection, replace it with a
             corrected token and throw away the rest

The acceptance rule (Leviathan et al. 2023, Chen et al. 2023) is chosen so the
output is distributed EXACTLY as if the big model had sampled every token itself.
Not approximately. This script checks that empirically.

To make the maths inspectable, both "models" are bigram language models over a
12-token vocabulary: the next-token distribution depends on the previous token.
Everything else -- draft, verify, accept, residual resampling -- is the real
algorithm.

    pip install numpy
    python speculative_decoding.py
"""

import numpy as np

V = 12                     # vocabulary size
START = 0                  # token every sequence starts from

_world = np.random.default_rng(0)
TARGET_LOGITS = _world.normal(0, 1.5, (V, V))     # the big, accurate model
DRAFT_NOISE = _world.normal(0, 1.0, (V, V))       # how the small model differs


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


P = softmax(TARGET_LOGITS)                         # P[prev] = target's next-token distribution


def draft_model(noise):
    """A draft model whose quality we control: 0 = identical to the target."""
    return softmax(TARGET_LOGITS + noise * DRAFT_NOISE)


def sample(dist, rng):
    return min(int(np.searchsorted(np.cumsum(dist), rng.random(), side="right")), V - 1)


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------

def speculative_step(prev, Q, k, rng, correct_residual=True):
    """One round: draft k tokens, verify them all with one target pass.

    Returns (new tokens, drafts accepted, drafts actually checked). Drafts after
    the first rejection are never checked -- they are simply discarded.
    """
    # 1. DRAFT -- the small model proposes k tokens, one after another
    drafts, ctx = [], prev
    for _ in range(k):
        token = sample(Q[ctx], rng)
        drafts.append(token)
        ctx = token

    # 2. VERIFY -- in a real system this loop is ONE forward pass of the big
    #    model over all k drafted positions at once.
    out, ctx = [], prev
    for token in drafts:
        p, q = P[ctx], Q[ctx]
        # Accept with probability min(1, p/q): tokens the target likes at least
        # as much as the draft are always kept.
        if rng.random() < min(1.0, p[token] / q[token]):
            out.append(token)
            ctx = token
            continue
        # 3. REJECT -- sample a replacement from the RESIDUAL max(0, p - q),
        #    renormalised. This is the step that makes the output exact.
        if correct_residual:
            residual = np.maximum(p - q, 0)
            out.append(sample(residual / residual.sum(), rng))
        else:
            out.append(sample(p, rng))      # plausible-looking, and WRONG
        accepted = len(out) - 1
        return out, accepted, accepted + 1

    # Every draft accepted: the target pass already computed the NEXT
    # distribution too, so we get one bonus token for free.
    out.append(sample(P[ctx], rng))
    return out, k, k


def target_generate(n, rng):
    tokens, prev = [], START
    for _ in range(n):
        prev = sample(P[prev], rng)
        tokens.append(prev)
    return tokens


def speculative_generate(n, Q, k, rng, correct_residual=True):
    tokens, prev = [], START
    target_passes = accepted = checked = 0
    while len(tokens) < n:
        new, acc, chk = speculative_step(prev, Q, k, rng, correct_residual)
        target_passes += 1
        accepted += acc
        checked += chk
        tokens.extend(new)
        prev = tokens[-1]
    return tokens[:n], target_passes, accepted, checked


# ---------------------------------------------------------------------------
# 1. Is it really lossless?
# ---------------------------------------------------------------------------

def joint_first_two(generate, n_samples, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros((V, V))
    for _ in range(n_samples):
        a, b = generate(rng)[:2]
        counts[a, b] += 1
    return counts / n_samples


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum()


def demo_lossless():
    print("=" * 76)
    print("1. IS THE OUTPUT REALLY THE SAME AS THE BIG MODEL'S?")
    print("=" * 76)
    Q = draft_model(noise=1.5)                      # a deliberately mediocre draft
    n = 60_000
    exact = P[START][:, None] * P                   # true P(first=a, second=b)

    plain = joint_first_two(lambda r: target_generate(2, r), n, seed=10)
    spec = joint_first_two(lambda r: speculative_generate(2, Q, 4, r)[0], n, seed=11)
    buggy = joint_first_two(lambda r: speculative_generate(2, Q, 4, r, False)[0], n, seed=12)

    print("  Distance from the TRUE distribution of the first two tokens, measured")
    print(f"  over {n:,} samples each (total variation distance, 0 = identical):\n")
    print(f"  {'big model sampling every token itself':<46} {tvd(plain, exact):.4f}")
    print(f"  {'speculative decoding (correct)':<46} {tvd(spec, exact):.4f}")
    print(f"  {'speculative, residual step replaced by p':<46} {tvd(buggy, exact):.4f}")
    print(f"  {'(for scale) draft vs big model, token 1 exact':<46} {tvd(Q[START], P[START]):.4f}")

    print(
        "\n  The first two rows differ only by sampling noise: speculative decoding\n"
        "  reproduces the big model's distribution even with a mediocre draft. The\n"
        "  third row keeps the accept rule but resamples a rejected token from p\n"
        "  instead of max(0, p - q) -- an easy mistake, and the test catches it.\n"
        "  The residual matters because tokens the draft over-proposes have already\n"
        "  been partly accepted; the replacement must supply only what is missing."
    )


# ---------------------------------------------------------------------------
# 2. How much faster?
# ---------------------------------------------------------------------------

def demo_speed():
    print("\n" + "=" * 76)
    print("2. HOW MUCH FASTER? -- it depends on how often the draft is right")
    print("=" * 76)
    draft_cost = 0.05          # one draft token costs 5% of one big-model pass
    n_tokens = 6_000
    print(f"  Draft cost assumed {draft_cost:.0%} of a big-model pass per token.")
    print("  acceptance  = share of CHECKED draft tokens the big model accepted")
    print("  tokens/pass = tokens produced per big-model forward pass (measured)")
    print("  formula     = (1 - a^(k+1)) / (1 - a), which assumes every guess has the")
    print("                same acceptance chance a\n")
    print(f"  {'draft quality':<14} {'k':>3} {'acceptance':>11} {'tokens/pass':>12}"
          f" {'formula':>8} {'speedup':>8}")
    print("  " + "-" * 62)

    rng = np.random.default_rng(20)
    best, gaps = (0.0, ""), []
    for label, noise in (("excellent", 0.3), ("good", 0.8), ("mediocre", 1.5), ("poor", 3.0)):
        Q = draft_model(noise)
        for k in (2, 4, 8):
            _, passes, accepted, checked = speculative_generate(n_tokens, Q, k, rng)
            alpha = accepted / checked
            per_pass = n_tokens / passes
            formula = (1 - alpha ** (k + 1)) / (1 - alpha)
            speedup = per_pass / (1 + k * draft_cost)
            gaps.append(abs(per_pass - formula) / per_pass)
            best = max(best, (speedup, f"{label}, k={k}"))
            print(f"  {label:<14} {k:>3} {alpha:>10.0%} {per_pass:>12.2f}"
                  f" {formula:>8.2f} {speedup:>7.2f}x")
        print()

    worst_gap = max(gaps)
    print(f"  Best here: {best[0]:.2f}x ({best[1]}).\n")
    print(
        "  What the table teaches:\n"
        "  - A good draft is everything. Acceptance rate, more than k, drives speedup.\n"
        "  - Bigger k is not always better: with a poor draft most extra guesses are\n"
        "    thrown away but still cost draft time, and speedup can fall below 1x."
    )
    if worst_gap < 0.10:
        print(f"  - The formula matches the measurement to within {worst_gap:.1%} everywhere.")
    else:
        print(
            f"  - The formula is off by up to {worst_gap:.0%}. It assumes every guess has the\n"
            "    same chance of acceptance; in reality some contexts are easy to predict\n"
            "    and some are hard, so use it for intuition and measure for decisions."
        )
    print(
        "\n  This counts big-model passes, which is the right cost model when decoding\n"
        "  is memory-bound (one user, or small batches). At large batch sizes the GPU\n"
        "  is already busy and the gains shrink -- measure on your own traffic."
    )


# ---------------------------------------------------------------------------
# 3. Greedy decoding
# ---------------------------------------------------------------------------

def demo_greedy():
    print("\n" + "=" * 76)
    print("3. GREEDY DECODING -- the simple special case")
    print("=" * 76)
    Q = draft_model(noise=0.8)
    n, k = 60, 4

    target, prev = [], START
    for _ in range(n):
        prev = int(P[prev].argmax())
        target.append(prev)

    spec, prev, passes = [], START, 0
    while len(spec) < n:
        drafts, ctx = [], prev
        for _ in range(k):
            ctx = int(Q[ctx].argmax())
            drafts.append(ctx)
        passes += 1
        ctx = prev
        for token in drafts:
            best = int(P[ctx].argmax())
            spec.append(best)                # always the big model's choice
            ctx = best
            if token != best:                # first mismatch ends the round
                break
        else:
            spec.append(int(P[ctx].argmax()))   # all matched: bonus token
        prev = spec[-1]
    spec = spec[:n]

    print(f"  big model alone       : {n} tokens, {n} forward passes")
    print(f"  speculative (k={k})     : {n} tokens, {passes} forward passes")
    print(f"  identical sequences   : {spec == target}")
    print(
        "\n  With temperature 0 the rule is just: keep drafts while they equal the big\n"
        "  model's top choice. The output cannot change -- only the number of passes."
    )


def main():
    demo_lossless()
    demo_speed()
    demo_greedy()


if __name__ == "__main__":
    main()
