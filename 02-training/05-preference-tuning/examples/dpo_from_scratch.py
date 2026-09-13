"""
Direct Preference Optimization, from scratch, in plain PyTorch.

    L_DPO = -log sigmoid( beta * [ (log pi(y_c) - log pi_ref(y_c))
                                 - (log pi(y_r) - log pi_ref(y_r)) ] )

To keep every number inspectable, the "language model" here is a policy over a
fixed set of candidate responses per prompt: pi(y|x) = softmax(logits[x])[y].
That is exactly the quantity DPO needs -- a log-probability of a whole response
given a prompt. In a real LLM that number is the sum of token log-probs; the
loss on top of it is identical.

A hidden reward decides which responses are actually better. Preference pairs
are sampled from it with realistic label noise. We train on some pairs and test
whether the model's IMPLICIT reward, beta * log(pi / pi_ref), recovers the
hidden ranking on pairs it never saw.

    pip install torch
    python dpo_from_scratch.py
"""

import math

import torch
import torch.nn.functional as F

N_PROMPTS = 40
N_RESPONSES = 8          # candidate responses per prompt
N_TRAIN_PAIRS = 1200
N_TEST_PAIRS = 1000
STEPS = 600
LR = 0.05


# ---------------------------------------------------------------------------
# The loss -- this is the entire technique
# ---------------------------------------------------------------------------

def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    """All inputs are log-probabilities of whole responses, shape (batch,)."""
    chosen_logratio = pol_chosen - ref_chosen          # how much MORE likely chosen became
    rejected_logratio = pol_rejected - ref_rejected    # how much more likely rejected became
    margin = beta * (chosen_logratio - rejected_logratio)
    return -F.logsigmoid(margin).mean(), margin


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

def make_world(seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden_reward = torch.randn(N_PROMPTS, N_RESPONSES, generator=g)
    # The reference (think: your SFT model) has its own opinions, only weakly
    # correlated with what people actually prefer.
    ref_logits = 0.3 * hidden_reward + torch.randn(N_PROMPTS, N_RESPONSES, generator=g)
    return hidden_reward, ref_logits


def sample_pairs(hidden_reward, n, seed):
    """Bradley-Terry sampling: P(a preferred over b) = sigmoid(r_a - r_b).

    So close calls are noisy, just like real annotator disagreement.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, N_PROMPTS, (n,), generator=g)
    a = torch.randint(0, N_RESPONSES, (n,), generator=g)
    b = (a + torch.randint(1, N_RESPONSES, (n,), generator=g)) % N_RESPONSES   # b != a
    p_a = torch.sigmoid(hidden_reward[x, a] - hidden_reward[x, b])
    a_wins = torch.rand(n, generator=g) < p_a
    chosen = torch.where(a_wins, a, b)
    rejected = torch.where(a_wins, b, a)
    return x, chosen, rejected


def log_probs(logits, x, y):
    return F.log_softmax(logits[x], dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)


def kl_to_reference(pol_logits, ref_logits):
    """Mean KL(pi || pi_ref) over prompts -- how far the policy has drifted."""
    p = F.log_softmax(pol_logits, dim=-1)
    q = F.log_softmax(ref_logits, dim=-1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def ranking_accuracy(scores, hidden_reward):
    """Across every pair of responses for every prompt: does `scores` order them
    the same way the hidden reward does?"""
    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    r_diff = hidden_reward.unsqueeze(2) - hidden_reward.unsqueeze(1)
    mask = r_diff != 0
    return ((s_diff * r_diff > 0) & mask).sum().item() / mask.sum().item()


def train(beta, hidden_reward, ref_logits, train_pairs, log=False, steps=STEPS):
    policy = ref_logits.clone().requires_grad_(True)     # start as a copy of the reference
    opt = torch.optim.Adam([policy], lr=LR)
    x, yc, yr = train_pairs
    ref_c, ref_r = log_probs(ref_logits, x, yc), log_probs(ref_logits, x, yr)

    history = []
    for step in range(steps + 1):
        pol_c, pol_r = log_probs(policy, x, yc), log_probs(policy, x, yr)
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        if log and step in (0, 50, 150, 300, 600):
            history.append((step, loss.item(), pol_c.mean().item(), pol_r.mean().item()))
        if step < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy.detach(), history


def pair_accuracy(logits, ref_logits, pairs):
    """Fraction of preference pairs where the implicit reward favours `chosen`."""
    x, yc, yr = pairs
    return (log_probs(logits, x, yc) - log_probs(ref_logits, x, yc)
            > log_probs(logits, x, yr) - log_probs(ref_logits, x, yr)).float().mean().item()


def main():
    torch.manual_seed(0)
    hidden_reward, ref_logits = make_world()
    train_pairs = sample_pairs(hidden_reward, N_TRAIN_PAIRS, seed=1)
    test_pairs = sample_pairs(hidden_reward, N_TEST_PAIRS, seed=2)

    # Label agreement ceiling: how often does the pair label match the TRUE
    # better response? No method can beat the noise baked into the data.
    x, yc, yr = test_pairs
    label_quality = (hidden_reward[x, yc] > hidden_reward[x, yr]).float().mean().item()

    print("=" * 74)
    print("1. SANITY CHECKS")
    print("=" * 74)

    lp = log_probs(ref_logits, *train_pairs[:2]), log_probs(ref_logits, train_pairs[0], train_pairs[2])
    loss0, _ = dpo_loss(lp[0], lp[1], lp[0], lp[1], beta=0.1)
    print(f"  loss at init (policy == reference) : {loss0.item():.6f}")
    print(f"  log 2                              : {math.log(2):.6f}")
    print("  Identical policy and reference -> margin 0 -> sigmoid(0) = 0.5 -> -log 0.5.")

    # Hand-computed single example.
    pc, pr, rc, rr, b = torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.5]), 0.5
    manual = -math.log(1 / (1 + math.exp(-(b * ((-1.0 - -2.0) - (-3.0 - -2.5))))))
    ours, _ = dpo_loss(pc, pr, rc, rr, b)
    print(f"\n  hand-computed loss for one example : {manual:.6f}")
    print(f"  dpo_loss()                         : {ours.item():.6f}")

    print("\n" + "=" * 74)
    print("2. TRAINING AT beta = 0.1")
    print("=" * 74)
    policy, history = train(0.1, hidden_reward, ref_logits, train_pairs, log=True)
    print(f"  {'step':>5} {'loss':>8} {'mean log pi(chosen)':>21} {'mean log pi(rejected)':>23}")
    for step, loss, lc, lr in history:
        print(f"  {step:>5} {loss:>8.4f} {lc:>21.3f} {lr:>23.3f}")
    _, _, c0, r0 = history[0]
    _, _, c1, r1 = history[-1]
    print(f"\n  chosen log-prob moved {c1 - c0:+.2f}, rejected moved {r1 - r0:+.2f}.")
    if c1 < c0:
        print(
            "  BOTH fell. DPO only optimises the gap between them relative to the\n"
            "  reference. Here it widened that gap mostly by pushing rejected down hard,\n"
            "  while chosen became less likely too -- the probability mass went to other\n"
            "  responses. This is a known DPO behaviour, not a bug in the demo. In a real\n"
            "  LLM it can mean the good responses themselves get rarer; a small SFT loss on\n"
            "  the chosen responses (LLaMA-Factory `pref_ftx`) counters it. Exercise 6."
        )
    else:
        print("  Chosen rose and rejected fell -- the gap widened from both sides.")

    print("\n" + "=" * 74)
    print("3. DID IT LEARN THE HIDDEN PREFERENCE?")
    print("=" * 74)
    print(f"  ranking accuracy vs hidden reward, all response pairs:")
    print(f"     reference (SFT) model : {ranking_accuracy(ref_logits, hidden_reward):.1%}")
    print(f"     implicit reward (DPO) : {ranking_accuracy(policy - ref_logits, hidden_reward):.1%}")
    print(f"     policy itself (DPO)   : {ranking_accuracy(policy, hidden_reward):.1%}")
    print(f"\n  held-out preference pairs predicted : {pair_accuracy(policy, ref_logits, test_pairs):.1%}")
    print(f"  ceiling set by label noise          : {label_quality:.1%}  (labels matching the true better response)")
    print(
        "\n  The implicit reward is beta * log(pi/pi_ref). DPO never trained a reward\n"
        "  model, yet one falls out of the policy -- that is the paper's central result.\n"
        "  Note what the policy itself scores: it still carries the reference's own\n"
        "  preferences, deliberately. That is the KL anchor at work."
    )

    print("\n" + "=" * 74)
    print("4. THE beta TRADE-OFF")
    print("=" * 74)
    sweep_steps = 3000   # to convergence, so beta's effect is not confused with undertraining
    print(f"  each run trained to convergence ({sweep_steps} steps)\n")
    print(f"  {'beta':>6} {'train pair acc':>15} {'held-out acc':>13} {'KL(pi || pi_ref)':>17}")
    print("  " + "-" * 55)
    kls = []
    for beta in (0.01, 0.05, 0.1, 0.5, 2.0):
        pol, _ = train(beta, hidden_reward, ref_logits, train_pairs, steps=sweep_steps)
        kl = kl_to_reference(pol, ref_logits)
        kls.append(kl)
        print(f"  {beta:>6} {pair_accuracy(pol, ref_logits, train_pairs):>15.1%} "
              f"{pair_accuracy(pol, ref_logits, test_pairs):>13.1%} {kl:>17.3f}")
    print(
        f"\n  KL falls from {kls[0]:.2f} to {kls[-1]:.2f} as beta rises: beta is a leash on how\n"
        "  far the policy may move away from the reference. That is the real lesson.\n\n"
        "  Accuracy barely moves, and it is worth being honest about why. This toy\n"
        "  policy is a free table of logits, so every beta finds the same ORDERING of\n"
        "  responses -- beta only changes how hard it pushes. In this toy, then, a large\n"
        "  beta gets the same accuracy with a fraction of the drift, and drifting costs\n"
        "  nothing we can measure.\n\n"
        "  A real LLM differs in exactly that last respect. Its weights are shared\n"
        "  across everything it does, so drifting far to fit preferences is where\n"
        "  fluency and general capability quietly degrade, and where it starts\n"
        "  exploiting quirks of the data such as length. A lookup table has no such\n"
        "  cost, which is why you measure it on a real model (project 2 in the README).\n\n"
        "  0.1 is the standard starting point. LLaMA-Factory: `pref_beta: 0.1`;\n"
        "  TRL: `DPOConfig(beta=0.1)`."
    )


if __name__ == "__main__":
    main()
