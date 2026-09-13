# 05 · Preference Tuning

SFT teaches a model to *answer*. It cannot teach it which of two reasonable answers is *better* —
more helpful, more honest, better judged in tone, more willing to say "I don't know". There is no
single gold response for "explain this politely to a frustrated customer"; there are better and
worse ones.

**Preference tuning learns from comparisons instead of examples.** "Response A is better than B"
is far easier for a human to judge — and far more informative for the model — than writing the
perfect response from scratch.

---

## 1. What this is

The data changes shape:

```
SFT data            :  (prompt, response)
Preference data     :  (prompt, chosen, rejected)
```

```json
{
  "instruction": "My order is three weeks late. What is going on?",
  "chosen":   "I'm sorry — three weeks is far too long. Let me check order #4471 now and tell you exactly where it is.",
  "rejected": "Orders can take time. Please be patient."
}
```

Both responses are fluent. Both are "correct" in an SFT sense. The preference says which one a
person actually wanted. Preference tuning pushes the probability of the chosen style up and the
rejected style down — relative to where the model started.

### The family tree

```
                 preference data (chosen vs rejected)
                              |
          +-------------------+---------------------+
          |                                         |
    RLHF (2022)                               Direct methods (2023+)
    1. train a reward model                   no reward model, no RL loop
    2. optimise the policy with PPO           a classification-style loss
       against it, with a KL penalty          on the preference pairs
          |                                         |
    powerful, expensive, unstable              DPO, IPO, ORPO, SimPO, KTO
```

---

## 2. RLHF with PPO — the original

This is how InstructGPT and early ChatGPT were aligned.

1. **Reward model.** Train a model to score a (prompt, response) pair, using the preference data:
   make `score(chosen) > score(rejected)`. This is the Bradley–Terry model:
   `P(chosen ≻ rejected) = σ(r_chosen − r_rejected)`.
2. **PPO.** Generate responses from the policy, score them with the reward model, and update the
   policy to increase reward — with a **KL penalty** keeping it close to the SFT model.

That KL penalty is the single most important idea in this module. Without it, the policy finds
responses that score highly with the reward model while being garbage to a human — **reward
hacking**. The reward model is only a proxy; optimise any proxy hard enough and it stops tracking
what you meant.

**The cost:** four models in memory at once (policy, reference, reward model, value model), an RL
loop that is notoriously sensitive to hyperparameters, and online generation during training. It
works, and at frontier labs with the infrastructure for it, it is still used. For almost everyone
else, it is the wrong starting point.

---

## 3. DPO — the one to start with

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) showed you can skip both the
reward model and the RL loop. The key result: the optimal policy under the RLHF objective has a
closed form, so the reward can be written *in terms of the policy itself*:

```
implicit reward   r(x, y)  =  β · log( π(y|x) / π_ref(y|x) )
```

Plug that into the Bradley–Terry preference model and you get a loss you can minimise with plain
gradient descent on your preference pairs:

```
L_DPO = − log σ( β · [ (log π(y_c) − log π_ref(y_c))  −  (log π(y_r) − log π_ref(y_r)) ] )
                      \_____ how much more likely ____/     \____ how much more likely ___/
                             chosen got vs reference             rejected got vs reference
```

Read it as: **increase the chosen response's likelihood *relative to the reference model* by more
than the rejected response's.** The reference model (a frozen copy of your SFT model) plays exactly
the role the KL penalty played in PPO.

### What β does

`β` controls how far the policy may drift from the reference.

| β | Behaviour |
|---|---|
| small (0.01–0.05) | Drifts far; fits preferences aggressively; higher risk of degradation |
| **0.1** | **The standard default.** Start here. |
| large (0.5+) | Stays close to the reference; conservative; may barely change |

`dpo_from_scratch.py` in this folder measures the drift half of this trade-off directly — trained to
convergence, KL to the reference falls from ~2.3 at β=0.01 to ~0.5 at β=2.0. The *cost* of drifting
(lost fluency and general capability) needs a real model with shared weights to show up, which is
why the script also explains what its toy cannot demonstrate.

### Why DPO won

| | PPO | DPO |
|---|---|---|
| Models in memory | 4 | 2 (policy + frozen reference) |
| Reward model | required | none |
| Online generation during training | yes | no |
| Stability | fiddly | close to SFT |
| Implementation | hundreds of lines of RL | a ~10-line loss |

---

## 4. The variants, and when to reach for each

| Method | What it changes | Use when |
|---|---|---|
| **DPO** | The baseline above | **Default.** You have (chosen, rejected) pairs and an SFT model. |
| **IPO** | Bounded loss that resists overfitting | DPO is overfitting a small preference set |
| **ORPO** | Folds preference into the SFT loss; **no reference model** | You want SFT + alignment in one pass, or have no strong SFT checkpoint; lower memory |
| **SimPO** | Length-normalised reward, no reference model | Your DPO model has learned to be verbose |
| **KTO** | Works from **unpaired** thumbs-up / thumbs-down | Your feedback is binary ratings, not comparisons — which is what most product logs contain |
| **PPO / GRPO** | Online RL against a reward | You have a *verifiable* reward (tests pass, maths answer correct) and the infrastructure |

A note on that last row. RL with **verifiable rewards** — a unit test passes, a maths answer checks
out — has become the main route to training reasoning models (see
[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)). It sidesteps the reward-hacking problem
because the reward is not a learned proxy. It is a different use case from preference tuning for
tone and helpfulness, and worth knowing as a separate branch.

---

## 5. The data, again, is the whole job

- **Chosen and rejected should differ in the property you care about.** If chosen is also always
  longer, the model learns "longer is better" — the most common failure in preference tuning.
  Check the length distribution of chosen vs rejected before you train.
- **Rejected responses should be plausible.** A rejected response that is obviously broken teaches
  nothing the model did not already know. The most informative pairs are close calls.
- **On-policy pairs work better.** Generate both responses *from your own SFT model*, then have
  them ranked. Pairs written by a different model describe a distribution your model is not in.
- **Start from a good SFT model.** DPO sharpens preferences; it does not install capabilities.
- **Label noise is real.** Human annotators agree with each other perhaps 60–75% of the time on
  subtle pairs. Budget for it and look at disagreements by hand.

### Where preference data comes from

- **Human ranking** of two sampled responses — best, slowest
- **AI feedback (RLAIF)** — a strong model ranks against a written rubric; see
  [Constitutional AI](https://arxiv.org/abs/2212.08073). Cheap, scalable, inherits the judge's biases.
- **Product signals** — thumbs up/down, regenerations, edits. Binary and unpaired, so a natural fit for KTO.
- **Public sets** — [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized),
  [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf). Good for learning.

---

## 6. Examples

### From scratch — DPO in plain PyTorch

```bash
pip install torch
python examples/dpo_from_scratch.py
```

[`dpo_from_scratch.py`](examples/dpo_from_scratch.py) implements the DPO loss and trains a small
policy on synthetic preference pairs generated from a hidden reward. It verifies and measures:

- at initialisation, policy == reference, so the loss is exactly `log 2 ≈ 0.693`
- the loss implementation matches a hand-computed value
- after training, the **implicit reward** `β·log(π/π_ref)` ranks *held-out* responses in agreement
  with the hidden reward the preferences came from
- a **β sweep**, trained to convergence: β tightens how far the policy drifts from the reference
  (KL) — with an honest note on why a toy cannot show the capability cost of drifting that a real
  LLM pays

### Real DPO with LLaMA-Factory

```bash
llamafactory-cli train examples/llamafactory_dpo.yaml
```

[`llamafactory_dpo.yaml`](examples/llamafactory_dpo.yaml) — DPO on top of an SFT adapter, with the
switches for ORPO, SimPO and KTO documented inline.

Preference dataset format (Alpaca style):

```json
[
  {
    "instruction": "user instruction",
    "input": "optional context",
    "chosen": "the better response",
    "rejected": "the worse response"
  }
]
```

Registered in `dataset_info.json` with `"ranking": true`:

```json
"my_prefs": {
  "file_name": "my_prefs.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

KTO uses unpaired examples with a boolean label instead — see the
[LLaMA-Factory data README](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) for
the exact column.

---

## 7. Exercises

1. **Check the starting loss.** Run `dpo_from_scratch.py` and confirm the initial loss is `log 2`.
   Explain in one sentence why it must be exactly that.
2. **Sweep β.** Read the β table. KL falls steadily as β rises while accuracy barely moves. Explain
   why a table of logits behaves that way, then name the cost of a small β that a real LLM would
   pay and this toy cannot show.
3. **Remove the reference.** Replace `log π − log π_ref` with `log π` alone (roughly what SimPO and
   ORPO do, minus their other safeguards). Watch the KL column. What stopped the drift before?
4. **Plant a length bias.** Make every chosen response longer than its rejected pair, independent
   of quality. Train. Then check whether the model learned quality or length.
5. **Flip 30% of labels.** Simulate annotator disagreement. How much does held-out accuracy fall?
   Compare DPO against IPO-style clipping if you implement it.
6. **Watch both likelihoods.** The script shows `log π(chosen)` *and* `log π(rejected)` both falling
   while the margin grows. Add an SFT term on the chosen responses — `loss + λ·(−log π(chosen))`,
   which is what LLaMA-Factory's `pref_ftx` does — and find the smallest λ that stops chosen from
   falling. What does it cost in held-out accuracy?
7. **DPO vs SFT-on-chosen.** Train one model with DPO and one with plain SFT on only the chosen
   responses. Compare on held-out pairs. What does the rejected response add?

---

## 8. Projects to build and test

### Beginner — Preference dataset builder
Sample two responses per prompt from a small SFT model, rank them with an LLM judge against a
written rubric, and export the pairs in LLaMA-Factory format.

**How to test it:** hand-label 50 pairs yourself and measure agreement with the judge. Report the
length difference between chosen and rejected — if chosen is systematically longer, your judge
has a length bias and you have found it before training on it.

### Intermediate — DPO on a real model
Take an SFT checkpoint from [02 · Fine-Tuning](../02-fine-tuning/) and DPO it on your pairs with
QLoRA on a free GPU.

**How to test it:** win rate of DPO model vs SFT model on 100 held-out prompts, judged blind. Also
check a general benchmark before and after — alignment that damages capability is a regression,
not a win.

### Intermediate — Method bake-off
DPO, ORPO, SimPO and KTO on the same data (binarise the pairs into thumbs-up/down for KTO).

**How to test it:** same base model, same budget, 3 seeds each. Report win rate, mean response
length, and training memory. Response length is not optional — several methods "win" partly by
getting longer.

### Advanced — Reward hacking demonstration
Train a reward model, optimise a policy against it with little or no KL constraint, and document
the degenerate behaviour it discovers.

**How to test it:** plot reward-model score and human-judged quality over training on the same
axes. The point where they diverge is reward hacking, visible. Then add the KL term back and show
it closes the gap.

---

## 9. Resources

**Understand it**
- [Illustrating RLHF](https://huggingface.co/blog/rlhf) — Hugging Face. The clearest single introduction.
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) — Chip Huyen. Excellent on the full pipeline and its trade-offs.
- [The RLHF Book](https://rlhfbook.com/) — Nathan Lambert. Free, thorough, current; the reference text for this module.
- [DPO vs PPO: which to use in production](https://www.spheron.network/blog/dpo-vs-ppo-rlhf-algorithm-production-llm-alignment/) — a practical decision guide.

**Papers**
- [InstructGPT](https://arxiv.org/abs/2203.02155) — RLHF with PPO, where it started
- [DPO](https://arxiv.org/abs/2305.18290) — read section 4; the derivation is short and elegant
- [IPO](https://arxiv.org/abs/2310.12036) · [ORPO](https://arxiv.org/abs/2403.07691) · [KTO](https://arxiv.org/abs/2402.01306) · [SimPO](https://arxiv.org/abs/2405.14734)
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — AI feedback instead of human labels
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) — RL with verifiable rewards

**Do it**
- [TRL `DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer) — also ORPO, KTO, reward modelling, PPO, GRPO
- [LLaMA-Factory DPO / KTO examples](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/train_lora)
- [UltraFeedback (binarized)](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) · [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)

---

**Previous:** [04 · QLoRA](../04-qlora/) · **Next track:** [03 · Inference](../../03-inference/)
