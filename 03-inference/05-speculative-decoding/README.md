# 05 · Speculative Decoding

> **In one sentence:** a small, fast model guesses the next few tokens, the big model checks all of
> the guesses in a single pass, and a carefully designed acceptance rule keeps the output
> **exactly** what the big model alone would have produced — just 2–4× faster.

It is one of the rare optimisations with no quality trade-off at all.

---

## 1. The idea, with an analogy

A senior writer produces one word at a time, slowly and carefully. A junior writer is quick but
makes mistakes.

So: the junior drafts the next sentence. The senior reads the whole draft **at once** and marks the
first word they disagree with. Everything before that word is kept; that word is replaced with the
senior's choice; the rest of the draft is thrown away. Repeat.

When the junior is usually right, the senior approves several words per review instead of writing
one. When the junior is wrong, you lose only the junior's (cheap) time.

---

## 2. Why this can be faster at all

From the [track introduction](../): decode is **memory-bound**. Each forward pass of a big model is
dominated by reading its weights from memory, not by the arithmetic.

Here is the key fact: **checking k tokens costs about the same as generating one.** Verifying a
draft is like prefill — the tokens are known, so the model can process all k positions in one
parallel pass. The expensive weight-reading happens once either way.

```
normal decoding :  pass -> 1 token, pass -> 1 token, pass -> 1 token, pass -> 1 token
speculative     :  draft 4 (cheap), ONE pass checks all 4 -> keeps 3 + fixes 1 = 4 tokens
```

---

## 3. The algorithm, step by step

```
repeat:
  1. DRAFT   the small model generates k tokens:  x1, x2, ..., xk
             and remembers its probabilities q(x)

  2. VERIFY  the big model runs ONE forward pass over all k positions
             and gives its own probabilities p(x) at each position

  3. ACCEPT  for i = 1..k:
               accept xi with probability  min(1, p(xi) / q(xi))
               if rejected:
                 sample a replacement from  max(0, p - q)   (renormalised)
                 discard x(i+1) ... xk and stop
             if ALL k were accepted:
               take one bonus token from the big model's last position for free
```

So every round produces **at least 1** and **at most k + 1** tokens for one big-model pass.

### Why the output is *exactly* the big model's

Two cases for each drafted token:

- The big model likes it **at least as much** as the draft did (`p ≥ q`) → always accept. The draft
  under-proposed it, if anything.
- The big model likes it **less** (`p < q`) → accept only with probability `p/q`, correcting for the
  draft's over-enthusiasm.

Then, on a rejection, the replacement is drawn from `max(0, p − q)`: only the tokens the draft
**under**-proposed. Put together, every token ends up chosen with probability exactly `p` — the big
model's distribution. This is standard **rejection sampling**, and the proof is a few lines in the
[Leviathan et al.](https://arxiv.org/abs/2211.17192) paper.

The script checks it statistically — 60,000 samples each of the first two tokens, measured as
distance from the true distribution (0 = identical):

| Method | Distance from the big model's true distribution |
|---|---|
| Big model sampling every token itself | 0.0149 |
| **Speculative decoding (correct)** | **0.0141** |
| Speculative, but replacement sampled from `p` instead of `max(0, p − q)` | **0.2091** |
| *(for scale)* the draft model vs the big model, first token | 0.3520 |

The first two differ only by sampling noise — even with a mediocre draft. The third row shows why the
residual step matters: an easy-looking shortcut, and it quietly changes the output distribution.

### Greedy decoding is even simpler

At temperature 0, just keep drafted tokens **while they equal the big model's top choice**. The
script produces the same 60 tokens as the big model alone, in **13** big-model passes instead of 60.

---

## 4. How much faster?

It depends almost entirely on the **acceptance rate** — how often the big model agrees with the
draft.

If each drafted token is accepted with probability α, the expected tokens per big-model pass is:

```
expected tokens per pass = (1 − α^(k+1)) / (1 − α)
```

and the real speedup also has to pay for drafting:

```
speedup ≈ tokens per pass / (1 + k × draft cost per token)
```

From [`speculative_decoding.py`](examples/speculative_decoding.py), with a draft costing 5% of a
big-model pass:

| Draft quality | k | Acceptance | Tokens per pass | Speedup |
|---|---|---|---|---|
| excellent | 2 | 88% | 2.65 | 2.41× |
| excellent | 4 | 88% | 3.93 | 3.27× |
| excellent | 8 | 89% | 5.82 | **4.16×** |
| good | 4 | 69% | 2.74 | 2.29× |
| mediocre | 4 | 48% | 1.89 | 1.58× |
| poor | 2 | 25% | 1.31 | 1.19× |
| poor | 8 | 25% | 1.33 | **0.95×** — slower than no speculation |

Three lessons:

1. **A good draft is everything.** Acceptance rate drives the speedup far more than k.
2. **More guesses is not always better.** With a poor draft, extra guesses are thrown away but still
   cost drafting time — you can end up *slower*.
3. **It helps most when decoding is memory-bound** — one user or small batches. At large batch sizes
   the GPU's compute is already busy, and the gains shrink.

In this toy the formula matched the measurement almost exactly; on real text, acceptance varies a lot
between easy tokens (boilerplate, code, copying from the prompt) and hard ones, so treat the formula
as intuition and measure on your own traffic.

---

## 5. Where the guesses come from

The "draft" does not have to be a separate model. The main families:

| Method | Where guesses come from | Notes |
|---|---|---|
| **Draft model** | a smaller model from the same family | must share the big model's **tokenizer**; extra model in memory |
| **N-gram / prompt lookup** | copy spans that already appear in the prompt or output | no extra model; excellent for RAG, summarisation, code editing |
| **Medusa** | extra prediction heads bolted onto the big model | no separate model; heads need training |
| **EAGLE / EAGLE-3** | a tiny head that predicts from the big model's internal features | high acceptance; widely supported |
| **MTP** (multi-token prediction) | heads the model was *trained* with | free when the model ships with them |
| **Lookahead decoding** | the big model itself, generating n-grams in parallel | no draft model or training |

vLLM supports EAGLE, MTP, draft models, n-gram, suffix decoding and more — see its
[speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

---

## 6. Where you will meet it

**vLLM:**
```bash
vllm serve <big-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<small-model-same-tokenizer>",
    "num_speculative_tokens": 5
  }'
```

**Hugging Face `transformers`** calls it *assisted generation*:
```python
outputs = model.generate(**inputs, assistant_model=small_model, max_new_tokens=200)
```

---

## 7. Common misconceptions

- **"It lowers quality."** No — with the correct acceptance rule the output distribution is exactly
  the big model's. (Some *lossy* variants relax the rule for extra speed; those do change output.)
- **"Bigger k is always faster."** The table shows the opposite with a poor draft.
- **"Any small model works as a draft."** It must use the **same tokenizer**, and it only helps if
  it agrees with the big model often on *your* traffic.
- **"It always helps a busy server."** At large batch sizes decoding stops being memory-bound, and
  the extra drafting work competes with real requests. Measure before enabling.

---

## 8. Examples

```bash
pip install numpy
python examples/speculative_decoding.py      # ~15 s on CPU
```

[`speculative_decoding.py`](examples/speculative_decoding.py) implements the full algorithm with two
small bigram "models" whose agreement you can control, and shows:
1. the statistical check that the output matches the big model — and catches a subtly wrong version
2. acceptance rate, tokens per pass and speedup across draft qualities and values of k
3. greedy speculative decoding producing an identical sequence in far fewer passes

---

## 9. Exercises

1. **Do one round by hand.** Vocabulary {a, b, c}. Draft probabilities q = (0.6, 0.3, 0.1), big model
   p = (0.3, 0.5, 0.2). The draft proposes "a". What is the probability it is accepted? If it is
   rejected, compute the replacement distribution `max(0, p − q)` renormalised.
2. **Prove it numerically.** For the example above, compute the total probability that the final
   token is a, b and c (accept path + reject path). Check that it equals p exactly.
3. **Find the best k.** For each draft quality in the script, find the k that maximises speedup.
   How does the best k change as the draft gets better?
4. **Make drafting expensive.** Change `draft_cost` from 5% to 30%. Which configurations still speed
   things up?
5. **Temperature.** Sharpen or flatten both models' distributions (divide logits by a temperature)
   and watch acceptance change. Why does creative, high-temperature sampling usually benefit less?
6. **Prompt lookup.** Write a draft function that proposes the tokens that followed the last
   occurrence of the current token earlier in the sequence. On repetitive sequences, compare its
   acceptance rate with the model-based draft.

---

## 10. Projects to build and test

### Beginner — Assisted generation benchmark
Use Hugging Face `transformers` with a small model as `assistant_model` for a larger model of the
same family.

**How to test it:** with greedy decoding the outputs must be identical with and without the
assistant for 20 prompts; report tokens per second for both on code, chat and summarisation prompts,
and explain which task benefits most.

### Intermediate — Prompt-lookup decoding
Implement n-gram prompt-lookup drafting around a real model (no draft model), with the correct
acceptance rule.

**How to test it:** on summarisation and document Q&A, where outputs copy from the input, measure
acceptance rate and speedup; confirm greedy output is identical to normal decoding.

### Advanced — Speculative decoding on a server
Serve a model with vLLM with and without speculative decoding (EAGLE or a draft model), and
load-test at several request rates.

**How to test it:** plot TPOT and throughput against request rate for both. Find the request rate at
which speculative decoding stops helping — and report it honestly, because that crossover is the
most useful number for a real deployment decision.

---

## 11. Resources

**Start here — beginner friendly**
- [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) — Google Research, by the people who introduced it. A clear, non-mathematical history.
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation) — Hugging Face, with visuals and code.
- [An Introduction to Speculative Decoding for Reducing Latency in AI Inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) — NVIDIA.

**Go deeper**
- [Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) — vLLM docs, every supported method.
- [Speculative Sampling](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) — TensorRT-LLM docs.
- [Speculative Decoding Production Guide](https://www.spheron.network/blog/speculative-decoding-production-guide/) — Spheron. Practical deployment advice.

**Papers**
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al.; the proof of losslessness
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., DeepMind (found independently)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) · [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057)

---

**Previous:** [04 · Paged Attention](../04-paged-attention/) · **Track:** [03 · Inference](../) ·
**Next track:** [04 · Serving](../../04-serving/)
