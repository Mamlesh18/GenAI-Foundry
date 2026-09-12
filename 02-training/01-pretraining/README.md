# 01 · Pretraining

Where a model learns language, facts and reasoning — by doing one thing, trillions of times:
**predict the next token.**

You will almost certainly never pretrain a frontier model. You should still understand this stage,
because it determines everything the model can ever do, and because every limitation you fight
later was set here.

---

## 1. What this is

Take a transformer with random weights. Feed it text. At every position, ask it to predict the next
token. Penalise it when it is wrong. Repeat for trillions of tokens.

```
input :  The  capital  of  France   is
target:  capital  of   France  is   Paris
         ^        ^    ^       ^    ^
         every position predicts the next one, in parallel, in a single forward pass
```

That parallelism is the gift of the causal mask from
[01 · Attention](../../01-foundations/01-attention-is-all-you-need/): position `i` can only see
positions `≤ i`, so one pass over a 4,096-token document produces 4,096 training signals at once.

**The loss is cross-entropy**, and it has a beautiful interpretation:

```
loss = -log P(correct_token)
```

- Model assigns 1.0 to the right token → loss 0. Perfect.
- Model assigns 0.5 → loss 0.69.
- Model assigns 0.001 → loss 6.9. Badly wrong, heavily penalised.

A randomly-initialised model with a 50,000-token vocabulary starts at `ln(50000) ≈ 10.8`. Watching
that number fall is watching the model learn. Good models land around 1.5–2.5.

**Perplexity** = `exp(loss)`. It means "on average, how many tokens is the model effectively
choosing between?" Perplexity 10 means it has narrowed 50,000 options down to about 10. It is the
same information as loss, on a scale that is easier to feel.

### There are no labels

This is the crucial insight. Nobody annotated anything. The text *is* the label — the next token is
always sitting right there. That is what makes training on the whole internet possible, and it is
why this is called **self-supervised** learning.

---

## 2. Why it matters

Everything the model knows, it knows from here. Fine-tuning and preference tuning are small
adjustments to a foundation laid in this stage. Specifically:

- **The knowledge cutoff** is when this data was collected.
- **The biases** are the data's biases.
- **The languages it is good at** are the languages that were well represented.
- **The context window** is bounded by what it was trained to handle.
- **What it cannot do** is usually what was missing or rare in the corpus.

You cannot fine-tune your way out of a pretraining gap. That is worth internalising before you
spend a week trying.

### What it costs

| Model | Tokens | Rough compute | Rough cost |
|---|---|---|---|
| GPT-2 (1.5B) | ~10B | small cluster, days | thousands |
| Llama 3 8B | ~15T | thousands of GPUs, weeks | millions |
| Frontier models | 15T+ | tens of thousands of GPUs | tens of millions+ |

This is why there are perhaps a dozen organisations doing it and everyone else starts from their
weights. **Continued pretraining** — taking an existing base model and training further on your own
domain corpus — is the version that is actually accessible, and is what the LLaMA-Factory config in
this folder does.

### Scaling laws, and the Chinchilla correction

[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) found loss falls predictably with
parameters, data and compute — smooth power laws across many orders of magnitude. This is why the
field bet on scale: the curve told you in advance that a bigger model would be better.

[Chinchilla (2022)](https://arxiv.org/abs/2203.15556) then showed the early large models were badly
*undertrained*. For a fixed compute budget, the optimum is roughly **20 tokens per parameter** —
so a 7B model wants ~140B tokens, not 300B parameters trained on 300B tokens.

The practical consequence you see today: smaller models trained on far more data. Llama 3 8B was
trained on ~15T tokens — about 1,900 tokens per parameter, *far* past Chinchilla-optimal, because
when a model will serve billions of requests, inference cost matters more than training efficiency.

---

## 3. Data: where the real work is

Modelling is the glamorous part. Data is the part that determines the outcome.

| Stage | What happens | Why |
|---|---|---|
| **Collection** | Common Crawl, code, books, Wikipedia, papers | Raw material |
| **Quality filtering** | Classifiers and heuristics drop low-quality pages | Most of the web is junk |
| **Deduplication** | Exact and fuzzy dedup across the corpus | Duplicates cause memorisation and waste compute |
| **Decontamination** | Remove text overlapping with benchmarks | Otherwise your evaluation is meaningless |
| **Mixing** | Weight the sources deliberately | More code improves reasoning; more books improves long-form |
| **Tokenization** | Text to integer ids | See [05 · Tokenization](../../01-foundations/05-tokenization/) |

Two things worth knowing:

**Deduplication matters more than people expect.** Duplicated text gets memorised verbatim rather
than generalised from, and it inflates your token count without adding information.

**Benchmark contamination is rampant.** If MMLU questions are in the training data, a high MMLU
score means nothing. When you see a surprising benchmark result, this is the first thing to suspect.

Open datasets worth knowing: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (15T
tokens, well documented), [The Pile](https://pile.eleuther.ai/),
[Dolma](https://huggingface.co/datasets/allenai/dolma),
[RedPajama](https://github.com/togethercomputer/RedPajama-Data).

---

## 4. Examples

### Run the from-scratch version

```bash
pip install torch
python examples/tiny_pretrain.py
```

[`tiny_pretrain.py`](examples/tiny_pretrain.py) pretrains a ~600K-parameter character-level
transformer from random weights on CPU in about 80 seconds. It is deliberately tiny, but it is the
*real* algorithm — same loss, same shifted-target trick, same loop that trains a 70B model.

On a typical run you will see loss fall from **3.30** (`ln(27)`, uniform guessing over the
character vocabulary) to about **0.24**, with training and validation loss within ~0.01 of each
other, and the samples turn from noise into novel grammatical sentences.

That near-zero gap is the point. Validation text is drawn from the same generator as the training
text but shares none of its sentences, so matching losses mean the model learned the *pattern*
rather than memorising the data. Exercise 4 makes it produce the opposite result on purpose.

> The corpus is generated from a small grammar rather than scraped prose, so the structure is
> genuinely learnable in 80 seconds on a CPU. That also makes perplexity unusually low — real
> language is far harder, and real models land nearer 1.5–2.5 on subword tokens.

### Continued pretraining with LLaMA-Factory

[`llamafactory_pretrain.yaml`](examples/llamafactory_pretrain.yaml) runs continued pretraining on a
real model with your own domain text.

```bash
llamafactory-cli train examples/llamafactory_pretrain.yaml
```

Dataset format for pretraining is just raw text, one document per record:

```json
[
  {"text": "Your domain document, as long as you like..."},
  {"text": "Another document..."}
]
```

No instructions, no labels — the text is the signal.

---

## 5. Exercises

1. **Watch the loss.** Run `tiny_pretrain.py`. Note the starting loss. Confirm it is close to
   `ln(vocab_size)` — that is the model guessing uniformly. Why is that the correct starting point?
2. **Compute perplexity.** Add perplexity (`exp(loss)`) to the training log. At loss 1.5, how many
   tokens is the model effectively choosing between?
3. **Break the shift.** Remove the target-shifting so inputs and targets align at the same
   position. Loss will collapse to near zero. Explain why that is a bug and not a breakthrough.
4. **Overfit deliberately.** Change `make_corpus(3000, seed=1)` to `make_corpus(20, seed=1)` and
   raise `STEPS` to 3000. Watch training loss keep falling while validation loss *climbs*. Sample
   from the result — you will get training sentences back verbatim. That is memorisation, and at
   scale it is exactly what deduplication exists to prevent. Note the step where the two curves
   part company: that is where you should have stopped.
5. **Chinchilla maths.** Your budget trains 1B parameters on 20B tokens. Chinchilla-optimal for 20B
   tokens is what model size? For the same compute, would you rather have 2B params/10B tokens or
   0.5B params/40B tokens?
6. **Estimate a real run.** Llama 3 8B: 8B parameters, 15T tokens. Using the rule
   `FLOPs ≈ 6 × params × tokens`, how many FLOPs? At 400 TFLOP/s per H100 at 40% utilisation, how
   many GPU-hours? How many days on 16,000 GPUs?

---

## 6. Projects to build and test

### Beginner — Character-level GPT on one book
Extend `tiny_pretrain.py`: bigger model, a full Project Gutenberg text, train/validation split.

**How to test it:** validation loss should fall below ~1.5 bits/char. Sample 500 characters at
temperature 0.2, 0.8 and 1.5 and describe the difference. If validation loss rises while training
loss falls, you are overfitting — say at which step it started.

### Intermediate — Data pipeline
Build quality filtering (length, symbol ratio, language ID), exact deduplication by hash, and fuzzy
deduplication by MinHash over a web text sample.

**How to test it:** report how many documents each stage removed and why. Hand-inspect 20 rejected
documents — if good ones are being dropped, your filter is too aggressive. That trade-off *is* the
job.

### Advanced — Continued pretraining that measurably helps
Take a small base model and continue pretraining on a domain corpus (legal, medical, a codebase)
with LLaMA-Factory.

**How to test it:** measure perplexity on held-out *domain* text before and after — it should drop.
Then measure perplexity on *general* text — if it rose sharply, you have caused catastrophic
forgetting, and the fix is mixing general data back into the corpus. Report both numbers; reporting
only the first is how people fool themselves.

### Advanced — Scaling law replication
Train 5 models of increasing size on the same data with a fixed compute budget each. Plot loss
against parameters on log-log axes.

**How to test it:** you should recover a straight line. Compare its slope to the published
Chinchilla exponent. Where your line deviates is where your training setup is suboptimal — finding
out why is the actual lesson.

---

## 7. Resources

**Build it**
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy. The canonical two hours.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — small, readable, actually trains.
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — a live leaderboard of training-speed improvements.

**Understand it**
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.)
- [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) (Chinchilla)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — unusually candid about data and infrastructure.
- [OLMo](https://allenai.org/olmo) — a genuinely open model: weights, data, code, logs. The best thing to study if you want to see a real run.

**Data**
- [FineWeb: decanting the web](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — an outstanding writeup of what data curation actually involves.
- [The Pile](https://arxiv.org/abs/2101.00027) · [Dolma](https://arxiv.org/abs/2402.00159)

---

**Next:** [02 · Fine-Tuning](../02-fine-tuning/)
