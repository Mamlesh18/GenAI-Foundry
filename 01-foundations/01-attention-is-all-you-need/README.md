# 01 · Attention Is All You Need

> Vaswani et al., 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
> Cited over 250,000 times. Every model you have heard of — GPT, Claude, Gemini, Llama — is a
> descendant of the eight pages in this paper.

---

## 1. What this is

Before 2017, the way to process a sentence was to walk through it one word at a time with a
recurrent network (RNN/LSTM). Word 50 could only be understood after words 1–49 had been
processed, in order. That has two fatal problems:

1. **It is sequential.** You cannot parallelise a loop that depends on its own previous output, so
   training is slow and cannot exploit modern GPUs.
2. **It forgets.** Information from word 1 has to survive 49 rounds of being squeezed through a
   fixed-size hidden state to reach word 50. Long-range dependencies get washed out.

This paper's answer is almost rude in its simplicity: **throw away recurrence entirely.** Let every
word look at every other word simultaneously and decide for itself which ones matter. That
operation is called **self-attention**, and it is the whole architecture.

The result — the **Transformer** — trains in parallel, models arbitrarily long-range relationships
in a single step, and scales beautifully. That last property is why the last decade happened.

---

## 2. Why it matters

| Property | RNN / LSTM | Transformer |
|---|---|---|
| Path length between two tokens | O(n) — must travel through every token between | **O(1)** — direct connection |
| Parallelisable during training | no, inherently sequential | **yes**, all positions at once |
| Long-range dependencies | degrade with distance | flat cost regardless of distance |
| Scales with more compute | poorly | **extremely well** |

That last row is the one that mattered. The Transformer did not just beat LSTMs on translation — it
turned out that if you make it bigger and feed it more data, it keeps getting better. That
observation is what the entire modern LLM industry is built on.

---

## 3. Core concepts

### 3.1 Self-attention: the one idea

Take the sentence:

> *The animal did not cross the street because **it** was too tired.*

What does "it" refer to? A human reads "tired" and resolves "it" to "animal". Self-attention lets
the model do the same thing, mechanically.

Every token produces three vectors, each by multiplying its embedding by a learned weight matrix:

| Vector | Learned via | Intuition | Library analogy |
|---|---|---|---|
| **Query** (Q) | `W_Q` | "What am I looking for?" | your search phrase |
| **Key** (K) | `W_K` | "What do I contain?" | the label on the shelf |
| **Value** (V) | `W_V` | "What do I actually pass on?" | the book itself |

The token "it" emits a Query meaning roughly *"I am a pronoun, I need a subject"*. The token
"animal" emits a Key meaning *"I am a subject noun"*. Those two vectors point in a similar
direction, so their dot product is large, so "it" pulls in a lot of "animal"'s Value.

### 3.2 The formula

The whole paper compresses to this line:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

Read left to right, it is four steps:

1. **`Q · Kᵀ`** — score every token against every other token. For `n` tokens you get an `n × n`
   matrix: row `i`, column `j` = "how much should token `i` care about token `j`?"
2. **`/ √d_k`** — *scaling*. Dot products of high-dimensional vectors grow large; large values push
   softmax into a region where gradients vanish. Dividing by the square root of the key dimension
   keeps the variance around 1. Small detail, big difference in trainability.
3. **`softmax(...)`** — turn each row of scores into weights that sum to 1. Now each token has a
   probability distribution over "who I am paying attention to".
4. **`· V`** — take the weighted average of everyone's Value vectors. That average *is* the token's
   new, context-aware representation.

So self-attention is a **learned, content-based weighted average**. That is it. Everything else in
the paper is engineering around this line.

### 3.3 Multi-head attention

One attention operation learns one kind of relationship. But "it" relates to "animal"
*grammatically* while "tired" relates to it *semantically*, and both matter.

So instead of one attention with dimension 512, run **8 attentions in parallel** with dimension 64
each, then concatenate the results and project them back to 512. Each **head** is free to
specialise — in practice, trained models develop heads that track syntax, heads that track
coreference, heads that just attend to the previous token, and heads that appear to do nothing.

Same total compute, much richer representation.

### 3.4 Positional encoding

Self-attention has a problem: it is a weighted *average*, and averages do not care about order.
"Dog bites man" and "man bites dog" would produce identical outputs. Unacceptable.

The fix: before the first layer, add a vector to each token embedding that encodes *where* the
token sits. The paper uses sinusoids of geometrically increasing wavelengths:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Why sinusoids rather than learned vectors? Because for any fixed offset `k`, `PE(pos+k)` is a
*linear function* of `PE(pos)` — so the model can learn to attend by relative distance, and it
generalises to sequence lengths never seen during training.

> **Modern note:** almost nobody uses sinusoidal encodings any more. Today's models use **RoPE**
> (Rotary Position Embedding) or **ALiBi**, which handle long contexts far better. The *problem*
> the paper identified is permanent; its specific solution was superseded. Worth knowing both.

### 3.5 The rest of the block

Around attention, each layer wraps:

- **Feed-forward network** — two linear layers with a non-linearity, applied to each position
  independently. Attention mixes information *between* tokens; the FFN processes it *within* a
  token. Roughly two-thirds of a transformer's parameters live here.
- **Residual connections** — `x + Sublayer(x)`. Gives gradients a clean path back through a deep
  stack, and lets each layer make a small edit rather than a full rewrite.
- **Layer normalisation** — keeps activations in a stable range. The paper puts it *after* the
  sublayer; modern models put it *before* (pre-LN), because it trains more stably.

### 3.6 Encoder, decoder, and what we actually use

The original paper is an **encoder–decoder** built for translation: the encoder reads the German,
the decoder writes the English. It has three attention flavours:

| Type | Where | What it does |
|---|---|---|
| Encoder self-attention | encoder | every input token sees every other input token |
| **Masked** self-attention | decoder | each token sees only tokens *before* it — no peeking at the answer being predicted |
| Cross-attention | decoder | output tokens attend to the encoder's representation of the input |

Modern LLMs (GPT, Llama, Claude, Gemini) are **decoder-only**: they keep the masked self-attention
stack and drop the encoder. That is the architecture you will actually work with. BERT went the
other way — encoder-only — and dominates embeddings and classification.

---

## 4. Examples

```bash
cd examples
pip install numpy
python self_attention_from_scratch.py
```

- [`self_attention_from_scratch.py`](examples/self_attention_from_scratch.py) — scaled dot-product
  attention, causal masking and multi-head attention in pure NumPy, with the attention matrix
  printed so you can see which token attends to which.

---

## 5. Exercises

1. **Trace the shapes.** For a batch of 2 sequences, 10 tokens each, `d_model=512`, `h=8` heads:
   write down the shape of Q, K, V, the attention score matrix, and the output. Then check your
   answer by adding print statements to the example script.
2. **Break the scaling.** Remove the `/ √d_k` term from the example and set `d_k = 512`. Print the
   softmax output. What happened to the distribution, and why does that kill learning?
3. **Prove order matters.** Run the example on `["dog", "bites", "man"]` and on
   `["man", "bites", "dog"]` *without* positional encoding. Compare the output vectors for "bites".
   Now add positional encoding and compare again.
4. **Read the causal mask.** Print the `n × n` mask used in decoder self-attention. Explain in one
   sentence why it must be applied *before* the softmax and not after.
5. **Count the parameters.** For the base model in the paper (`d_model=512`, `d_ff=2048`, 6 layers),
   compute how many parameters sit in attention versus in the feed-forward networks. Which is
   bigger, and by how much?

---

## 6. Projects to build and test

### Beginner — Attention visualiser
Take a short sentence, run it through a small pretrained model (`bert-base-uncased` via Hugging
Face `transformers`), extract the attention weights with `output_attentions=True`, and render them
as a heatmap with matplotlib.

**How to test it:** feed it *"The animal did not cross the street because it was too tired"* and
find a head where "it" attends strongly to "animal". Then swap "tired" for "wide" and check whether
the attention target moves to "street".

### Intermediate — Transformer block from scratch
Implement a full pre-LN decoder block in PyTorch: masked multi-head attention, FFN, residuals,
layer norm. No `nn.Transformer`, no `nn.MultiheadAttention`.

**How to test it:** assert your output matches `torch.nn.functional.scaled_dot_product_attention`
to within `1e-5` on random inputs, and verify the causal mask by checking that changing token `t+1`
never changes the output at position `t`.

### Advanced — Tiny GPT trained on one book
Stack your blocks into a small decoder-only model (about 6 layers, ~10M parameters), train it
character-level on a public-domain text from Project Gutenberg, and sample from it.

**How to test it:** track validation loss; it should drop below roughly 1.5 bits/char. Generate 500
characters at temperature 0.2, 0.8 and 1.5 and describe the difference. Compare your result and
your code against Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 7. Resources

**Primary**
- [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — read it. Eight pages, and more readable than its reputation.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP. The entire paper reimplemented line-by-line alongside the text.

**Explanations**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar. The best diagrams anywhere; read this *with* the paper.
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers) — Peter Bloem. Rigorous and code-first.
- [Self-Attention from Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) — Sebastian Raschka. Builds the mechanism step by step in PyTorch.
- [Attention Is All You Need — Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need) — unusually good on historical context and impact.

**Video**
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy. Two hours, worth every minute.
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown. The best visual intuition for Q/K/V.

**What came after**
- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — how position is actually encoded today.
- [FlashAttention](https://arxiv.org/abs/2205.14135) — memory-efficient attention; why long contexts became affordable.
- [A Survey of Transformers](https://arxiv.org/abs/2106.04554) — the family tree of variants.

---

**Next:** [02 · LLMs](../02-llms/)
