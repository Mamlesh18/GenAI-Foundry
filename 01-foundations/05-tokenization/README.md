# 05 · Tokenization

Models do not read text. They read integers. Tokenization is the translation layer, and almost
every strange LLM behaviour you have ever seen traces back to it.

---

## 1. What this is

A tokenizer splits text into **tokens** — chunks that sit somewhere between a character and a word
— and maps each to an integer id from a fixed vocabulary.

```
"Tokenization is weird"  ->  ["Token", "ization", " is", " weird"]  ->  [30642, 1634, 374, 8987]
```

Notice: `"Token"` and `"ization"` are separate, the leading space belongs to the token, and a
four-word phrase became four tokens only by coincidence.

**Rough rule of thumb for English:** 1 token ≈ 4 characters ≈ ¾ of a word. 1,000 tokens is about
750 words. That ratio is much worse for code, for languages that do not use spaces, and for
non-Latin scripts.

### Why not just use words or characters?

| Approach | Vocabulary | Problem |
|---|---|---|
| Characters | ~100 | Sequences become enormous; attention is O(n²) |
| Words | millions | Every typo and rare name is `<UNK>`; the embedding table explodes |
| **Subwords** | 32K–256K | **Just right** — common words are one token, rare words decompose |

Subword tokenization means the model can represent *any* string, including words invented after
training, by falling back to smaller pieces.

### Byte Pair Encoding (BPE)

The dominant algorithm. Training it is almost embarrassingly simple:

1. Start with a vocabulary of individual bytes.
2. Count every adjacent pair in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until you reach the target vocabulary size.

Frequent sequences ("the", "ing", "def ") become single tokens; rare ones stay decomposed. Modern
variants — byte-level BPE, WordPiece, Unigram, SentencePiece — differ in details, not in spirit.

---

## 2. Why it matters

**Cost and context are measured in tokens.** Your bill and your context window are both counted in
this unit, so knowing what inflates the count is a direct financial concern.

**It explains the "stupid" failures.** The famous ones all live here:

| Behaviour | Tokenizer explanation |
|---|---|
| Cannot count the r's in "strawberry" | It sees ~3 tokens, not 10 letters. The individual r's are not separate objects to it. |
| Bad at reversing strings | Same reason — no character-level access. |
| Arithmetic errors on long numbers | "12345" may split as "123"+"45"; digit alignment is arbitrary. |
| Worse in Hindi or Tamil than English | Non-Latin scripts often cost 2–5× more tokens for the same meaning. |
| Odd glitches on rare strings | Tokens that appeared in tokenizer training but barely in model training have near-random embeddings — the `SolidGoldMagikarp` phenomenon. |

**It is a fairness issue.** If the same sentence costs three times more tokens in your language, you
pay three times more and fit one third as much into context. This is a real and measured
inequality across languages.

---

## 3. Examples

```bash
pip install tiktoken transformers
```

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
for text in ["strawberry", "स्ट्रॉबेरी", "def fib(n):", "1234567890"]:
    ids = enc.encode(text)
    print(f"{text!r:20} -> {len(ids):2d} tokens  {[enc.decode([i]) for i in ids]}")
```

Play with it visually first: **<https://tiktokenizer.vercel.app>** — paste text, see the split
live. Ten minutes there is worth more than this whole README.

---

## 4. Exercises

1. **Count the r's yourself.** Tokenize "strawberry". Print each token. Now explain to someone else
   exactly why the model gets this wrong, without saying "it's bad at counting".
2. **Language tax.** Take the same paragraph in English, Hindi, Tamil and Chinese. Compare token
   counts. Compute the cost multiplier for each.
3. **Code vs prose.** Tokenize 500 characters of Python and 500 characters of English. Which is
   denser, and why does that matter for a coding assistant's context window?
4. **Whitespace matters.** Compare `"hello"`, `" hello"`, `"Hello"` and `"HELLO"`. Different ids?
   What does that imply about prompts that begin mid-sentence?
5. **Train your own BPE.** Implement the four-step algorithm above on a page of text with a target
   vocabulary of 300. Print the merges in order. The first ones will be unsurprising; the
   interesting ones start around merge 50.
6. **Compare tokenizers.** Run the same text through GPT-4's `cl100k_base`, Llama's tokenizer and
   Gemini's. Where do they disagree most?

---

## 5. Projects to build and test

### Beginner — Token cost calculator
A CLI that takes a file and a model name and reports token count, estimated cost, and how much of
the context window it would consume.

**How to test it:** check your count against the provider's own `usage` field on a real API call.
They should match, or you have picked the wrong encoding.

### Intermediate — BPE from scratch
Implement training and encoding/decoding yourself, no libraries.

**How to test it:** assert `decode(encode(text)) == text` for a few hundred varied strings
including emoji, accents and code. Then compare your merge list against `tiktoken` trained on the
same corpus — you should see substantial overlap in the early merges.

### Intermediate — Multilingual tokenization audit
Measure token counts for the same content across 10 languages and publish a table of the cost
multiplier relative to English.

**How to test it:** use a parallel corpus (the same text professionally translated) so you are
comparing meaning, not length. Write up the fairness implications.

---

## 6. Resources

- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Karpathy, 2 hours. The definitive treatment; build one alongside him.
- [minbpe](https://github.com/karpathy/minbpe) — the accompanying minimal, readable implementation.
- [Tiktokenizer](https://tiktokenizer.vercel.app) — interactive visualiser. Start here.
- [Hugging Face — Tokenizers chapter](https://huggingface.co/learn/llm-course/chapter6/1) — free and thorough.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — the original BPE-for-NLP paper.
- [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) — anomalous tokens, and what they reveal.
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)

---

**Previous:** [04 · Prompting Techniques](../04-prompting-techniques/) ·
**Next:** [06 · Embeddings](../06-embeddings/)
