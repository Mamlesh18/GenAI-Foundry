# 02 · Fine-Tuning (SFT)

Pretraining produced a brilliant autocomplete. Ask it a question and it may cheerfully generate
twenty more questions, because continuing text is all it has ever done.

**Supervised fine-tuning** teaches it to answer instead of continue.

---

## 1. What this is

Take a pretrained base model. Train it further on a curated set of `(instruction, good response)`
pairs. Same next-token objective, same loss function — the only thing that changes is the data, and
one crucial detail about which tokens you compute loss on.

```
BASE MODEL                                  AFTER SFT
---------------------------------------     ---------------------------------------
prompt: "What is the capital of France?"     prompt: "What is the capital of France?"

output: "What is the capital of Spain?       output: "The capital of France is Paris."
         What is the capital of Italy?
         What is the largest city in..."     it ANSWERS
it CONTINUES the pattern
```

The scale is dramatically smaller than pretraining: **thousands of examples, not trillions of
tokens.** Hours on one GPU, not weeks on thousands. This stage is genuinely accessible.

### Loss masking: the detail that matters most

This is the one mechanical difference from pretraining, and getting it wrong silently ruins results.

You train on the full sequence — prompt and response concatenated — but you only compute **loss on
the response tokens**:

```
tokens :  <user> What is 2+2? <assistant> The answer is 4.
loss on:  ─────── masked out ───────────── ✓  ✓   ✓  ✓ ✓
```

Why? You want the model to learn *how to respond to* instructions, not *how to generate*
instructions. If you compute loss on the prompt too, you are partly training it to write user
messages — which is not the job, and dilutes the signal you actually want.

In code this is done by setting masked label positions to `-100`, the index PyTorch's
`cross_entropy` ignores. The from-scratch example in this folder shows exactly this.

### Chat templates

The model needs to know where the user stops and the assistant starts. That is what a **chat
template** does — special tokens wrapping each turn:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

Every model family has its own. Llama's differs from Qwen's differs from Mistral's.

> **The single most common fine-tuning bug:** training with one template and serving with another.
> Quality collapses and the cause is invisible in the loss curve. Use
> `tokenizer.apply_chat_template()` rather than building strings by hand, and in LLaMA-Factory set
> `template:` to match your model exactly.

---

## 2. Why it matters

SFT is where a model becomes *yours*. It is how you get:

- A consistent voice, format or persona without a 3,000-token system prompt
- A small model doing a narrow task as well as a large one, at a fraction of the cost
- Domain-specific output shapes — clinical notes, legal clauses, your internal ticket format
- Reliable behaviour that prompting only achieves sometimes

**And what it does not get you: knowledge.** SFT teaches behaviour. Training on 500 Q&A pairs about
your product does not install those facts reliably — it teaches the model to answer *confidently in
that style*, which if anything makes hallucination worse. Facts belong in [RAG](../../06-rag/).

---

## 3. The dataset is the whole job

80% of the work, and effectively 100% of the outcome. Everything else is hyperparameters that
barely matter by comparison.

| Principle | Detail |
|---|---|
| **Quality over quantity** | 500–1,000 excellent examples beat 10,000 mediocre ones, reliably. |
| **Consistency over volume** | Inconsistent labels teach inconsistency. Ten annotators with different styles will produce a model with no style. |
| **Cover your edge cases** | Include the hard, ambiguous and should-refuse cases. A dataset of easy examples produces a model that fails exactly where you need it. |
| **Match inference format** | Byte-identical to what you will send in production. Mismatches here are silent and fatal. |
| **Hold out 10–20%** | And never train on it. |
| **Read 50 examples by hand** | You will find errors. Everyone does. This is not optional. |

### Where data comes from

- **Human written** — best quality, slowest, most expensive
- **Existing logs** — your support tickets, code reviews, past work. Usually the best source people already have and overlook.
- **Distillation** — a large model generates training data for a small one. Fast and effective; check the source model's licence permits it.
- **Public datasets** — [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5), [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). Good for learning, rarely right for a specific task.

### Catastrophic forgetting

Fine-tuning narrowly makes a model worse at everything else. Train hard on JSON extraction and it
may lose conversational ability, multilingual competence, or basic reasoning.

Mitigations: fewer epochs (2–3, not 10), lower learning rate, mix ~10–20% general instruction data
into your set, prefer LoRA over full fine-tuning, and **always test capabilities you did not train
on.** People ship regressions constantly because they only measured the thing they were optimising.

---

## 4. Examples

### From scratch — see loss masking work

```bash
pip install torch transformers
python examples/sft_from_scratch.py
```

[`sft_from_scratch.py`](examples/sft_from_scratch.py) runs a complete SFT loop on a tiny model on
CPU. It prints the label tensor so you can *see* which positions are masked to `-100`, then trains
with and without masking so you can compare what the model learns. Under 200 lines, no `trl`, no
`peft` — just the mechanism.

### Real SFT with LLaMA-Factory

```bash
llamafactory-cli train examples/llamafactory_sft.yaml      # train
llamafactory-cli chat  examples/llamafactory_chat.yaml     # talk to it
llamafactory-cli export examples/llamafactory_merge.yaml   # merge adapter into base
```

[`build_dataset.py`](examples/build_dataset.py) generates a correctly-formatted Alpaca dataset plus
the `dataset_info.json` entry LLaMA-Factory needs — the registration step people always miss.

### Dataset formats

**Alpaca** — single-turn, simplest:

```json
[
  {
    "instruction": "Classify the sentiment of this review.",
    "input": "The battery dies in two hours.",
    "output": "negative"
  }
]
```

**ShareGPT** — multi-turn conversations:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is 2+2?"},
      {"from": "gpt",   "value": "4"},
      {"from": "human", "value": "And times 3?"},
      {"from": "gpt",   "value": "12"}
    ],
    "system": "You are a concise maths tutor."
  }
]
```

Register a custom file in `LLaMA-Factory/data/dataset_info.json`:

```json
"my_task": {
  "file_name": "my_task.json",
  "formatting": "alpaca",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
}
```

---

## 5. Exercises

1. **See the mask.** Run `sft_from_scratch.py` and read the printed label tensor. Count the `-100`
   positions. Confirm they line up exactly with the prompt.
2. **Break the mask.** Disable masking (compute loss on everything). Train both ways and compare
   what the model generates. What did the unmasked version learn to do that you did not want?
3. **Template mismatch.** Train with one chat template, then generate using a different one.
   Observe the quality collapse. Note that the training loss curve gave you no warning at all.
4. **Quality vs quantity.** Build 100 excellent examples and 1,000 sloppy ones. Train on each.
   Evaluate both on the same held-out set. Report the result.
5. **Overfit on purpose.** Train 10 epochs on 50 examples. Plot training and validation loss. Mark
   the step where validation loss turns upward — that is your real stopping point.
6. **Measure forgetting.** Fine-tune narrowly on JSON extraction. Then test general conversation,
   arithmetic and another language. Quantify what you lost.
7. **Beat the prompt.** Before fine-tuning, write the best prompt you can and score it. Then
   fine-tune. Did you actually win? Half the time the honest answer is no.

---

## 6. Projects to build and test

### Beginner — Style transfer fine-tune
Fine-tune a small model to write in one consistent voice (terse technical, or friendly explainer)
using 200–500 examples.

**How to test it:** 20 held-out prompts scored blind by someone else against the base model with a
style prompt. If a prompt gets you the same result, you have learned something useful about when
fine-tuning is worth it.

### Intermediate — Domain extractor
Fine-tune a 1–3B model to extract structured fields from documents in your domain, and beat a
prompted 70B model on accuracy *and* cost.

**How to test it:** labelled test set of 100 documents. Report accuracy, latency and cost per
1,000 documents for both. The small fine-tuned model should win decisively on cost — state the
accuracy gap honestly.

### Intermediate — Dataset quality experiment
Same base model, same hyperparameters, four dataset variants: 100 clean, 100 noisy, 1,000 clean,
1,000 noisy.

**How to test it:** one evaluation set across all four. Produce the 2×2 table. This experiment
teaches more about fine-tuning than any tutorial.

### Advanced — Distillation pipeline
Use a large model to generate training data, filter it for quality, and fine-tune a small model on
the result.

**How to test it:** measure the small model against both the teacher and its own prompted baseline.
Report cost per request for all three. Then measure how much the filtering step mattered by
training on unfiltered data too.

---

## 7. Resources

- [Finetuning LLMs](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — Sebastian Raschka. The clearest overview of the landscape.
- [Hugging Face — Supervised fine-tuning chapter](https://huggingface.co/learn/llm-course/chapter11/1)
- [TRL `SFTTrainer` docs](https://huggingface.co/docs/trl/sft_trainer) — the standard library, including completion-only loss
- [LLaMA-Factory SFT guide](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)
- [Unsloth notebooks](https://github.com/unslothai/unsloth) — free Colab, 2× faster, working end to end
- [InstructGPT paper](https://arxiv.org/abs/2203.02155) — where SFT-then-preference-tuning was established
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — 1,000 carefully curated examples beating far larger datasets. The best evidence for "quality over quantity".
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) — the project that showed cheap instruction tuning works

---

**Previous:** [01 · Pretraining](../01-pretraining/) · **Next:** [03 · LoRA](../03-lora/)
