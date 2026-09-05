# 10 · Fine-Tuning

**Level:** Advanced · **Time:** ~8 hours

---

## 1. What you will be able to do

Decide — with evidence — whether fine-tuning is the right answer, and if it is, adapt an open model
with LoRA on a single consumer GPU and prove it beat the prompted baseline.

---

## 2. Prerequisites

- Skills: [02 · Prompt Engineering](../02-prompt-engineering/), [08 · Evaluation](../08-evaluation-and-testing/) — **both, genuinely**
- Concepts: [02 · Training](../../02-training/), [LoRA](../../02-training/03-lora/), [QLoRA](../../02-training/04-qlora/)
- A GPU: a free Colab/Kaggle T4 is enough for a 7B model with QLoRA

---

## 3. Read this before you start

Fine-tuning is the answer far less often than people expect. Work through this honestly:

| Your problem | Fine-tuning? | Actually do |
|---|---|---|
| It does not know facts about my company | **No** | [RAG](../06-rag-pipeline/) |
| It needs current information | **No** | RAG or [tools](../04-tool-calling/) |
| Output format is inconsistent | **No** | [Structured outputs](../03-structured-outputs/) |
| It is bad at reasoning through my task | **Usually no** | Better prompting, then a better model |
| I need a specific voice or style, consistently | **Yes** | Fine-tune on examples of that style |
| I need a narrow task at much lower cost | **Yes** | Fine-tune a small model to match a big one |
| My prompt is 3,000 tokens of instructions | **Yes** | Bake the instructions into weights |
| I need a specialised domain format (legal clauses, clinical notes) | **Probably** | Fine-tune on real examples |

The rule: **fine-tuning teaches behaviour, not knowledge.** It changes *how* the model responds, not
*what* it knows. Wanting it to know something is a retrieval problem.

If you cannot state a measured gap that prompting failed to close, stop here and go back to
[skill 02](../02-prompt-engineering/).

---

## 4. The concept, briefly

**Full fine-tuning** updates every weight. For a 7B model that needs roughly 60GB+ of GPU memory —
out of reach on consumer hardware.

**LoRA** freezes the original weights and trains small low-rank adapter matrices injected beside
them. You train perhaps 0.1–1% of the parameters, get most of the quality, and the adapter is a few
megabytes you can swap at will.

**QLoRA** additionally loads the base model in 4-bit. A 7B model fits in about 6GB, which is a free
Colab T4. This is what made fine-tuning accessible to everyone.

---

## 5. Build it

1. **Measure the prompted baseline.** On your evaluation set from skill 08. Write the number down.
   Without it you cannot claim any improvement, and roughly half of fine-tuning projects turn out
   not to beat a well-written prompt.

2. **Build the dataset.** This is 80% of the work and 100% of the outcome.
   - 500–1,000 high-quality examples beats 10,000 mediocre ones, reliably.
   - Format must exactly match inference-time format — a mismatch here silently ruins the result.
   - Consistency matters more than volume: inconsistent labels teach inconsistency.
   - Hold out 10–20% for validation and never train on it.

3. **Inspect your data by hand.** Read 50 examples. You will find errors. Everyone does.

4. **Pick a base model.** Start smaller than you think — a fine-tuned 7B often beats a prompted 70B
   on a narrow task, at a fraction of the cost. Check the licence permits your use.

5. **Set up QLoRA.** `unsloth` or `peft` + `bitsandbytes` + `trl`. Start with rank 16, alpha 32,
   dropout 0.05, targeting the attention projections.

6. **Train.** 2–3 epochs. Watch training *and* validation loss. Validation loss rising while
   training loss falls is overfitting — stop.

7. **Evaluate against the baseline.** Same evaluation set, same metrics. If it does not beat the
   prompted baseline, the honest conclusion is that your data is not good enough yet.

8. **Check for catastrophic forgetting.** Test capabilities you did *not* fine-tune on. Narrow
   fine-tuning often degrades general ability, and people ship without noticing.

9. **Iterate on data, not hyperparameters.** When results disappoint, the answer is almost always
   more or better examples — not a different learning rate.

10. **Deploy the adapter.** Serve base + adapter with vLLM. Adapters are small, so you can host
    several specialised ones against one base model.

---

## 6. Checkpoint

- [ ] You can state, with numbers, why prompting was insufficient
- [ ] Dataset of 500+ examples, manually inspected, with a held-out validation split
- [ ] Training and validation loss curves recorded, with no overfitting at your chosen checkpoint
- [ ] Your fine-tuned model beats the prompted baseline on your evaluation set by a stated margin
- [ ] You have checked for catastrophic forgetting on unrelated capabilities
- [ ] You can state the cost per request before and after
- [ ] The adapter serves behind an API endpoint
- [ ] You could reproduce the whole run from your scripts

---

## 7. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Skipping the prompted baseline | You cannot tell whether it helped | Measure first, always |
| Fine-tuning to add knowledge | It hallucinates more confidently | Use RAG |
| Training/inference format mismatch | Quality collapses inexplicably | Byte-identical formatting |
| Too few examples | It memorises instead of generalising | 500+, quality over quantity |
| Never reading the data | Garbage in, garbage out | Hand-inspect 50 examples |
| No validation split | Overfitting goes unnoticed | Hold out 10–20% |
| Tuning hyperparameters instead of data | Marginal gains, lots of wasted GPU time | Improve the dataset |
| Not testing general capability | Ships with a regression nobody measured | Test unrelated tasks too |
| Starting with a 70B model | Slow, expensive, usually unnecessary | Start at 7B and scale only if needed |

---

## 8. Going further

- **Preference tuning:** DPO on preference pairs, when you have "A is better than B" data rather
  than gold answers. See [preference tuning](../../02-training/05-preference-tuning/)
- **Distillation:** use a large model to generate training data for a small one
- **Multiple adapters:** one base model, several task-specific adapters, swapped at request time
- **Quantize the result** for cheaper serving — see [quantization](../../03-inference/03-quantization/)

**Links:** [LoRA](https://arxiv.org/abs/2106.09685) ·
[QLoRA](https://arxiv.org/abs/2305.14314) ·
[Unsloth](https://github.com/unslothai/unsloth) — fastest way to start, free Colab notebooks ·
[Hugging Face PEFT](https://huggingface.co/docs/peft) ·
[TRL](https://huggingface.co/docs/trl)
