# 04 · Types of Prompting

Module 03 covered how to write *one* good prompt. This module is the toolbox: named, researched
techniques for *structuring* a prompt, each solving a different failure.

The skill being built here is not memorising them. It is **diagnosis** — recognising which failure
you are looking at, and reaching for the technique that addresses it.

---

## 1. The decision table

Start here. Find your symptom, use the technique.

| Your problem | Technique | Cost |
|---|---|---|
| Simple task, model already knows it | [Zero-shot](#31-zero-shot) | cheapest |
| Output format or style is inconsistent | [Few-shot](#32-few-shot) | + example tokens |
| It gets multi-step reasoning wrong | [Chain-of-Thought](#33-chain-of-thought-cot) | + reasoning tokens |
| CoT is right sometimes and wrong others | [Self-Consistency](#34-self-consistency) | n× the calls |
| It needs live data or a calculator | [ReAct](#35-react-reason--act) | multi-turn |
| The task is too big for one pass | [Prompt Chaining](#36-prompt-chaining) | multi-call |
| It needs to explore alternatives | [Tree of Thoughts](#37-tree-of-thoughts-tot) | expensive |
| The first draft is nearly right | [Self-Refine](#38-self-refine--reflexion) | 2–3× calls |
| You need machine-readable output | [Structured Output](#39-structured-output) | free |
| It invents facts | [RAG prompting](#310-rag-grounded-prompting) | + retrieval |
| Tone or expertise level is wrong | [Role Prompting](#311-role--persona-prompting) | free |
| You cannot work out a good prompt yourself | [Meta Prompting](#312-meta-prompting) | one extra call |

---

## 2. The one rule

**Escalate only when you need to.** Each technique down that table costs more tokens, more latency
and more complexity. Start zero-shot. Measure. Add the cheapest technique that fixes the actual
failure. Chain-of-thought on a sentiment classifier is wasted money; few-shot on a task the model
already nails is wasted context.

---

## 3. The techniques

### 3.1 Zero-shot

Just ask. No examples, no scaffolding.

```
Classify the sentiment of this review as positive, negative or neutral.

Review: "The battery lasts forever but the screen scratches if you look at it."
Sentiment:
```

**Use when:** the task is common and the output shape is obvious.
**Fails when:** the task is unusual, or you need a specific format.
**Note:** modern instruction-tuned models are far better at zero-shot than the models these
techniques were originally invented for. Always try zero-shot first — you may be done.

---

### 3.2 Few-shot

Show 2–5 worked examples inside the prompt, then give the real input. The model infers the pattern
without any training. This is **in-context learning**, and it remains one of the highest-return
techniques available.

```
Review: "Battery dies in two hours."          Sentiment: negative
Review: "Arrived on time. Does what it says." Sentiment: neutral
Review: "I have bought three more since."     Sentiment: positive

Review: "The battery lasts forever but the screen scratches easily."
Sentiment:
```

**Why it works so well:** examples communicate format, tone, edge-case handling and implicit rules
all at once — things that take paragraphs to describe and are still ambiguous when you do.

**Getting it right:**
- Cover your edge cases in the examples, not just the easy path.
- Keep the format *byte-identical* across examples. Inconsistency is what the model copies.
- Balance the classes. Three positives and one negative biases the output toward positive.
- Order matters more than it should — especially the last example. Shuffle and re-measure.
- 3–5 examples is usually the sweet spot; returns flatten quickly after that.

📄 [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

---

### 3.3 Chain-of-Thought (CoT)

Make the model reason *before* it answers.

Because an LLM generates one token at a time with fixed computation per token, an immediate answer
gives it no room to work. Reasoning tokens are literally the space in which the computation happens.

**Zero-shot CoT** — one line, famously effective:

```
Q: A shop has 23 apples. It uses 20 for lunch and buys 6 more. How many are left?
A: Let's think step by step.
```

**Few-shot CoT** — include the reasoning in your examples:

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many now?
A: Roger started with 5. Two cans of 3 balls is 6 balls. 5 + 6 = 11. The answer is 11.

Q: A shop has 23 apples, uses 20, then buys 6 more. How many are left?
A:
```

**Use when:** arithmetic, logic, multi-step planning, anything where a wrong intermediate step
poisons the result.
**Do not use when:** the task is a single lookup or classification — you pay for tokens that add
nothing.

> **Important 2026 caveat:** models with built-in reasoning (Claude with extended thinking, Gemini
> thinking models, the o-series) already do this internally. Bolting "think step by step" onto them
> is redundant and occasionally harmful. Know your model before reaching for CoT.

📄 [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) ·
📄 [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)

---

### 3.4 Self-Consistency

CoT, but sampled many times, taking the majority answer.

```
1. Run the same CoT prompt 5 times at temperature ~0.7
2. Extract the final answer from each
3. Return the most common one
```

One reasoning chain can take a wrong turn. Five independent chains rarely take the *same* wrong
turn, so the correct answer usually wins the vote. It is an ensemble method wearing a prompt.

**Cost:** exactly n× the calls. Worth it when accuracy matters more than money — and only for tasks
with a checkable, discrete answer.

📄 [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)

---

### 3.5 ReAct (Reason + Act)

Interleave **thinking** with **doing**. The model reasons about what it needs, calls a tool, reads
the result, and repeats.

```
Thought: I need today's exchange rate. I do not know it.
Action:  search("USD to INR rate today")
Observation: 1 USD = 88.4 INR
Thought: Now I can convert 250 USD.
Action:  calculator("250 * 88.4")
Observation: 22100
Answer:  250 USD is about 22,100 INR.
```

This is the foundation of every agent you will ever build. It fixes the two biggest LLM weaknesses
at once: stale knowledge and inability to compute. The loop is grounded in observations from the
real world rather than the model's priors.

**Watch out for:** loops that never terminate (always cap iterations), tool errors the model
ignores, and the security surface — tool output is untrusted input. See
[07 · Agents](../../07-agents/).

📄 [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)

---

### 3.6 Prompt Chaining

Break one large task into a pipeline of focused calls, each consuming the last one's output.

```
Document -> [extract key claims] -> [verify each against source] -> [write summary] -> Output
```

**Why:** each step gets the model's full attention on a narrow job; you can inspect and test each
stage; you can use a cheap model for easy steps and an expensive one only where it counts.

**Cost:** more calls, more latency, and errors compound down the chain — validate between steps.

---

### 3.7 Tree of Thoughts (ToT)

Generate several candidate next steps, evaluate them, keep the promising branches, and explore
further. Search, rather than a single linear chain.

Powerful for puzzles, planning and creative problems with a large search space. Also expensive and
fiddly. In practice, prompt chaining plus self-consistency covers most real needs. Know this exists;
reach for it rarely.

📄 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)

---

### 3.8 Self-Refine / Reflexion

Generate, critique, revise.

```
1. "Write a function that parses ISO-8601 dates."
2. "Review the function above. List concrete bugs and missing edge cases."
3. "Rewrite it, fixing every issue you listed."
```

Critiquing is an easier task than generating, so the second pass often catches real problems. Two
honest limits: the model may critique things that are already fine, and it will not reliably find
errors it lacks the knowledge to see. Grounding the critique in something external — test results,
a linter, a rubric — makes it far more reliable than pure self-review.

📄 [Self-Refine](https://arxiv.org/abs/2303.17651) · 📄 [Reflexion](https://arxiv.org/abs/2303.11366)

---

### 3.9 Structured Output

Demand a machine-readable shape and validate it. Non-negotiable for anything that feeds code.

```
Return ONLY valid JSON matching this schema, with no markdown fences:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "key_phrases": [<string>, ...]
}
```

**Do it properly:**
- Use the provider's native support where it exists — JSON mode, `response_format`, tool/function
  calling, or a schema parameter. Constrained decoding beats asking nicely.
- Validate every response against a real schema (`pydantic`, `jsonschema`). Never `json.loads()` a
  model response without a `try`.
- Retry with the parse error appended to the prompt — self-correction on a concrete error message
  works well.
- Watch for markdown fences (` ```json `) wrapping the output. Strip them defensively.

---

### 3.10 RAG-grounded prompting

Retrieve relevant documents, put them in the prompt, and instruct the model to answer *only* from
them.

```
Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: NOT_FOUND.
Cite the source id for every claim.

<context>
[1] {chunk_one}
[2] {chunk_two}
</context>

Question: {question}
```

The three lines doing the work: *only from the context*, *an explicit way to say "I don't know"*,
and *citations* so a human can verify. Full treatment in [06 · RAG](../../06-rag/).

---

### 3.11 Role / Persona prompting

`"You are an experienced paediatric nurse explaining to a worried parent."`

Conditions vocabulary, depth, tone and what the model considers relevant. Cheap and genuinely
effective for *style and framing*.

**Be honest about its limits:** telling a model it is an expert does not add knowledge it does not
have. The evidence that personas improve factual accuracy is weak. Use roles to shape *how* it
answers, not to make it *know* more.

---

### 3.12 Meta prompting

Use the model to write or improve the prompt.

```
I want to build a prompt that extracts structured data from invoices.
Write an optimal prompt for this, then list three ways it could be misread
and fix each.
```

Surprisingly good, especially as a starting point when you are stuck. Always test the generated
prompt against real inputs — it will read beautifully and still be wrong.

---

### 3.13 Worth knowing about

| Technique | One line |
|---|---|
| **Least-to-Most** | Decompose into subproblems, solve simplest first, feed answers forward. |
| **Generated Knowledge** | Ask the model to state relevant facts first, then answer using them. |
| **Step-Back** | Ask a more general question first, then apply the principle to the specific case. |
| **Program-Aided (PAL)** | Have it write code to compute the answer instead of doing arithmetic in text. |
| **Directional Stimulus** | Provide hints or keywords to steer without dictating the answer. |
| **Contrastive CoT** | Show both correct *and* incorrect reasoning so it learns what to avoid. |
| **Prompt Ensembling** | Ask the same question several different ways, aggregate the answers. |

The [Prompt Report](https://arxiv.org/abs/2406.06608) catalogues 58 of these systematically.

---

## 4. Examples

```bash
cd examples
pip install -r ../../02-llms/examples/requirements.txt
python 01_zero_and_few_shot.py
```

| File | Technique |
|---|---|
| [`01_zero_and_few_shot.py`](examples/01_zero_and_few_shot.py) | Zero-shot vs few-shot on a task where format consistency matters |
| [`02_chain_of_thought.py`](examples/02_chain_of_thought.py) | Direct answer vs CoT on multi-step reasoning |
| [`03_self_consistency.py`](examples/03_self_consistency.py) | Majority voting over n sampled reasoning chains |
| [`04_react_loop.py`](examples/04_react_loop.py) | A minimal ReAct agent with a calculator and a fake search tool |
| [`05_structured_output.py`](examples/05_structured_output.py) | JSON output with schema validation and retry-on-failure |

---

## 5. Exercises

1. **Build the ladder.** Take one hard task. Solve it zero-shot, then few-shot, then CoT, then
   self-consistency. Record accuracy and token cost at each rung. Where did the curve flatten?
2. **Poison the examples.** In a few-shot prompt, make one example use a slightly different output
   format. Run 20 inputs. How often does the model copy the odd one out?
3. **Order effects.** Take four few-shot examples and try all orderings you can stomach. Does
   accuracy move? Which position matters most?
4. **Class imbalance.** Build a sentiment classifier with 4 positive and 1 negative example. Test
   on a balanced set. Measure the bias, then fix it.
5. **CoT where it hurts.** Apply chain-of-thought to a trivial classification task. Compare
   accuracy and cost against zero-shot. Write down what you conclude.
6. **Break ReAct.** Give the agent a question that needs a tool it does not have. What does it do?
   Now add a max-iteration cap and a "no suitable tool" escape and try again.
7. **Force a parse failure.** Get a model to return malformed JSON (long output plus `max_tokens`
   works). Confirm your validator catches it and your retry recovers.
8. **Persona test.** Ask a factual question as a plain assistant and as "a Nobel laureate in
   physics". Score both for accuracy over 10 questions. Did the persona help, or only change tone?

---

## 6. Projects to build and test

### Beginner — Technique comparison harness
One task, six techniques, one script. Print a table of accuracy, tokens, latency and cost.

**How to test it:** use a task with objectively checkable answers (GSM8K-style word problems).
Publish the table in your README. Anyone should be able to rerun it and get similar numbers.

### Beginner — Few-shot example selector
Given a bank of 50 labelled examples, pick the 3 most similar to the incoming input (by embedding
similarity) and build the few-shot prompt dynamically.

**How to test it:** compare dynamic selection against three fixed examples across 50 test inputs.
Dynamic selection should win — quantify by how much. Connects to [06 · Embeddings](../06-embeddings/).

### Intermediate — Self-consistency calculator
A CLI that answers maths word problems by sampling n reasoning chains and majority-voting.

**How to test it:** run against 50 GSM8K problems at n=1, 3, 5, 10. Plot accuracy against cost.
Find the point where extra samples stop paying for themselves.

### Intermediate — ReAct agent with real tools
Extend the ReAct example with a real search API, a Python execution sandbox, and a file reader.

**How to test it:** ask questions unanswerable without tools ("what is the population of Chennai
divided by the number of days since the Transformer paper was published?"). Verify each tool call
in the trace. Then try to make it loop forever, and fix that.

### Advanced — Prompt technique router
A classifier that inspects an incoming request and automatically picks the cheapest technique
likely to succeed — zero-shot for simple lookups, CoT for reasoning, ReAct when tools are needed.

**How to test it:** build a labelled set of 100 requests. Measure the router's accuracy against
always-CoT and always-zero-shot baselines, on both quality and cost. The win condition is matching
always-CoT quality at meaningfully lower cost.

---

## 7. Resources

**Reference**
- [Prompt Engineering Guide — Techniques](https://www.promptingguide.ai/techniques) — a page per technique with papers and examples.
- [The Prompt Report](https://arxiv.org/abs/2406.06608) — 58 techniques, systematically surveyed and taxonomised.
- [Anthropic — Prompt engineering techniques](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)

**The papers behind the techniques**
- [Few-shot / GPT-3](https://arxiv.org/abs/2005.14165)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903) · [Zero-shot CoT](https://arxiv.org/abs/2205.11916)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [Self-Refine](https://arxiv.org/abs/2303.17651) · [Reflexion](https://arxiv.org/abs/2303.11366)
- [Least-to-Most](https://arxiv.org/abs/2205.10625)
- [Program-Aided Language Models](https://arxiv.org/abs/2211.10435)

**Practice**
- [DSPy](https://dspy.ai/) — stop hand-tuning prompts; compile and optimise them programmatically.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — the standard maths-reasoning benchmark for testing these techniques.

---

**Previous:** [03 · Prompting](../03-prompting/) ·
**Next:** [05 · Tokenization](../05-tokenization/)
