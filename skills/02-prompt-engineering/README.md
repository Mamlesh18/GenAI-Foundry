# 02 · Prompt Engineering

**Level:** Beginner · **Time:** ~4 hours

---

## 1. What you will be able to do

Look at a bad model output, correctly diagnose *why* it is bad, and apply the cheapest technique
that fixes it — then prove the fix worked with numbers rather than an impression.

---

## 2. Prerequisites

- Skill: [01 · First LLM App](../01-first-llm-app/)
- Concepts: [03 · Prompting](../../01-foundations/03-prompting/) and
  [04 · Prompting Techniques](../../01-foundations/04-prompting-techniques/)

---

## 3. The concept, briefly

Prompting is diagnosis, not incantation. There is no magic phrase. There is a failure, a cause, and
a technique that addresses that cause.

The skill is the mapping:

| Symptom | Cause | Technique |
|---|---|---|
| Format varies between runs | You described the format instead of showing it | Few-shot examples |
| Multi-step reasoning is wrong | No room to compute before answering | Chain-of-thought |
| Right sometimes, wrong others | Single sample, high variance | Self-consistency |
| Invents facts | No grounding, no permission to decline | Retrieval + escape hatch |
| Too verbose / wrong register | No role or length constraint | Role prompting + explicit limits |
| Ignores an instruction | Buried mid-prompt, or phrased as a prohibition | Move it to the end; phrase positively |
| Output will not parse | Asked politely for JSON | Schema + validation + native JSON mode |

The second half of the skill is **measurement**. A prompt change that you cannot demonstrate is an
improvement is a superstition. You need a test set before you need a clever technique.

---

## 4. Build it

1. **Pick a real task with checkable answers.** Classification, extraction or maths — something
   where "correct" is not a matter of taste. Sentiment on product reviews works well.

2. **Build a test set of 20 inputs with known correct answers.** This is the actual work. Include
   at least five genuinely hard or ambiguous cases; a test set of easy examples teaches you nothing.

3. **Write a runner.** A function that takes a prompt template, runs all 20 inputs, and reports
   accuracy, mean tokens and mean latency. Everything after this is one command.

4. **Establish the zero-shot baseline.** Simplest possible prompt. Record the numbers. Do not skip
   this — you may find you are already done.

5. **Add the six anatomy components one at a time.** Role, task, context, examples, format,
   constraints. Rerun after each. Note which one moved the number, and by how much.

6. **Try few-shot.** Three examples covering your edge cases. Rerun. Then deliberately make one
   example inconsistent and watch what the model copies.

7. **Try chain-of-thought.** Rerun. Compare the accuracy gain against the token cost. On an easy
   task it will cost more and gain nothing — that is a result worth having.

8. **Try self-consistency** on whichever technique won. Five samples, majority vote. Note the
   accuracy gain and the 5× cost.

9. **Write up the table.** Six rows, four columns: technique, accuracy, tokens, latency. Decide
   which you would actually ship, and why.

10. **Try to break the winner.** Adversarial inputs, empty input, input in another language, input
    containing "ignore your instructions". Fix what breaks.

---

## 5. Checkpoint

- [ ] You have a results table comparing at least four techniques on the same test set
- [ ] You can state, with a number, how much each technique gained or cost
- [ ] Your best prompt beats the zero-shot baseline by a margin you can defend
- [ ] You can explain a case where a *more advanced* technique made things worse
- [ ] Your prompt survives an input containing an injection attempt
- [ ] Someone else can run your harness and reproduce your numbers

The real checkpoint: given a new failing prompt, you can name the likely cause and the fix in under
a minute.

---

## 6. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Judging by one example | You optimise for a sample of one and ship a regression | Always 20+ test cases |
| Changing several things at once | You cannot tell what helped | One variable per run |
| Only easy test cases | Everything scores 100%, you learn nothing | Deliberately include hard cases |
| Reaching for CoT immediately | Paying 5× for no gain | Start zero-shot, escalate on evidence |
| Inconsistent few-shot examples | The model copies your inconsistency | Byte-identical format across examples |
| Imbalanced few-shot classes | Systematic bias toward the over-represented class | Balance them, then verify |
| Phrasing constraints as prohibitions | "Do not be verbose" is weakly followed | "Answer in under 50 words" |
| No escape hatch | It invents an answer rather than declining | "If you cannot determine this, reply UNKNOWN" |

---

## 7. Going further

- **Next skill:** [03 · Structured Outputs](../03-structured-outputs/)
- Add an LLM-as-judge scorer for tasks without objective answers — then check the judge against
  your own labels before trusting it
- Version your prompts in git and make the harness run in CI
- Try [DSPy](https://dspy.ai/) — optimise prompts programmatically instead of by hand

**Links:** [Prompt Engineering Guide](https://www.promptingguide.ai/) ·
[The Prompt Report](https://arxiv.org/abs/2406.06608) ·
[Anthropic prompt engineering](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
