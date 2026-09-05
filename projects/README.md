# Projects

Concepts teach you why. Skills teach you how. Projects are where you find out whether you actually
know it.

Every project below states **what to build** and **how to test it**. The test matters more than the
build — a demo that worked once is not evidence of anything, and building the habit of proving your
own work is most of what separates a professional from a hobbyist.

---

## How to work through these

1. **Do not start from a tutorial.** Read the brief, then build. Look things up as you get stuck.
2. **Write the test criteria down before you start.** In your README, as checkboxes.
3. **Break it deliberately.** Empty input, enormous input, wrong language, adversarial input, no
   network. A project that only works on the happy path is half-finished.
4. **Publish it.** A public repo with a README stating what you measured is worth more to an
   employer than a certificate.

---

## Tier 1 · Foundations
*After [`01-foundations`](../01-foundations/) modules 01–04 and [skills 01–03](../skills/).*

| Project | Build | Test |
|---|---|---|
| **Attention visualiser** | Heatmaps of real attention weights from a pretrained model | Find a head where "it" attends to its referent; change the sentence and watch the target move |
| **Terminal chatbot** | Streaming chat with memory, `/reset`, `/save`, token counter | Remembers turn 1 at turn 6; forgets after reset; saved transcript reloads |
| **Token cost calculator** | File in, token count and cost estimate out | Your count matches the provider's `usage` field exactly |
| **Prompt technique benchmark** | One task, six techniques, one results table | Objectively checkable answers; anyone can rerun and reproduce your numbers |
| **Text utility CLI** | Summarise / translate / extract / fix-grammar in one command | Survives an empty file, a 100k-word file, and a file of pure emoji |

## Tier 2 · Applied
*After [skills 04–06](../skills/).*

| Project | Build | Test |
|---|---|---|
| **Semantic search over your notes** | Embed a markdown folder, answer queries with scored passages | recall@5 on 10 known-answer questions; low scores for uncovered topics |
| **Chat with your PDFs** | Full RAG: chunk, embed, retrieve, cite | Returns NOT_FOUND for every unanswerable question; every citation resolves |
| **Structured data extractor** | Invoices or resumes to validated JSON | 50 documents, zero unhandled exceptions, all schema-valid |
| **Duplicate ticket detector** | Cluster near-duplicate support tickets | Plant known duplicates; report precision and recall at your threshold |
| **Meeting notes to actions** | Transcript in, structured action items with owners and dates out | Ambiguous ownership is flagged, not guessed |

## Tier 3 · Systems
*After [skills 07–09](../skills/).*

| Project | Build | Test |
|---|---|---|
| **Research agent** | Multi-step agent with search, calculator and file tools | 20 tasks × 5 runs; report success rate, mean steps, cost, top failure modes |
| **Code review bot** | Reviews a diff and posts findings | Plant known bugs; measure detection rate and false positives |
| **RAG evaluation suite** | Retrieval, faithfulness, correctness and refusal metrics | Deliberately degrade retrieval; confirm the right metric drops |
| **Multi-provider gateway** | One API over several providers, with fallback and cost tracking | Simulate an outage; confirm failover; produce a real cost comparison |
| **Prompt regression CI** | Version-controlled prompts, evaluated on every change | Degrade a prompt on purpose; the build must fail |

## Tier 4 · Deep
*After [`02-training`](../02-training/), [`03-inference`](../03-inference/) and [skill 10](../skills/10-fine-tuning/).*

| Project | Build | Test |
|---|---|---|
| **Tiny GPT from scratch** | Decoder-only model trained on one book | Validation loss below ~1.5 bits/char; compare samples across temperatures |
| **Modern architecture rebuild** | RMSNorm, RoPE, SwiGLU, GQA | Beats a 2017-style model of identical size and budget; ablate each change |
| **LoRA fine-tune** | Adapt a 7B model to a narrow task on a free GPU | Beats your prompted baseline by a stated margin; check for forgetting |
| **Scaling law replication** | 5 model sizes, fixed compute, log-log loss plot | You should recover a straight line; compare the slope to Chinchilla |
| **FlashAttention implementation** | Tiled, IO-aware attention kernel | Matches naive attention to 1e-4; memory goes quadratic to linear |

---

## Portfolio advice

Three finished, tested, documented projects beat ten half-built ones. For each, your README should
answer:

- **What problem does this solve?** One paragraph, no jargon.
- **How well does it work?** Numbers. Accuracy, latency, cost.
- **How do you know?** Your test set and method.
- **What are its limits?** Where it fails, honestly stated. This section impresses people more than
  any other.

---

## Adding a project

Add a row to the right tier with a real, objective test criterion. If you cannot write down how
someone would verify it works, the brief is not finished.
