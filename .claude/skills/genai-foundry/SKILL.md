---
name: genai-foundry
description: Add or update learning content in the GenAI-Foundry repo — a track, a module, a README or a runnable example. Enforces this repo's conventions (every number measured from a real run, every link verified, CPU-runnable ASCII-only examples), runs the verification pass, then COMMITS AND PUSHES after each completed module. Use whenever the user asks to build out, fill in, extend or correct any folder in this repo.
---

# GenAI-Foundry content workflow

This repo is an ordered learning resource for generative AI: concept tracks, hands-on skills, and
project briefs. Content is written to be **read, run, and broken on purpose** by students and
professionals.

**The rule that makes this repo different from a blog: nothing goes in a README unless it was
measured or verified. Then it gets committed and pushed immediately.**

---

## 1. Repo layout

```
01-foundations/ … 08-multimodal/   concept tracks, each with numbered modules
  <NN-module>/README.md            the teaching document
  <NN-module>/examples/*.py        runnable code the README quotes numbers from
skills/                            hands-on capability guides (pass/fail checkpoints)
projects/                          project briefs with test criteria
resources/                         curated links
```

A **track** has a `README.md` (the map) plus numbered module folders. A **module** has a `README.md`
and usually an `examples/` folder.

---

## 2. The unit of work

**One "thing" = one module** (its README plus its examples), **or** one track-level README.

Finish a thing completely — research, code, run, write, verify — then commit and push it before
starting the next. Never leave a half-finished module uncommitted while moving on.

---

## 3. The loop, per thing

```
1. RESEARCH   verify every link and fact you intend to cite (WebSearch / WebFetch)
2. BUILD      write the example script(s) first
3. RUN        execute them; capture the real output
4. WRITE      write the README, quoting the numbers you just measured
5. VERIFY     links resolve, scripts compile, output is ASCII
6. COMMIT     one commit for the module
7. PUSH       immediately
```

### Step 1 — Research

- **Verify before citing.** Fetch every URL you plan to link. If a page 404s, redirects somewhere
  unexpected, or blocks fetching, either drop it or say in the reply that it is unverified.
- Prefer primary sources: official docs, the paper's arXiv abstract page, the project's repo.
- Check arXiv IDs by fetching `arxiv.org/abs/<id>` and confirming the exact title.
- For tools and flags (vLLM, SGLang, LLaMA-Factory…), read the current docs. **Never write a CLI
  flag from memory** — defaults and names move between releases.
- If a doc contradicts what you believed, the doc wins; say so in the reply.

### Step 2 — Build the example

Conventions, all non-negotiable:

| Rule | Why |
|---|---|
| **No API key required to run** | A learner must be able to run it immediately. Mock the model if needed |
| **Stdlib or numpy preferred** | `torch`, `sentence-transformers`, `transformers` are available here; use them only when they earn their place |
| **ASCII-only printed output** | The Windows console is cp1252; a `≈` or `→` in a `print()` crashes it. Use `--`, `->`, `<=` |
| **`127.0.0.1`, never `localhost`** | `localhost` resolves to IPv6 first on this machine and adds ~2 s per request, which silently ruins latency measurements |
| **Deterministic** | Seed every RNG so the README's numbers reproduce |
| **Fast** | Target under ~90 s on a CPU. Shrink the model or the step count, not the lesson |
| **Self-contained** | One file per idea; some duplication between modules is fine and preferable to a shared utils import |
| **Teach in the comments** | Explain *why*, not what the line does |
| **Graceful degradation** | If an optional dependency is missing, print how to install it and skip that section |

Each script should print numbered sections and end with a "what to take away" block.

### Step 3 — Run it

Use the interpreter that has the ML libraries:

```bash
"/c/Program Files/Python312/python.exe" -u <script.py>
```

The bare `python` on PATH may point at a venv without `torch`. Always pass `-u`.

**Read the output properly.** This is where most defects surface. Ask of every section:

- Does the number actually demonstrate the claim, or did it come out flat/backwards?
- Is a "dramatic" result an artefact of the setup rather than the phenomenon?
- Did a demo silently take the wrong branch (a refusal path that never fires, a threshold never hit)?

### Step 4 — Write the README

Follow the shape used across the repo:

1. **Title + one-sentence definition** in a `>` blockquote
2. **The problem** — plain language, an analogy where it helps
3. **How it works** — an ASCII diagram for anything architectural
4. **The numbers** — tables quoting your measured output, labelled as measured
5. **Practical guidance** — defaults, flags, when to use and when not to
6. **Honest limits** — where the technique or the demo fails
7. **Examples / Exercises / Projects** — projects state *how to test it*
8. **Resources** — grouped "Start here" (beginner) / "Go deeper" / "Papers"
9. **Navigation footer** — previous / next links

Style: British spelling, plain words, no hype. Prefer "this fails when…" over "this is powerful".
Beginners are the default audience; say what a term means the first time it appears.

### Step 5 — Verify

```bash
cd <repo root>
"/c/Program Files/Python312/python.exe" - <<'PYEOF'
import os, re, glob, py_compile
from urllib.parse import unquote
bad, total = [], 0
for md in glob.glob('**/*.md', recursive=True):
    if '.git' in md or '_TEMPLATE' in md: continue
    base = os.path.dirname(md)
    text = re.sub(r'`[^`\n]*`', '', open(md, encoding='utf-8').read())   # ignore code spans
    for m in re.finditer(r'\[([^\]]*)\]\(([^)\s]+)\)', text):
        t = m.group(2)
        if t.startswith(('http://','https://','mailto:','#')): continue
        total += 1
        p = unquote(t.split('#')[0])
        if p and not os.path.exists(os.path.normpath(os.path.join(base, p))):
            bad.append(f"{md} -> {t}")
print(f"links: {total} relative checked, {len(bad)} broken", *bad, sep="\n  ")
pys = glob.glob('<track>/**/*.py', recursive=True)
for f in pys: py_compile.compile(f, doraise=True)
print("non-ASCII:", [f for f in pys if any(ord(c) > 127 for c in open(f, encoding='utf-8').read())] or "none")
PYEOF
find <track> -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

All three must pass: **0 broken links, everything compiles, no non-ASCII in `.py` files.**

### Step 6 — Commit

One commit per module. Stage only that module's files.

```bash
git add <module path>
git commit -m "$(cat <<'EOF'
feat(<track>): <module> — <what a reader gains>

- <example script>: <what it demonstrates, with the headline number>
- README: <sections added>
- verified: <N> links, scripts run on CPU in <time>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: <the session URL from this session's attribution reminder>
EOF
)"
```

Take the `Claude-Session` URL from the current session's attribution reminder — do not copy a stale
one from an old commit. If the harness gives no attribution guidance, include only the
`Co-Authored-By` line.

### Step 7 — Push

```bash
git push
```

Push after **every** commit, so nothing important exists only on this machine.

If the push is rejected because the branch has diverged: **stop and tell the user.** Do not force
push, and do not rebase over their commits without asking. Someone else commits to this repo.

---

## 4. Hard rules

1. **Every number in a README comes from a run you did.** No estimates presented as measurements.
   If a figure comes from a cited source, name the source next to it.
2. **When the measurement contradicts the narrative, change the narrative.** This has happened
   repeatedly and is the most valuable thing this repo does. A module that says "hybrid search does
   not always help, here is the data" teaches more than one that repeats the usual claim.
3. **Say what the demo cannot show.** Toy corpora and mock models have limits; state them where a
   reader might over-generalise.
4. **Every external link verified this session.** No exceptions for "well-known" URLs.
5. **Never invent a flag, metric name, model id or API shape.** Check the docs.
6. **Report honestly in the reply**: what you ran, what you could not run (no GPU, no API key), and
   what you fixed after running it.

---

## 5. Repo-specific gotchas

| Gotcha | Handling |
|---|---|
| Bash heredocs mangle `\n` inside Python strings | For multi-edit patches, write a patch script to the scratchpad with the Write tool (raw strings, assert each match is unique), then run it |
| `python` on PATH lacks torch | Use `"/c/Program Files/Python312/python.exe"` |
| Windows console is cp1252 | ASCII-only in anything a script prints |
| `localhost` costs ~2 s per request | Use `127.0.0.1` in examples and in tests |
| No `GROQ_API_KEY` / `GEMINI_API_KEY` here | API examples must fail with a helpful message; test that path, and say in the reply that live calls are unverified |
| No GPU | Training/serving commands are doc-verified, not run. Say so |
| `sentence-transformers` + `all-MiniLM-L6-v2` work | Real embeddings are available — prefer them over faking semantic similarity |
| Another person commits to this repo | Check `git log` before assuming your last commit is HEAD |

---

## 6. Definition of done

Before the commit, confirm:

- [ ] Every script runs clean and its output matches what the README claims
- [ ] Every number in the README traceable to a run or a named source
- [ ] Every link verified (external) and resolving (internal)
- [ ] Scripts compile; printed output is ASCII; RNGs seeded
- [ ] README has: definition, diagram, measured numbers, honest limits, exercises, projects with
      test criteria, grouped resources, navigation footer
- [ ] `__pycache__` removed
- [ ] Committed with attribution, and **pushed**

Then report to the user: what was built, the headline measured results, what you fixed after
running it, and anything you could not verify.
