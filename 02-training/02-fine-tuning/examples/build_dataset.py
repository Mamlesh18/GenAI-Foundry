"""
Build a correctly-formatted SFT dataset for LLaMA-Factory -- including the
dataset_info.json registration step that everybody forgets.

Writes:
    my_task.json         the training data, Alpaca format
    my_task_sharegpt.json the same idea in multi-turn ShareGPT format
    dataset_info_entry.json  what to paste into LLaMA-Factory/data/dataset_info.json

    python build_dataset.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Your raw data
# ---------------------------------------------------------------------------
# In reality this comes from your support tickets, code reviews, past work --
# whatever record you already have of the task being done well. Existing logs
# are the best source most teams have and the one they most often overlook.

RAW = [
    ("The battery dies in two hours.", "negative"),
    ("Arrived on time, does exactly what it says.", "neutral"),
    ("I have bought three more since.", "positive"),
    ("Screen is gorgeous but the trackpad rattles.", "mixed"),
    ("Support took eleven days to reply.", "negative"),
    ("Does the job. Nothing remarkable either way.", "neutral"),
    ("Genuinely the best purchase I made this year.", "positive"),
    ("Great value, though the manual is useless.", "mixed"),
]

INSTRUCTION = (
    "Classify the sentiment of the product review as exactly one of: "
    "positive, negative, neutral, mixed. Reply with the label only."
)


# ---------------------------------------------------------------------------
# 2. Alpaca format -- single turn, the simplest thing that works
# ---------------------------------------------------------------------------

def build_alpaca(rows):
    return [
        {
            "instruction": INSTRUCTION,   # what to do
            "input": review,              # what to do it to
            "output": label,              # the response -- the ONLY part loss is computed on
        }
        for review, label in rows
    ]


# ---------------------------------------------------------------------------
# 3. ShareGPT format -- multi-turn conversations
# ---------------------------------------------------------------------------

def build_sharegpt(rows):
    """Pair rows up into two-turn conversations, to show the multi-turn shape."""
    convos = []
    for i in range(0, len(rows) - 1, 2):
        (r1, l1), (r2, l2) = rows[i], rows[i + 1]
        convos.append({
            "conversations": [
                {"from": "human", "value": f"{INSTRUCTION}\n\nReview: {r1}"},
                {"from": "gpt", "value": l1},
                {"from": "human", "value": f"And this one?\n\nReview: {r2}"},
                {"from": "gpt", "value": l2},
            ],
            "system": "You are a precise sentiment classifier.",
        })
    return convos


# ---------------------------------------------------------------------------
# 4. Validation -- catch the errors that silently ruin a fine-tune
# ---------------------------------------------------------------------------

VALID_LABELS = {"positive", "negative", "neutral", "mixed"}


def validate(examples):
    problems = []
    seen = set()

    for i, ex in enumerate(examples):
        if not ex["output"].strip():
            problems.append(f"[{i}] empty output")
        if ex["output"] not in VALID_LABELS:
            problems.append(f"[{i}] output {ex['output']!r} is not one of {sorted(VALID_LABELS)}")
        if ex["input"] in seen:
            problems.append(f"[{i}] duplicate input: {ex['input'][:40]!r}")
        seen.add(ex["input"])

    # Class balance. An imbalanced set produces a model biased toward the
    # over-represented label, and it will look fine on a matching test set.
    counts = {}
    for ex in examples:
        counts[ex["output"]] = counts.get(ex["output"], 0) + 1

    return problems, counts


def main():
    random.seed(0)
    rows = RAW[:]
    random.shuffle(rows)

    # Always hold data out, and never train on it.
    split = int(0.8 * len(rows))
    train_rows, eval_rows = rows[:split], rows[split:]

    train = build_alpaca(train_rows)
    evalset = build_alpaca(eval_rows)
    sharegpt = build_sharegpt(RAW)

    (OUT / "my_task.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (OUT / "my_task_eval.json").write_text(json.dumps(evalset, indent=2), encoding="utf-8")
    (OUT / "my_task_sharegpt.json").write_text(json.dumps(sharegpt, indent=2), encoding="utf-8")

    # The registration entry. LLaMA-Factory will not find your file without this.
    entry = {
        "my_task": {
            "file_name": "my_task.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "my_task_sharegpt": {
            "file_name": "my_task_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt",
            },
        },
    }
    (OUT / "dataset_info_entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DATASET BUILT")
    print("=" * 72)
    print(f"  train : {len(train)} examples  -> my_task.json")
    print(f"  eval  : {len(evalset)} examples  -> my_task_eval.json")
    print(f"  multi : {len(sharegpt)} conversations -> my_task_sharegpt.json")

    problems, counts = validate(train + evalset)

    print("\n" + "=" * 72)
    print("VALIDATION")
    print("=" * 72)
    if problems:
        print("  problems found:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  no structural problems found")

    print("\n  class balance:")
    for label, n in sorted(counts.items()):
        print(f"    {label:>8}: {'#' * n} ({n})")
    print(
        "\n  Check this every time. An imbalanced set produces a model biased toward\n"
        "  the over-represented label -- and it will still score well on a test set\n"
        "  that shares the same imbalance, which is how the bug survives to production."
    )

    print("\n" + "=" * 72)
    print("SAMPLE EXAMPLE (Alpaca)")
    print("=" * 72)
    print(json.dumps(train[0], indent=2))

    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    print(
        "  1. Copy my_task.json into LLaMA-Factory/data/\n"
        "  2. Merge dataset_info_entry.json into LLaMA-Factory/data/dataset_info.json\n"
        "     (it is one big JSON object -- add these keys to it, do not replace the file)\n"
        "  3. Set `dataset: my_task` in your training YAML\n"
        "  4. llamafactory-cli train llamafactory_sft.yaml\n\n"
        "  Before you train: READ 50 EXAMPLES BY HAND. You will find errors.\n"
        "  Everyone does, and no hyperparameter will rescue a bad dataset."
    )


if __name__ == "__main__":
    main()
