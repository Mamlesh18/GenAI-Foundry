"""
The anatomy of a prompt, demonstrated by building one up piece by piece.

The same underlying task runs four times. Each version adds components from the
six-part anatomy in the README. Read the four outputs side by side -- the
sharpening is the entire lesson.

Requires GROQ_API_KEY. See ../../02-llms/examples/01_first_call_groq.py for setup.
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

CODE_UNDER_REVIEW = '''
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates
'''


# Four versions of the same request, each adding anatomy components.
PROMPTS = [
    {
        "label": "v1  --  task only",
        "has": "task",
        "system": None,
        "user": f"Review this code.\n{CODE_UNDER_REVIEW}",
    },
    {
        "label": "v2  --  + role, + specific task",
        "has": "role, task",
        "system": "You are a senior Python reviewer.",
        "user": f"Review this function for correctness bugs.\n{CODE_UNDER_REVIEW}",
    },
    {
        "label": "v3  --  + context, + constraints",
        "has": "role, task, context, constraints",
        "system": "You are a senior Python reviewer.",
        "user": (
            "Review the function below for correctness bugs only -- ignore style.\n"
            "Context: this runs in a hot loop over lists of up to 100,000 items, so "
            "algorithmic complexity matters more than readability.\n"
            "Report at most 3 issues, most severe first.\n"
            f"{CODE_UNDER_REVIEW}"
        ),
    },
    {
        "label": "v4  --  + output format, + escape hatch",
        "has": "role, task, context, format, constraints, escape hatch",
        "system": (
            "You are a senior Python reviewer. You reply with JSON and nothing else -- "
            "no markdown fences, no commentary."
        ),
        "user": (
            "Review the function below for correctness and performance bugs only -- ignore style.\n"
            "Context: this runs in a hot loop over lists of up to 100,000 items.\n\n"
            "Return a JSON array of at most 3 objects, most severe first, each with keys:\n"
            '  "issue"    -- one sentence\n'
            '  "severity" -- one of "low", "medium", "high"\n'
            '  "line"     -- the 1-indexed line number within the snippet\n'
            "If there are no real bugs, return [].\n\n"
            f"<code>{CODE_UNDER_REVIEW}</code>"
        ),
    },
]


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see ../../02-llms/examples/01_first_call_groq.py")

    client = Groq(api_key=api_key)

    for prompt in PROMPTS:
        messages = []
        if prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,   # low: we want the review, not creativity
            max_tokens=500,
        )
        answer = response.choices[0].message.content

        print("=" * 72)
        print(prompt["label"])
        print(f"components: {prompt['has']}")
        print("=" * 72)
        print(answer.strip())
        print(f"\n[{response.usage.completion_tokens} output tokens]\n")

    print("=" * 72)
    print("WHAT TO NOTICE")
    print("=" * 72)
    print(
        "v1 wanders -- it comments on naming, docstrings, whatever it feels like.\n"
        "v2 narrows the scope but the shape of the answer is still unpredictable.\n"
        "v3 finds the real problem (O(n^2), plus duplicates reported more than once).\n"
        "v4 returns something a program can consume without parsing prose.\n\n"
        "Only v4 is usable in software. The model did not get smarter between v1 and\n"
        "v4 -- you removed the ways it could reasonably have misread you."
    )


if __name__ == "__main__":
    main()
