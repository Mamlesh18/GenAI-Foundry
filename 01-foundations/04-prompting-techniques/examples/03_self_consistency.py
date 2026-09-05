"""
Self-consistency: sample n reasoning chains, take the majority answer.

One chain of thought can take a wrong turn. Five independent chains rarely take
the SAME wrong turn, so the correct answer usually wins the vote. This is an
ensemble method dressed as a prompting technique.

Requires GROQ_API_KEY. See ../../02-llms/examples/01_first_call_groq.py for setup.
"""

import os
import re
import sys
from collections import Counter

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
N_SAMPLES = 5

PROBLEM = (
    "A baker makes 5 trays of 12 muffins. He sells 3/4 of all the muffins, "
    "gives 6 away, and drops one tray's worth on the floor. How many muffins "
    "does he have left?"
)

PROMPT = (
    f"{PROBLEM}\n\n"
    "Reason step by step, then give the final answer on its own line as 'ANSWER: <number>'."
)


def extract_answer(text):
    """Prefer the explicit ANSWER: line; fall back to the last number in the text."""
    match = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see ../../02-llms/examples/01_first_call_groq.py")

    client = Groq(api_key=api_key)

    print("=" * 72)
    print("PROBLEM")
    print("=" * 72)
    print(PROBLEM)
    print(f"\nSampling {N_SAMPLES} independent reasoning chains at temperature 0.8...\n")

    answers, total_tokens = [], 0

    for i in range(1, N_SAMPLES + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            # Temperature must be > 0, or every chain is identical and voting is pointless.
            temperature=0.8,
            max_tokens=600,
        )
        text = response.choices[0].message.content
        total_tokens += response.usage.completion_tokens

        answer = extract_answer(text)
        answers.append(answer)

        # Show the last reasoning line so you can see where chains diverge.
        tail = [line for line in text.strip().split("\n") if line.strip()][-1]
        print(f"  chain {i}: answer = {answer}   ...{tail[-60:]}")

    print("\n" + "=" * 72)
    print("VOTE")
    print("=" * 72)

    votes = Counter(a for a in answers if a is not None)
    if not votes:
        print("No parseable answers. Check your extraction logic.")
        return

    for answer, count in votes.most_common():
        bar = "#" * count
        print(f"  {answer:>10}  {bar} ({count}/{len(answers)})")

    winner, count = votes.most_common(1)[0]
    agreement = count / len(answers)

    print(f"\nmajority answer : {winner}")
    print(f"agreement       : {agreement:.0%}")
    print(f"cost            : {total_tokens} output tokens ({N_SAMPLES}x a single chain)")

    print("\n" + "=" * 72)
    print("WHAT TO NOTICE")
    print("=" * 72)
    print(
        "If all chains agree, one call would have been enough -- you paid 5x for\n"
        "nothing. If they split, self-consistency just saved you from a wrong answer\n"
        "you would otherwise have shipped.\n\n"
        "The agreement rate is itself useful: treat low agreement as a confidence\n"
        "signal and escalate those cases to a human or a stronger model.\n\n"
        "Only works for tasks with a discrete, checkable answer. You cannot majority-vote\n"
        "on an essay."
    )


if __name__ == "__main__":
    main()
