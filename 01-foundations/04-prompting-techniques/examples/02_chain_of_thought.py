"""
Chain-of-Thought: what happens when you give the model room to work.

Each problem below needs several dependent steps. Asked for the answer directly,
the model has a fixed, small amount of computation to produce it. Asked to reason
first, the reasoning tokens ARE the computation.

Requires GROQ_API_KEY. See ../../02-llms/examples/01_first_call_groq.py for setup.
"""

import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

PROBLEMS = [
    {
        "q": (
            "A library had 340 books. It donated 1/4 of them, then received a shipment "
            "of 85 books, then lost 12 in a flood. How many books does it have now?"
        ),
        "answer": 328,
    },
    {
        "q": (
            "A train leaves at 09:20 and travels 240 km at 80 km/h. It then stops for "
            "25 minutes before travelling a further 150 km at 75 km/h. What time does it arrive?"
        ),
        "answer": None,  # 14:45 -- checked by eye
    },
    {
        "q": (
            "Ravi is twice as old as Meera. In 8 years, the sum of their ages will be 61. "
            "How old is Ravi now?"
        ),
        "answer": 30,
    },
]

DIRECT = "{q}\n\nReply with only the final answer. No explanation."

COT = "{q}\n\nLet's think step by step, then give the final answer on its own line as 'ANSWER: <value>'."


def ask(client, prompt, max_tokens):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip(), response.usage.completion_tokens


def extract_number(text):
    """Pull the last number out of a response, which is where the answer usually is."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see ../../02-llms/examples/01_first_call_groq.py")

    client = Groq(api_key=api_key)
    total_direct_tokens = total_cot_tokens = 0

    for i, problem in enumerate(PROBLEMS, 1):
        print("=" * 72)
        print(f"PROBLEM {i}")
        print("=" * 72)
        print(problem["q"])

        direct, direct_tokens = ask(client, DIRECT.format(q=problem["q"]), 60)
        cot, cot_tokens = ask(client, COT.format(q=problem["q"]), 600)
        total_direct_tokens += direct_tokens
        total_cot_tokens += cot_tokens

        print(f"\n--- DIRECT ({direct_tokens} tokens) ---")
        print(direct)

        print(f"\n--- CHAIN-OF-THOUGHT ({cot_tokens} tokens) ---")
        print(cot)

        if problem["answer"] is not None:
            expected = float(problem["answer"])
            d_ok = extract_number(direct) == expected
            c_ok = extract_number(cot) == expected
            print(f"\nexpected {problem['answer']}   direct: {'OK' if d_ok else 'WRONG'}"
                  f"   cot: {'OK' if c_ok else 'WRONG'}")
        print()

    print("=" * 72)
    print("THE TRADE-OFF")
    print("=" * 72)
    print(f"direct : {total_direct_tokens} output tokens total")
    print(f"cot    : {total_cot_tokens} output tokens total"
          f"  ({total_cot_tokens / max(total_direct_tokens, 1):.1f}x more)")
    print(
        "\nCoT costs several times more tokens and latency. On multi-step problems it\n"
        "buys real accuracy. On a sentiment classifier it buys nothing at all.\n"
        "Measure before you adopt it.\n\n"
        "2026 caveat: models with built-in reasoning (Claude extended thinking, Gemini\n"
        "thinking models, the o-series) already do this internally. Adding 'think step\n"
        "by step' to those is redundant and sometimes counterproductive."
    )


if __name__ == "__main__":
    main()
