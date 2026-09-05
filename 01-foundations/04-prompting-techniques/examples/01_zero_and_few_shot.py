"""
Zero-shot vs few-shot, on a task where format consistency is the whole point.

The task: extract structured fields from messy support tickets. Zero-shot, the
model will do something reasonable but different every time. Few-shot, it copies
your format exactly.

Requires GROQ_API_KEY. See ../../02-llms/examples/01_first_call_groq.py for setup.
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

TICKETS = [
    "hi my order #4471 never showed up its been 2 weeks, really annoyed",
    "Can you tell me if the blue one comes in a large? Thanks!",
    "APP CRASHES EVERY TIME I OPEN SETTINGS. iphone 14. fix this",
    "just wanted to say the support last week was excellent, thank you",
]

ZERO_SHOT = """Extract the category, urgency and any order number from this support ticket.

Ticket: {ticket}"""

FEW_SHOT = """Extract fields from support tickets.

Ticket: "my package arrived smashed, order 1102, need a refund now"
category: shipping | urgency: high | order_id: 1102

Ticket: "do you ship to Ireland?"
category: question | urgency: low | order_id: none

Ticket: "the login button does nothing on firefox"
category: bug | urgency: medium | order_id: none

Ticket: "thanks for sorting my exchange so quickly!"
category: praise | urgency: low | order_id: none

Ticket: "{ticket}"
"""


def run(client, template, ticket):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": template.format(ticket=ticket)}],
        temperature=0.0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see ../../02-llms/examples/01_first_call_groq.py")

    client = Groq(api_key=api_key)

    print("=" * 72)
    print("ZERO-SHOT  --  no examples, model decides the format")
    print("=" * 72)
    for ticket in TICKETS:
        print(f"\nticket : {ticket[:60]}")
        print(f"output : {run(client, ZERO_SHOT, ticket)}")

    print("\n\n" + "=" * 72)
    print("FEW-SHOT  --  four examples, model copies the format")
    print("=" * 72)
    for ticket in TICKETS:
        print(f"\nticket : {ticket[:60]}")
        print(f"output : {run(client, FEW_SHOT, ticket)}")

    print("\n\n" + "=" * 72)
    print("WHAT TO NOTICE")
    print("=" * 72)
    print(
        "Zero-shot answers are all correct-ish and all shaped differently -- some\n"
        "prose, some bullets, different category words each time. You cannot parse that.\n\n"
        "Few-shot answers are one line, same separator, same field order, and the\n"
        "category vocabulary is constrained to the four words you demonstrated. You\n"
        "never wrote down a single rule -- the examples carried all of it.\n\n"
        "That is in-context learning, and it is the highest-return trick in prompting.\n\n"
        "Try this: change ONE example above to use a different separator, rerun, and\n"
        "watch how often the model copies your inconsistency. Examples are contagious\n"
        "in both directions."
    )


if __name__ == "__main__":
    main()
