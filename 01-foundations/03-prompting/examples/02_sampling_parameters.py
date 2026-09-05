"""
Sampling parameters, one at a time.

Reading a table of parameters teaches you very little. Watching the same prompt
change behaviour as you move one dial teaches you a lot. Each demo below isolates
a single parameter.

Requires GROQ_API_KEY. See ../../02-llms/examples/01_first_call_groq.py for setup.
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def ask(client, prompt, **params):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **params,
    )
    return response.choices[0].message.content.strip(), response.choices[0].finish_reason


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_temperature(client):
    header("temperature -- how sharply the distribution is sampled")
    prompt = "Give one metaphor for how an LLM works. One sentence, no preamble."

    for temp in (0.0, 1.0, 1.8):
        print(f"\ntemperature={temp}")
        for _ in range(3):
            text, _ = ask(client, prompt, temperature=temp, max_tokens=60)
            print(f"   {text}")

    print(
        "\nLow temperature repeats itself: it keeps picking the top token.\n"
        "High temperature explores -- more original, and eventually incoherent.\n"
        "Pick by task: near 0 for facts and code, ~0.7 for prose, >1 to brainstorm."
    )


def demo_top_p(client):
    header("top_p -- nucleus sampling: how much of the distribution is eligible")
    prompt = "Name one unusual colour. Reply with the colour name only."

    for top_p in (0.1, 0.5, 1.0):
        print(f"\ntop_p={top_p}")
        results = [ask(client, prompt, temperature=1.0, top_p=top_p, max_tokens=15)[0]
                   for _ in range(5)]
        print(f"   {results}")

    print(
        "\ntop_p=0.1 considers only the few most likely tokens, so answers cluster.\n"
        "top_p=1.0 leaves the whole vocabulary eligible.\n"
        "Tune temperature OR top_p, not both -- they interact confusingly."
    )


def demo_max_tokens(client):
    header("max_tokens -- your hard cost and length ceiling")
    prompt = "Explain how a transformer works."

    for limit in (30, 150):
        text, reason = ask(client, prompt, temperature=0.5, max_tokens=limit)
        print(f"\nmax_tokens={limit}  ->  finish_reason='{reason}'")
        print(f"   {text}")

    print(
        "\nfinish_reason='length' means the answer was CUT OFF mid-thought, not finished.\n"
        "Check this field before showing output to a user or parsing it -- a truncated\n"
        "JSON response is invalid JSON, and that is a production bug waiting to happen."
    )


def demo_stop(client):
    header("stop -- end generation when a string appears")
    prompt = "List three programming languages, one per line, numbered 1. 2. 3."

    text, reason = ask(client, prompt, temperature=0.0, max_tokens=100)
    print(f"\nwithout stop  (finish_reason='{reason}'):\n{text}")

    text, reason = ask(client, prompt, temperature=0.0, max_tokens=100, stop=["3."])
    print(f"\nwith stop=['3.']  (finish_reason='{reason}'):\n{text}")

    print(
        "\nUseful when you generate delimited or structured output and want to stop\n"
        "the moment a section ends -- it saves tokens, latency and money."
    )


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see ../../02-llms/examples/01_first_call_groq.py")

    client = Groq(api_key=api_key)
    demo_temperature(client)
    demo_top_p(client)
    demo_max_tokens(client)
    demo_stop(client)

    header("REMEMBER")
    print(
        "temperature=0 is greedy sampling, NOT a determinism guarantee. Server-side\n"
        "batching and GPU floating-point non-associativity can still change results\n"
        "between identical calls. Test on properties (valid JSON, required keys,\n"
        "length bounds), never on exact output strings."
    )


if __name__ == "__main__":
    main()
