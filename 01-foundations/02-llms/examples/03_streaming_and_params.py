"""
Streaming, and what temperature actually does.

Two things every LLM developer needs to feel rather than read about:

  1. Streaming -- tokens arriving one at a time, which is why chat UIs feel fast
     even though the full answer takes just as long to finish.
  2. Temperature -- run the same prompt several times at different settings and
     watch determinism turn into creativity (and then into nonsense).

Requires GROQ_API_KEY. See 01_first_call_groq.py for setup.
"""

import os
import sys
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see 01_first_call_groq.py for setup.")
    return Groq(api_key=api_key)


def demo_streaming(client):
    print("=" * 70)
    print("STREAMING")
    print("=" * 70)
    print("Watch the text appear token by token:\n")

    start = time.time()
    first_token_at = None

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Explain what a transformer is, in about 60 words."}],
        stream=True,
    )

    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            if first_token_at is None:
                first_token_at = time.time() - start
            print(piece, end="", flush=True)

    total = time.time() - start
    print(f"\n\ntime to first token : {first_token_at:.2f}s")
    print(f"total time          : {total:.2f}s")
    print(
        "\nThe gap between those two numbers is why you stream. The user starts\n"
        "reading after the first number, not the second."
    )


def demo_temperature(client):
    print("\n" + "=" * 70)
    print("TEMPERATURE")
    print("=" * 70)

    prompt = "Invent a name for a coffee shop that is also a bookstore. Reply with the name only."

    for temp in (0.0, 0.7, 1.5):
        print(f"\ntemperature = {temp}")
        print("-" * 40)
        for run in range(3):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=20,
            )
            print(f"  run {run + 1}: {response.choices[0].message.content.strip()}")

    print(
        "\nAt 0.0 the model near-always takes the highest-probability token, so runs\n"
        "repeat. As temperature rises, lower-probability tokens get a real chance:\n"
        "more variety, and eventually more nonsense.\n"
        "\n  Rule of thumb:\n"
        "    0.0 - 0.3   extraction, classification, code, anything factual\n"
        "    0.5 - 0.8   general conversation, explanation\n"
        "    1.0 - 1.5   brainstorming, fiction, deliberate variety\n"
        "\nNote: temperature=0 is 'greedy', not a guarantee of identical output --\n"
        "batching and floating-point non-determinism on the server can still vary."
    )


def main():
    client = get_client()
    demo_streaming(client)
    demo_temperature(client)


if __name__ == "__main__":
    main()
