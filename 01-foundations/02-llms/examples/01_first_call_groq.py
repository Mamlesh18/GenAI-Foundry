"""
Your first LLM API call, using Groq.

Setup
-----
1. Get a free API key at https://console.groq.com/keys
2. Set it in your environment:

       macOS/Linux:  export GROQ_API_KEY="gsk_..."
       PowerShell:   $env:GROQ_API_KEY="gsk_..."

   ...or put GROQ_API_KEY=gsk_... in a .env file next to this script.
3. pip install -r requirements.txt
4. python 01_first_call_groq.py
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Model IDs change over time. If this one 404s, pick a current one from
# https://console.groq.com/docs/models
MODEL = "llama-3.3-70b-versatile"


def get_client():
    """Build a client, failing with a useful message rather than a stack trace."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "GROQ_API_KEY is not set.\n"
            "  1. Get a free key at https://console.groq.com/keys\n"
            "  2. export GROQ_API_KEY='gsk_...'   (PowerShell: $env:GROQ_API_KEY='gsk_...')"
        )
    return Groq(api_key=api_key)


def main():
    client = get_client()

    # -- The minimal call --------------------------------------------------
    # `messages` is the entire conversation. The API is stateless: it knows
    # nothing except what you send in this list, every single time.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise teacher. Answer in at most three sentences.",
            },
            {
                "role": "user",
                "content": "What is a large language model, and what is it actually predicting?",
            },
        ],
        temperature=0.7,   # randomness: 0 = deterministic-ish, higher = more varied
        max_tokens=300,    # hard cap on the response length
    )

    print("=" * 70)
    print("RESPONSE")
    print("=" * 70)
    print(response.choices[0].message.content)

    # -- Everything else the response carries ------------------------------
    # Reading this metadata is a habit worth forming now: it is how you track
    # cost, catch truncated answers, and debug latency in production.
    usage = response.usage
    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    print(f"model          : {response.model}")
    print(f"finish_reason  : {response.choices[0].finish_reason}")
    print(f"prompt tokens  : {usage.prompt_tokens}")
    print(f"output tokens  : {usage.completion_tokens}")
    print(f"total tokens   : {usage.total_tokens}")

    print(
        "\nfinish_reason 'stop' means the model finished naturally.\n"
        "'length' means it hit max_tokens and was cut off mid-thought -- always\n"
        "check this before showing a response to a user."
    )


if __name__ == "__main__":
    main()
