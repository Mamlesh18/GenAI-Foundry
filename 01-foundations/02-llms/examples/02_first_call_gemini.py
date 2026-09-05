"""
The same first call, using Google Gemini.

Run this right after 01_first_call_groq.py and compare them. Almost everything
that differs is spelling, not substance -- that is the point of doing both.

Setup
-----
1. Get a free API key at https://aistudio.google.com/apikey
2. export GEMINI_API_KEY="..."     (PowerShell: $env:GEMINI_API_KEY="...")
3. pip install -r requirements.txt
4. python 02_first_call_gemini.py
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Current model list: https://ai.google.dev/gemini-api/docs/models
MODEL = "gemini-2.5-flash"


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY is not set.\n"
            "  1. Get a free key at https://aistudio.google.com/apikey\n"
            "  2. export GEMINI_API_KEY='...'   (PowerShell: $env:GEMINI_API_KEY='...')"
        )
    return genai.Client(api_key=api_key)


def main():
    client = get_client()

    response = client.models.generate_content(
        model=MODEL,
        contents="What is a large language model, and what is it actually predicting?",
        config=types.GenerateContentConfig(
            # Gemini calls it system_instruction; Groq/OpenAI call it a system message.
            # Same concept: standing instructions that apply to the whole conversation.
            system_instruction="You are a concise teacher. Answer in at most three sentences.",
            temperature=0.7,
            max_output_tokens=300,
        ),
    )

    print("=" * 70)
    print("RESPONSE")
    print("=" * 70)
    print(response.text)

    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    usage = response.usage_metadata
    print(f"model         : {MODEL}")
    print(f"prompt tokens : {usage.prompt_token_count}")
    print(f"output tokens : {usage.candidates_token_count}")
    print(f"total tokens  : {usage.total_token_count}")

    print(
        "\nCompare with 01_first_call_groq.py:\n"
        "  messages=[...]        vs  contents=...\n"
        "  system message        vs  system_instruction\n"
        "  max_tokens            vs  max_output_tokens\n"
        "  .choices[0].message   vs  .text\n"
        "Different spelling, identical idea. Learn the idea, look up the spelling."
    )


if __name__ == "__main__":
    main()
