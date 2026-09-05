"""
Structured output: getting JSON you can actually parse, and surviving when you cannot.

Three levels, in increasing order of how much you should trust them:

  1. Ask nicely            -- works most of the time, fails silently and badly
  2. Ask + validate + retry -- works in production
  3. Provider JSON mode     -- constrained decoding; use it when available

Requires GROQ_API_KEY. See ../../02-llms/examples/01_first_call_groq.py for setup.
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

REVIEW = (
    "Bought this laptop for my daughter's degree. Screen is gorgeous and the battery "
    "genuinely does last a full day. But the trackpad rattles, and support took eleven "
    "days to reply about it. At 78,000 rupees I expected better build quality."
)

SCHEMA_DESCRIPTION = """Return ONLY a JSON object, with no markdown fences and no commentary:
{
  "sentiment": "positive" | "negative" | "mixed",
  "score": <float between -1 and 1>,
  "aspects": [{"name": <string>, "opinion": "positive" | "negative"}],
  "price_mentioned": <number or null>
}"""

REQUIRED_KEYS = {"sentiment", "score", "aspects", "price_mentioned"}
VALID_SENTIMENTS = {"positive", "negative", "mixed"}


# ---------------------------------------------------------------------------
# Validation -- the part people skip and then debug at 2am
# ---------------------------------------------------------------------------

def strip_fences(text):
    """Models love wrapping JSON in ```json fences even when told not to."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


def validate(raw):
    """Return (parsed, None) on success or (None, error_message) on failure."""
    try:
        data = json.loads(strip_fences(raw))
    except json.JSONDecodeError as exc:
        return None, f"Not valid JSON: {exc}"

    if not isinstance(data, dict):
        return None, "Top level must be a JSON object."

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return None, f"Missing required keys: {sorted(missing)}"

    if data["sentiment"] not in VALID_SENTIMENTS:
        return None, f"sentiment must be one of {sorted(VALID_SENTIMENTS)}, got {data['sentiment']!r}"

    if not isinstance(data["score"], (int, float)) or not -1 <= data["score"] <= 1:
        return None, f"score must be a number between -1 and 1, got {data['score']!r}"

    if not isinstance(data["aspects"], list):
        return None, "aspects must be a list."

    for aspect in data["aspects"]:
        if not isinstance(aspect, dict) or {"name", "opinion"} - aspect.keys():
            return None, f"each aspect needs 'name' and 'opinion' keys, got {aspect!r}"

    return data, None


# ---------------------------------------------------------------------------
# The three levels
# ---------------------------------------------------------------------------

def level_1_ask_nicely(client):
    print("=" * 72)
    print("LEVEL 1 -- just ask")
    print("=" * 72)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{SCHEMA_DESCRIPTION}\n\nReview: {REVIEW}"}],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content
    print(raw)

    parsed, error = validate(raw)
    print(f"\nvalidation: {'PASSED' if parsed else 'FAILED -- ' + error}")
    print("\nOften fine. But 'often' is not a guarantee, and the failure mode is a\n"
          "JSONDecodeError in production at 2am.")


def level_2_validate_and_retry(client, max_attempts=3):
    print("\n" + "=" * 72)
    print("LEVEL 2 -- validate, and retry with the error message")
    print("=" * 72)

    messages = [{"role": "user", "content": f"{SCHEMA_DESCRIPTION}\n\nReview: {REVIEW}"}]

    for attempt in range(1, max_attempts + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.0, max_tokens=400
        )
        raw = response.choices[0].message.content

        parsed, error = validate(raw)
        if parsed:
            print(f"attempt {attempt}: valid")
            print(json.dumps(parsed, indent=2))
            return parsed

        print(f"attempt {attempt}: invalid -- {error}")
        # Feed the concrete error back. Models are good at fixing a specific complaint.
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"That failed validation: {error}\nReturn corrected JSON only.",
        })

    print(f"failed after {max_attempts} attempts -- fall back to a default, do not crash")
    return None


def level_3_json_mode(client):
    print("\n" + "=" * 72)
    print("LEVEL 3 -- provider JSON mode (constrained decoding)")
    print("=" * 72)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                # Most providers require the word "JSON" to appear in the prompt
                # when JSON mode is enabled.
                "content": f"Extract JSON from this review.\n{SCHEMA_DESCRIPTION}\n\nReview: {REVIEW}",
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content
        print(raw)

        parsed, error = validate(raw)
        print(f"\nvalidation: {'PASSED' if parsed else 'FAILED -- ' + error}")
        print(
            "\nJSON mode constrains DECODING, so syntactically invalid JSON becomes\n"
            "impossible. It does NOT guarantee your schema -- keys can still be missing\n"
            "and values can still be nonsense. Keep the validator from level 2."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"JSON mode unavailable for this model: {exc}")
        print("Not every model supports it. Levels 1 and 2 always work.")


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see ../../02-llms/examples/01_first_call_groq.py")

    client = Groq(api_key=api_key)
    level_1_ask_nicely(client)
    level_2_validate_and_retry(client)
    level_3_json_mode(client)

    print("\n" + "=" * 72)
    print("THE RULE")
    print("=" * 72)
    print(
        "Never call json.loads() on a model response without a try/except and a\n"
        "schema check. Use the provider's native JSON/tool-calling support where it\n"
        "exists, validate the result anyway, and always have a fallback path.\n\n"
        "For real projects use pydantic or jsonschema rather than the hand-rolled\n"
        "validate() above -- it is written out longhand here so you can see exactly\n"
        "what a validator has to check."
    )


if __name__ == "__main__":
    main()
