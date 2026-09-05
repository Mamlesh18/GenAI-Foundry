"""
A working terminal chatbot -- and a demonstration of where "memory" comes from.

The key insight: the API is stateless. It remembers nothing. A conversation is
just a list that you keep appending to and resending in full, every turn.

Run it, then type /history to see the entire thing the model is actually reading.

Requires GROQ_API_KEY. See 01_first_call_groq.py for setup.

Commands:
    /history   show the raw message list being sent to the model
    /reset     clear the conversation
    /tokens    show cumulative token usage
    /quit      exit
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = (
    "You are a friendly tutor for someone learning about generative AI. "
    "Keep answers short and concrete. Use an example whenever it helps."
)


class Chat:
    def __init__(self, client, model=MODEL, system_prompt=SYSTEM_PROMPT):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0

    def send(self, user_text):
        # 1. Append what the user said.
        self.messages.append({"role": "user", "content": user_text})

        # 2. Send the ENTIRE history. This is the whole trick.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=500,
        )

        reply = response.choices[0].message.content

        # 3. Append the reply too -- otherwise the model will not know what it
        #    just said, and the next turn will have amnesia. (Try deleting this
        #    line: it is exercise 3 in the README.)
        self.messages.append({"role": "assistant", "content": reply})

        self.total_tokens += response.usage.total_tokens
        return reply

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def show_history(self):
        print("\n--- exactly what gets sent to the model next turn ---")
        for i, msg in enumerate(self.messages):
            content = msg["content"].replace("\n", " ")
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  [{i}] {msg['role']:>9}: {content}")
        print(f"--- {len(self.messages)} messages; it all resends every single turn ---\n")


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set -- see 01_first_call_groq.py for setup.")

    chat = Chat(Groq(api_key=api_key))

    print("=" * 70)
    print("CHAT  --  /history  /reset  /tokens  /quit")
    print("=" * 70)
    print("Tip: tell it your name, ask a few things, then ask 'what is my name?'")
    print("Then run /reset and ask again.\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("bye")
            break
        if user_input == "/history":
            chat.show_history()
            continue
        if user_input == "/reset":
            chat.reset()
            print("[conversation cleared -- it now knows nothing about you]\n")
            continue
        if user_input == "/tokens":
            print(f"[cumulative tokens used: {chat.total_tokens}]\n")
            continue

        try:
            print(f"\nbot > {chat.send(user_input)}\n")
        except Exception as exc:  # noqa: BLE001 - surface the real error while learning
            print(f"\n[error] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
