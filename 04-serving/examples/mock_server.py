"""
A fake LLM server that speaks the OpenAI API -- so you can learn serving
without a GPU.

Real engines (vLLM, SGLang, TGI, Ollama, and the cloud providers) all expose
the same HTTP endpoints. This file implements the three that matter, returning
canned text with realistic delays:

    GET  /health                  is the server alive?
    GET  /v1/models               which models are loaded?
    POST /v1/chat/completions     generate (streaming or not)

Run it in one terminal:

    python mock_server.py                 # listens on http://localhost:8000

...then point the client at it from another:

    python openai_client.py

Swap the URL for a real vLLM (port 8000) or SGLang (port 30000) server later
and the client does not change. That is the whole point of a standard API.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "mock/tiny-llm"

# Delays chosen to imitate a real server: a slow first token (that is PREFILL,
# reading your whole prompt) then a steady stream of output tokens (DECODE).
TIME_TO_FIRST_TOKEN = 0.40      # seconds
TIME_PER_OUTPUT_TOKEN = 0.03    # seconds

CANNED_REPLY = (
    "Serving means putting a model behind an API so many people can use it at "
    "once. The server batches their requests together, keeps a KV cache for "
    "each conversation, and streams tokens back as they are produced."
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):          # keep the console readable
        print(f"  [server] {fmt % args}")

    # -- helpers ----------------------------------------------------------
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path.rstrip("/") == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mock"}],
            })
        else:
            self._send_json({"error": {"message": f"unknown path {self.path}"}}, 404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": {"message": f"unknown path {self.path}"}}, 404)
            return

        request = self._read_json()
        words = CANNED_REPLY.split()
        max_tokens = int(request.get("max_tokens") or 0) or len(words)
        words = words[:max_tokens]

        if request.get("stream"):
            self._stream(words)
        else:
            time.sleep(TIME_TO_FIRST_TOKEN + TIME_PER_OUTPUT_TOKEN * len(words))
            self._send_json({
                "id": "chatcmpl-mock", "object": "chat.completion",
                "created": int(time.time()), "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": " ".join(words)},
                    "finish_reason": "stop",
                }],
                # Real servers report this. Watch it: it is your bill.
                "usage": {"prompt_tokens": 42, "completion_tokens": len(words),
                          "total_tokens": 42 + len(words)},
            })

    def _stream(self, words):
        """Server-sent events: one small JSON chunk per token, then [DONE]."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(delta, finish=None):
            chunk = {
                "id": "chatcmpl-mock", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        time.sleep(TIME_TO_FIRST_TOKEN)          # prefill
        send({"role": "assistant", "content": ""})
        for i, word in enumerate(words):
            time.sleep(TIME_PER_OUTPUT_TOKEN)    # one decode step
            send({"content": word if i == 0 else " " + word})
        send({}, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("=" * 70)
    print("MOCK LLM SERVER (no GPU, no model -- just the API shape)")
    print("=" * 70)
    print(f"  listening on : http://127.0.0.1:{args.port}")
    print(f"  model id     : {MODEL_ID}")
    print("  endpoints    : /health, /v1/models, /v1/chat/completions")
    print(f"\n  Now run:  python openai_client.py --base-url http://127.0.0.1:{args.port}/v1")
    print("  Stop with Ctrl+C.\n")

    # 127.0.0.1, not "localhost": on some machines "localhost" resolves to IPv6
    # first and the client wastes ~2 seconds failing over to IPv4 -- which would
    # show up as a nonsense time-to-first-token measurement.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
