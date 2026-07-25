"""OpenAI-compatible capture endpoint, v2: forces a multi-turn agent loop.

Returns a tool_call (read-only `glob`) until the conversation already
contains 2 tool results, then stops. Captures every request body so we can
measure tool-schema variability ACROSS turns within one session.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CAPTURE = Path(__file__).with_name("captured_requests_v2.jsonl")
MAX_TOOL_TURNS = 2


def count_tool_results(messages):
    n = 0
    for m in messages or []:
        if m.get("role") == "tool":
            n += 1
        # some clients fold tool results into user messages as parts
        content = m.get("content")
        if isinstance(content, list):
            n += sum(1 for part in content
                     if isinstance(part, dict) and part.get("type") == "tool-result")
    return n


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "probe-model", "object": "model", "owned_by": "probe"}
            ]})
        else:
            self._json(200, {"ok": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"_unparsed": raw.decode("utf-8", "replace")}

        with CAPTURE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.time(),
                "path": self.path,
                "body": body,
            }, ensure_ascii=False) + "\n")

        has_tools = bool(body.get("tools"))
        done_turns = count_tool_results(body.get("messages"))
        want_tool_call = has_tools and done_turns < MAX_TOOL_TURNS

        if body.get("stream"):
            self._stream(body, want_tool_call, done_turns)
            return

        if want_tool_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_probe_{done_turns}",
                    "type": "function",
                    "function": {
                        "name": "glob",
                        "arguments": json.dumps({"pattern": "*.md"}),
                    },
                }],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "probe done"}
            finish = "stop"

        self._json(200, {
            "id": "chatcmpl-probe",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "probe-model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def _stream(self, body, want_tool_call, done_turns):
        base = {
            "id": "chatcmpl-probe",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "probe-model"),
        }
        if want_tool_call:
            deltas = [
                {"role": "assistant", "tool_calls": [{
                    "index": 0,
                    "id": f"call_probe_{done_turns}",
                    "type": "function",
                    "function": {
                        "name": "glob",
                        "arguments": json.dumps({"pattern": "*.md"}),
                    },
                }]},
            ]
            finish = "tool_calls"
        else:
            deltas = [{"role": "assistant", "content": "probe done"}]
            finish = "stop"

        chunks = [dict(base, choices=[{"index": 0, "delta": d, "finish_reason": None}])
                  for d in deltas]
        chunks.append(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}],
                           usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))

        payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        data = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"capturing to {CAPTURE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8099), Handler).serve_forever()
