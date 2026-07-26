"""Logging reverse proxy for agent-trace collection (ticket #7).

OpenCode -> :9101 (this) -> VeloxMesh gateway :9100 -> ... -> model.
Captures every request body, response headers (X-Queue-Wait-Ms), and the full
SSE/JSON response body; derives completion token counts from the final usage.
One JSONL row per request.
"""
import json
import sys
import time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = ("127.0.0.1", 9100)
CAPTURE = Path(sys.argv[1] if len(sys.argv) > 1 else "agent_trace.jsonl")


def parse_usage_and_text(content_type, body_bytes):
    """Extract usage dict + assistant text/tool_calls from JSON or SSE body."""
    usage, finish, n_tool_calls, text_len = None, None, 0, 0
    try:
        if content_type.startswith("text/event-stream"):
            for line in body_bytes.decode("utf-8", "replace").splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content"):
                        text_len += len(delta["content"])
                    if delta.get("tool_calls"):
                        n_tool_calls += sum(1 for tc in delta["tool_calls"] if tc.get("id"))
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        else:
            d = json.loads(body_bytes)
            usage = d.get("usage")
            for ch in d.get("choices", []):
                msg = ch.get("message", {})
                text_len = len(msg.get("content") or "")
                n_tool_calls = len(msg.get("tool_calls") or [])
                finish = ch.get("finish_reason")
    except Exception as exc:  # capture must never break the chain
        return {"parse_error": str(exc)}, finish, n_tool_calls, text_len
    return usage, finish, n_tool_calls, text_len


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _relay(self):
        t0 = time.time()
        length = int(self.headers.get("Content-Length") or 0)
        req_body = self.rfile.read(length) if length else b""

        conn = http.client.HTTPConnection(*UPSTREAM, timeout=300)
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "connection", "accept-encoding")}
        hdrs["Accept-Encoding"] = "identity"
        conn.request(self.command, self.path, body=req_body or None, headers=hdrs)
        resp = conn.getresponse()
        resp_body = resp.read()  # buffered relay: fine at agent request rates
        conn.close()

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

        ctype = resp.getheader("Content-Type", "")
        usage, finish, n_tc, text_len = parse_usage_and_text(ctype, resp_body)
        try:
            body_json = json.loads(req_body) if req_body else {}
        except json.JSONDecodeError:
            body_json = {"_unparsed": req_body.decode("utf-8", "replace")[:2000]}
        with CAPTURE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": t0,
                "path": self.path,
                "status": resp.status,
                "e2e_ms": round((time.time() - t0) * 1000, 1),
                "queue_wait_ms": resp.getheader("X-Queue-Wait-Ms"),
                "request_id": resp.getheader("X-Request-Id"),
                "usage": usage,
                "finish_reason": finish,
                "resp_tool_calls": n_tc,
                "resp_text_chars": text_len,
                "body": body_json,
            }, ensure_ascii=False) + "\n")

    def do_POST(self):
        self._relay()

    def do_GET(self):
        self._relay()


if __name__ == "__main__":
    print(f"trace proxy :9101 -> {UPSTREAM}, capturing to {CAPTURE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 9101), Handler).serve_forever()
