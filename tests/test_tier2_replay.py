from __future__ import annotations

import unittest
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ltr_training.tier2 import (
    build_request,
    completed_sample_ids,
    replay_labels,
    summarize_results,
)


class Tier2ReplayTests(unittest.TestCase):
    def test_toolace_schema_is_sent_as_openai_tools(self) -> None:
        row = {
            "sample_id": "sample-1",
            "prompt": "use a tool",
            "tool_schema": (
                "Here is a list of functions in JSON format that you can invoke:\n"
                '[{"name":"Lookup Tool","description":"find it","parameters":'
                '{"type":"dict","properties":{"query":{"type":"string"}},'
                '"required":["query"]}}]'
            ),
            "history": [],
        }

        request = build_request(row, model="qwen-tier2")

        tool = request["tools"][0]["function"]
        self.assertEqual(tool["name"], "Lookup_Tool")
        self.assertEqual(tool["parameters"]["type"], "object")
        self.assertEqual(request["messages"], [{"role": "user", "content": "use a tool"}])

    def test_request_pins_greedy_4096_and_disables_thinking(self) -> None:
        row = {
            "sample_id": "sample-1",
            "prompt": "new question",
            "tool_schema": "tools here",
            "history": [["user", "old question"], ["assistant", "old answer"]],
        }

        request = build_request(row, model="qwen-tier2", max_tokens=4096)

        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(request["messages"][-1], {"role": "user", "content": "new question"})
        self.assertEqual(request["messages"][0], {"role": "system", "content": "tools here"})

    def test_only_successful_rows_are_complete_for_resume(self) -> None:
        rows = [
            {"sample_id": "ok", "status": "ok"},
            {"sample_id": "retry", "status": "error"},
        ]

        self.assertEqual(completed_sample_ids(rows), {"ok"})

    def test_summary_reports_censor_failure_and_throughput(self) -> None:
        rows = [
            {"status": "ok", "output_length": 10, "censored": False, "elapsed_seconds": 2.0},
            {"status": "ok", "output_length": 4096, "censored": True, "elapsed_seconds": 4.0},
            {"status": "error", "elapsed_seconds": 1.0},
        ]

        summary = summarize_results(rows, expected_count=3)

        self.assertEqual(summary["attempted"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertAlmostEqual(summary["failure_rate"], 1 / 3)
        self.assertAlmostEqual(summary["censor_rate"], 1 / 2)
        self.assertAlmostEqual(summary["requests_per_second"], 3 / 7)
        self.assertEqual(summary["output_length"]["p50"], 10)
        self.assertEqual(summary["output_length"]["max"], 4096)

    def test_summary_uses_wall_time_for_concurrent_throughput(self) -> None:
        rows = [
            {"status": "ok", "output_length": 100, "elapsed_seconds": 2.0},
            {"status": "ok", "output_length": 100, "elapsed_seconds": 2.0},
        ]

        summary = summarize_results(rows, expected_count=2, wall_elapsed_seconds=2.1)

        self.assertAlmostEqual(summary["elapsed_seconds"], 2.1)
        self.assertAlmostEqual(summary["output_tokens_per_second"], 200 / 2.1)

    def test_concurrent_replay_keeps_jsonl_ledger_valid(self) -> None:
        state = {"active": 0, "max_active": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                with lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                try:
                    time.sleep(0.05)
                    body = json.dumps(
                        {
                            "usage": {"completion_tokens": 12},
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": "complete answer"},
                                }
                            ],
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with lock:
                        state["active"] -= 1

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                labels = root / "labels.jsonl"
                ledger = root / "ledger.jsonl"
                source_rows = [
                    {
                        "sample_id": f"sample-{index}",
                        "source": "toolace",
                        "source_revision": "revision",
                        "session_id": f"session-{index}",
                        "tier2_split": "train",
                        "prompt": f"question {index}",
                        "history": [],
                        "tool_schema": "",
                    }
                    for index in range(4)
                ]
                labels.write_text(
                    "".join(json.dumps(row) + "\n" for row in source_rows),
                    encoding="utf-8",
                )

                rows = replay_labels(
                    labels_path=labels,
                    ledger_path=ledger,
                    endpoint=f"http://127.0.0.1:{server.server_port}",
                    model="test-model",
                    model_revision="revision",
                    concurrency=4,
                    capture_text=True,
                )

                ledger_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
                self.assertEqual(len(rows), 4)
                self.assertEqual(len(ledger_rows), 4)
                self.assertEqual({row["sample_id"] for row in ledger_rows}, {
                    "sample-0", "sample-1", "sample-2", "sample-3"
                })
                self.assertTrue(all(row["response_text"] == "complete answer" for row in rows))
                self.assertTrue(all(row["tier2_split"] == "train" for row in rows))
                self.assertGreaterEqual(state["max_active"], 2)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
