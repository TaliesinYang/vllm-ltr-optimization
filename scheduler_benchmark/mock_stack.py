"""CPU-only mock VeloxMesh -> decision service -> vLLM HTTP stack."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping

from scheduler_benchmark.decision_service import (
    DecisionApplication,
    create_decision_server,
)
from scheduler_benchmark.gateway_transport import (
    GatewayDecisionAudit,
    apply_decision_to_payload,
    call_decision_service,
)


class MockEngineServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        self.last_payload: dict[str, object] | None = None
        self.request_count = 0
        super().__init__(server_address, MockEngineHandler)


class MockEngineHandler(BaseHTTPRequestHandler):
    server: MockEngineServer

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            payload = _read_json_request(self)
        except ValueError:
            self.send_error(422)
            return
        self.server.last_payload = payload
        self.server.request_count += 1
        events = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        _write_response(self, 200, "text/event-stream", events)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class MockGatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        decision_endpoint: str,
        engine_endpoint: str,
        decision_timeout_s: float,
    ) -> None:
        self.decision_endpoint = decision_endpoint
        self.engine_endpoint = engine_endpoint
        self.decision_timeout_s = decision_timeout_s
        self.decision_request_count = 0
        self.last_audit: GatewayDecisionAudit | None = None
        super().__init__(server_address, MockGatewayHandler)


class MockGatewayHandler(BaseHTTPRequestHandler):
    server: MockGatewayServer

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            payload = _read_json_request(self)
            request_id = self.headers["X-Request-Id"]
            if not request_id:
                raise ValueError("X-Request-Id is required")
            workflow_id = self.headers.get("X-Workflow-Id", request_id)
            step_id = self.headers.get("X-Step-Id", "0")
            decision_id = f"decision-{request_id}"
            decision_request = _build_decision_request(
                payload,
                request_id=request_id,
                decision_id=decision_id,
                workflow_id=workflow_id,
                step_id=step_id,
                conversation_id=self.headers.get("X-Conversation-Id", request_id),
                previous_tool_gap_ms=int(
                    self.headers.get("X-Previous-Tool-Gap-Ms", "0")
                ),
            )
            engine_payload = _strip_decision_only_xargs(payload)
        except (KeyError, TypeError, ValueError):
            self.send_error(422)
            return

        self.server.decision_request_count += 1
        rpc_result = call_decision_service(
            self.server.decision_endpoint,
            decision_request,
            timeout_s=self.server.decision_timeout_s,
        )
        forwarded, audit = apply_decision_to_payload(
            engine_payload,
            rpc_result,
            expected_decision_id=decision_id,
            workflow_id=workflow_id,
            step_id=step_id,
        )
        self.server.last_audit = audit
        try:
            status, content_type, body = _forward_to_engine(
                self.server.engine_endpoint, forwarded
            )
        except urllib.error.URLError:
            self.send_error(503)
            return
        _write_response(self, status, content_type, body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class MockGatewayStack:
    """Context manager for the three real HTTP boundaries used by CPU E2E."""

    def __init__(
        self,
        application: DecisionApplication,
        *,
        host: str = "127.0.0.1",
        decision_port: int = 0,
        engine_port: int = 0,
        gateway_port: int = 0,
        decision_timeout_s: float = 2.0,
    ) -> None:
        self._host = host
        self._decision_timeout_s = decision_timeout_s
        self._decision_server = create_decision_server(
            application, host=host, port=decision_port
        )
        self._engine_server = MockEngineServer((host, engine_port))
        self._gateway_server = MockGatewayServer(
            (host, gateway_port),
            decision_endpoint=_endpoint(self._decision_server, "/v1/decision"),
            engine_endpoint=_endpoint(self._engine_server, "/v1/chat/completions"),
            decision_timeout_s=decision_timeout_s,
        )
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> MockGatewayStack:
        for server in (
            self._decision_server,
            self._engine_server,
            self._gateway_server,
        ):
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            self._threads.append(worker)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        servers = (
            self._gateway_server,
            self._engine_server,
            self._decision_server,
        )
        for server in servers:
            server.shutdown()
            server.server_close()
        for worker in self._threads:
            worker.join(timeout=2.0)

    @property
    def gateway_endpoint(self) -> str:
        return _endpoint(self._gateway_server, "/v1/chat/completions")

    @property
    def engine_endpoint(self) -> str:
        return _endpoint(self._engine_server, "/v1/chat/completions")

    @property
    def decision_endpoint(self) -> str:
        return _endpoint(self._decision_server, "/v1/decision")

    @property
    def last_engine_payload(self) -> dict[str, object]:
        if self._engine_server.last_payload is None:
            raise RuntimeError("mock engine has not received a request")
        return self._engine_server.last_payload

    @property
    def last_gateway_audit(self) -> GatewayDecisionAudit:
        if self._gateway_server.last_audit is None:
            raise RuntimeError("mock gateway has not made a decision")
        return self._gateway_server.last_audit

    @property
    def decision_request_count(self) -> int:
        return self._gateway_server.decision_request_count

    @property
    def engine_request_count(self) -> int:
        return self._engine_server.request_count


def _build_decision_request(
    payload: Mapping[str, object],
    *,
    request_id: str,
    decision_id: str,
    workflow_id: str,
    step_id: str,
    conversation_id: str,
    previous_tool_gap_ms: int,
) -> dict[str, object]:
    messages = payload.get("messages")
    if messages is None:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt or messages is required")
        messages = [{"role": "user", "content": prompt}]
    request = {
        "schema_version": "1.0",
        "request_id": request_id,
        "decision_id": decision_id,
        "model_id": payload["model"],
        "workflow_id": workflow_id,
        "step_id": step_id,
        "conversation_id": conversation_id,
        "request_age_ms": 0,
        "messages": messages,
        "generation_controls": {
            "temperature": payload.get("temperature", 0.0),
            "top_p": payload.get("top_p", 1.0),
            "seed": payload.get("seed", 42),
            "max_tokens": payload.get("max_tokens", 2048),
            "stream": payload.get("stream", False),
        },
        "previous_tool_gap_ms": previous_tool_gap_ms,
    }
    if "tools" in payload:
        request["tools"] = payload["tools"]
    if "tool_choice" in payload:
        request["tool_choice"] = payload["tool_choice"]
    xargs = payload.get("vllm_xargs", {})
    if not isinstance(xargs, Mapping):
        raise ValueError("vllm_xargs must be an object")
    if "ltr_tool_schema" in xargs:
        tool_schema_text = xargs["ltr_tool_schema"]
        if not isinstance(tool_schema_text, str):
            raise ValueError("ltr_tool_schema must be a string")
        request["tool_schema_text"] = tool_schema_text
    return request


def _strip_decision_only_xargs(
    payload: Mapping[str, object],
) -> dict[str, object]:
    stripped = dict(payload)
    xargs = payload.get("vllm_xargs")
    if isinstance(xargs, Mapping):
        engine_xargs = dict(xargs)
        engine_xargs.pop("ltr_tool_schema", None)
        stripped["vllm_xargs"] = engine_xargs
    return stripped


def _forward_to_engine(
    endpoint: str, payload: Mapping[str, object]
) -> tuple[int, str, bytes]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return (
                response.status,
                response.headers.get_content_type(),
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()


def _read_json_request(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    try:
        content_length = int(handler.headers["Content-Length"])
        parsed = json.loads(handler.rfile.read(content_length))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON request") from exc
    if not isinstance(parsed, dict):
        raise ValueError("request must be an object")
    return parsed


def _write_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    content_type: str,
    body: bytes,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _endpoint(server: ThreadingHTTPServer, path: str) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"
