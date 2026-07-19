#!/usr/bin/env python3
"""Small aiohttp chat-completions SSE server for the local chain smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

from aiohttp import web


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capture", required=True, type=Path)
    return parser.parse_args()


def _validate_prediction_reliable(body: object) -> tuple[bool, str]:
    if not isinstance(body, Mapping):
        return False, "request body must be a JSON object"
    xargs = body.get("vllm_xargs")
    if not isinstance(xargs, Mapping):
        return False, "vllm_xargs must be a JSON object"
    flag = xargs.get("prediction_reliable")
    if type(flag) is not int:  # bool is deliberately rejected despite bool <: int.
        return False, "vllm_xargs.prediction_reliable must be a JSON integer, not bool"
    if flag not in (0, 1):
        return False, "vllm_xargs.prediction_reliable must be 0 or 1"
    return True, "prediction_reliable is a JSON integer"


async def _append_capture(
    app: web.Application,
    *,
    body: object,
    valid: bool,
    reason: str,
) -> None:
    record = {
        "body": body,
        "validation": {
            "ok": valid,
            "prediction_reliable_json_type": (
                "integer" if valid else "contract_violation"
            ),
            "reason": reason,
        },
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    async with app["capture_lock"]:
        with app["capture_path"].open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        reason = f"request body is not valid JSON: {exc}"
        await _append_capture(
            request.app,
            body={"raw_body": await request.text()},
            valid=False,
            reason=reason,
        )
        return web.json_response({"error": reason}, status=400)

    valid, reason = _validate_prediction_reliable(body)
    await _append_capture(request.app, body=body, valid=valid, reason=reason)
    if not valid:
        return web.json_response({"error": reason}, status=400)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-LTR-Prediction-Reliable-JSON-Type": "integer",
        },
    )
    await response.prepare(request)
    for content in ("one", "two", "three"):
        event = {"choices": [{"delta": {"content": content}}]}
        await response.write(f"data: {json.dumps(event)}\n\n".encode())
    usage = {"choices": [], "usage": {"completion_tokens": 3}}
    await response.write(f"data: {json.dumps(usage)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


def create_app(capture_path: Path) -> web.Application:
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    app = web.Application()
    app["capture_path"] = capture_path
    app["capture_lock"] = asyncio.Lock()
    app.router.add_get("/healthz", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def main() -> None:
    args = parse_args()
    web.run_app(create_app(args.capture), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
