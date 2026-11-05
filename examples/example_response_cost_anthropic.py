#!/usr/bin/env python3
"""
Reading a request's cost from the response -- Anthropic Messages API.

The Anthropic-compatible endpoint (`POST /v1/messages`) reports the cost in the
same place as the token counters:

    response["usage"]["cost"]

This example uses only the standard library, so it works without the `anthropic`
SDK installed. `example_response_cost.py` covers the OpenAI-shaped endpoints
(`/v1/chat/completions`, `/v1/responses`) with the `openai` SDK.

Run:
    python3 example_response_cost_anthropic.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE_URL = os.environ.get("AIR_BASE_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("AIR_API_KEY", "sk-your-master-key-here")
MODEL = os.environ.get("AIR_MODEL", "claude-sonnet-4.6")


def post(path: str, payload: dict, *, stream: bool = False):
    """POST to the router. Without `stream` returns the parsed body; with it, yields SSE events."""
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("x-api-key", API_KEY)
    request.add_header("anthropic-version", "2023-06-01")
    if stream:
        request.add_header("Accept", "text/event-stream")

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:300]}") from exc

    if not stream:
        with response:
            return json.loads(response.read())

    def events():
        with response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    return events()


def main() -> int:
    print(f"base={BASE_URL}")
    print(f"model={MODEL}\n")

    # 1) non-stream
    body = post(
        "/v1/messages",
        {
            "model": MODEL,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Reply with one word: hello"}],
        },
    )
    usage = body.get("usage") or {}
    print(f"  non-stream   cost=${usage.get('cost')}")
    print(f"               input={usage.get('input_tokens')} output={usage.get('output_tokens')}")

    # 2) stream: the cost arrives in the message_delta event's usage.
    cost = None
    for event in post(
        "/v1/messages",
        {
            "model": MODEL,
            "max_tokens": 20,
            "stream": True,
            "messages": [{"role": "user", "content": "Reply with one word: hello"}],
        },
        stream=True,
    ):
        usage = event.get("usage")
        if isinstance(usage, dict) and usage.get("cost") is not None:
            cost = usage["cost"]
    print(f"  stream       cost=${cost}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
