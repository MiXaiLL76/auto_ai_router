#!/usr/bin/env python3
"""
Reading a request's cost from the response.

The router bills every request and reports what it cost in the response body, in
`usage.cost` (USD) -- the same place the token counters live, so a client that
already reads `usage` gets the cost for free:

    response["usage"]["cost"]

The cost is the router's own cost for the request: input tokens at the model's
input rate, output tokens at its output rate, and cached prompt tokens at the
cache-read rate. It is NOT the upstream provider's invoice -- the router
resells at its own price, so compare against your own price list if that is what
you bill on.

Which organization gets the cost is a per-organization setting
(`response_cost_mode`): `off` (the default) sends nothing, `total` sends
`usage.cost`, `breakdown` sends `usage.cost` plus a per-component breakdown.

Run:
    python3 example_response_cost.py
"""

import os
import sys
from typing import Optional

from openai import OpenAI

# Point these at your router. The defaults target a local proxy.
BASE_URL = os.environ.get("AIR_BASE_URL", "http://localhost:8080/v1")
API_KEY = os.environ.get("AIR_API_KEY", "sk-your-master-key-here")


def cost_of(response) -> Optional[float]:
    """The request's cost, or None when the organization is not opted in."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    # The OpenAI SDK types do not declare `cost` (it is a router extension), so
    # it is read off the model object rather than the typed attribute.
    return getattr(usage, "cost", None)


def chat(client: OpenAI, model: str) -> None:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one word: hello"}],
        max_tokens=20,
    )
    usage = response.usage
    print(f"  non-stream   cost=${cost_of(response)}")
    print(f"               prompt={usage.prompt_tokens} completion={usage.completion_tokens}")


def chat_stream(client: OpenAI, model: str) -> None:
    # `stream_options={"include_usage": True}` is what makes the provider send
    # a final usage chunk. Without it there is no usage at all in the stream,
    # and therefore no cost.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one word: hello"}],
        max_tokens=20,
        stream=True,
        stream_options={"include_usage": True},
    )
    cost = None
    for chunk in stream:
        if chunk.usage is not None:
            cost = getattr(chunk.usage, "cost", None)
    print(f"  stream       cost=${cost}")


def responses(client: OpenAI, model: str) -> None:
    response = client.responses.create(
        model=model,
        input="Reply with one word: hello",
        max_output_tokens=20,
    )
    usage = response.usage
    print(f"  responses    cost=${getattr(usage, 'cost', None)}")
    print(f"               input={usage.input_tokens} output={usage.output_tokens}")


def main() -> int:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    print(f"base={BASE_URL}")
    print(f"model={model}\n")

    for name, call in (
        ("chat/completions", chat),
        ("chat/completions stream", chat_stream),
        ("responses", responses),
    ):
        try:
            call(client, model)
        except Exception as exc:  # noqa: BLE001 - a demo script reports, it does not raise
            print(f"  {name:<24} failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
