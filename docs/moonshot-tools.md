# The router-run Moonshot Search, Search Pro and Fetch tools

Moonshot's models cannot run the built-in search tools themselves: the built-in `$web_search` was switched off on 2026-10-20, and its replacement is Moonshot's own REST APIs, which live outside the chat protocol and which a model cannot reach by itself. The router runs them.

This document shows the whole exchange with real bodies. Every example below is produced by `internal/proxy/moonshot_examples_test.go`, so it is the router's actual behaviour and not a hand-written guess:

```
go test ./internal/proxy/ -run TestMoonshotExamplesArePrinted -v
```

## 1. The client's request

The client asks for the search the ordinary way — the built-in tool, the way any OpenAI-compatible client asks a model that can run it itself:

```http
POST /v1/chat/completions
Authorization: Bearer <client key>
Content-Type: application/json
```

```json
{
  "model": "kimi-k3.1-pro",
  "messages": [{"role": "user", "content": "When was Kimi K2 released?"}],
  "tools": [{"type": "web_search"}]
}
```

For the Pro search the client asks for `{"type": "web_search_pro"}`, and for a page read `{"type": "web_fetch"}`. The three may be asked for together.

The client also may pass a `web_search_options.search_context_size` or the built-in tool's own `search_context_size` — the router carries it, but Moonshot's own tariff does not depend on it (the price is per call).

## 2. What the router sends upstream

The built-in tool is replaced with an ordinary function tool, so the model can only reach it by calling a function. The upstream sees:

```json
{
  "model": "kimi-k3.1-pro",
  "messages": [{"role": "user", "content": "When was Kimi K2 released?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "moonshot_web_search",
        "description": "Search the web and return the matching pages. Arguments: {\"text_query\": string, \"limit\": integer, \"timeout_seconds\": integer, \"include_content\": boolean}. Use this instead of a built-in web search.",
        "parameters": {
          "type": "object",
          "properties": {
            "text_query": {"type": "string", "description": "The search query."},
            "limit": {"type": "integer", "description": "How many results to return."},
            "timeout_seconds": {"type": "integer", "description": "How long the provider may search, in seconds."},
            "include_content": {"type": "boolean", "description": "Whether to include the page content in the results."}
          },
          "required": ["text_query"]
        }
      }
    }
  ]
}
```

The takeover happens **before** the Responses-to-Chat conversion, because that conversion drops every built-in tool: the adapter's replacement is a function tool, which the conversion keeps. This is why the same works on `/v1/responses` too.

The client's own function tools are left exactly as they were — the upstream serves them, and the router does not run a client's tools on its behalf.

## 3. The model asks for the search

The upstream answers with a tool call, as it would for any function:

```json
{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "model": "kimi-k3.1-pro",
  "choices": [{
    "index": 0,
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "moonshot_web_search",
          "arguments": "{\"text_query\":\"kimi k2 release\",\"limit\":5,\"timeout_seconds\":30,\"include_content\":true}"
        }
      }]
    }
  }],
  "usage": {"prompt_tokens": 412, "completion_tokens": 37, "total_tokens": 449}
}
```

## 4. The router's call to Moonshot

The router runs the tool itself, with the credential's key and the credential's proxy. The client's own `Authorization` never reaches Moonshot — the header is built from scratch:

```http
POST https://api.moonshot.ai/v1/tools/search
Authorization: Bearer <the credential's key>
Content-Type: application/json
```

```json
{
  "text_query": "kimi k2 release",
  "limit": 5,
  "timeout_seconds": 30,
  "include_content": true
}
```

The Pro search carries what only it has:

```http
POST https://api.moonshot.ai/v1/tools/search_pro
```

```json
{
  "text_query": "kimi k2 release",
  "limit": 5,
  "timeout_seconds": 30,
  "include_content": true,
  "sites": ["arxiv.org"],
  "time_window": "week"
}
```

And the fetch carries only the URL:

```http
POST https://api.moonshot.ai/v1/tools/fetch
```

```json
{"url": "https://moonshot.ai/news/kimi-k2"}
```

The provider answers with the pages it found:

```json
{
  "search_results": [{
    "title": "Moonshot AI releases Kimi K2",
    "url": "https://moonshot.ai/news/kimi-k2",
    "snippet": "Moonshot AI has released the Kimi K2 model.",
    "content": "Kimi K2 is Moonshot AI's ..."
  }]
}
```

## 5. The result goes back to the model

The router appends the assistant's tool-call message and the tool result to the conversation, keeping the `tool_call_id`, and asks the model again:

```json
{
  "model": "kimi-k3.1-pro",
  "messages": [
    {"role": "user", "content": "When was Kimi K2 released?"},
    {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "moonshot_web_search",
          "arguments": "{\"text_query\":\"kimi k2 release\",\"limit\":5,\"timeout_seconds\":30,\"include_content\":true}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "moonshot_web_search",
      "content": "{\"results\":[{\"title\":\"Moonshot AI releases Kimi K2\",\"url\":\"https://moonshot.ai/news/kimi-k2\",\"snippet\":\"Moonshot AI has released the Kimi K2 model.\",\"summary\":\"\",\"content\":\"Kimi K2 is Moonshot AI's ...\",\"site_name\":null,\"date\":null}]}"
    }
  ],
  "tools": [ ... the same function tool ... ]
}
```

The model's `reasoning_content` is preserved, and the loop continues until the model answers without asking for a tool (bounded by `server.moonshot_tools.max_steps`, default 8).

## 6. The client's answer

The client gets the final answer — never the tool call, because it declared a `web_search` and not the router's function:

```json
{
  "id": "chatcmpl-2",
  "object": "chat.completion",
  "model": "kimi-k3.1-pro",
  "choices": [{
    "index": 0,
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "Moonshot AI released Kimi K2 on 2025-07-11."
    }
  }],
  "usage": {"prompt_tokens": 1587, "completion_tokens": 64, "total_tokens": 1651}
}
```

On a streaming request the client gets the last step's stream; the steps that asked for the tool are not relayed.

This holds for the Responses API too: `/v1/responses` with `stream: true` gets the last step's stream converted to the Responses shape (`response.output_text.delta`, `response.completed`), not the model's tool call. How Codex talks.

## 7. The spend row

The client's answer is billed for the tokens of **both** steps (the loop's steps are separate upstream requests) and for the router's own search call:

```
Spend: 0.015201
PromptTokens: 1999          (412 + 1587, both steps)
CompletionTokens: 101       (37 + 64, both steps)
```

```
tokens:  1999 * 1e-6 + 101 * 2e-6 = 0.002201
search:  1 * 0.013              = 0.013
                                 ------
                                 0.015201
```

The breakdown in the spend row's metadata:

```json
"cost_breakdown": {
  "input_cost": 0.001999,
  "output_cost": 0.000202,
  "moonshot_search_cost": 0.013,
  "moonshot_search_pro_cost": 0,
  "moonshot_fetch_cost": 0,
  "web_search_cost": 0,
  "web_fetch_cost": 0,
  "tool_usage_cost": 0.013,
  "total_cost": 0.015201
},
"usage_object": {
  "prompt_tokens": 1999,
  "completion_tokens": 101,
  "total_tokens": 2100,
  "server_tool_use": {
    "moonshot_search_requests": 1,
    "moonshot_search_pro_requests": 0,
    "moonshot_fetch_requests": 0,
    "web_search_requests": 0,
    "web_search_context_size": null
  }
}
```

Note that `web_search_requests` is **0**: it is the model's own built-in search, which the provider never ran here. The router's call is counted only as `moonshot_search_requests`, because it is priced by Moonshot's own tariff — counting it in both would charge the client twice.

## When the step budget runs out

The model may keep asking for the search — and each round is another upstream request (and another bill). The loop is bounded by `server.moonshot_tools.max_steps` (default 8).

When the budget is spent, the router asks the model to answer with what it has found, **with the tools taken away**. A tool call for the router's own function must never reach the client: the client declared a built-in `web_search`, not a function named `moonshot_web_search`, so a tool call for it would be a call for a tool nobody declared.

## The control spends

| What happened                                       | Tool calls billed | Tool cost  |
| --------------------------------------------------- | ----------------- | ---------- |
| One successful Basic search                         | 1 Basic           | $0.013     |
| One successful Pro search                           | 1 Pro             | $0.013     |
| One successful Fetch (page content)                 | 1 Fetch           | $0.013     |
| One Basic + one Pro + one Fetch                     | 3                 | **$0.039** |
| A search that returned nothing                      | 0                 | $0         |
| A search that failed (400/401/402/403/429/5xx)      | 0                 | $0         |
| A search that timed out                             | 0                 | $0         |
| A Fetch whose page had no content                   | 0                 | $0         |
| A Fetch the provider refused (`markdown_not_found`) | 0                 | $0         |

The tokens of every step of the loop are billed separately, on top of the tool calls.

One successful Basic search, one Pro search and one Fetch = **$0.039** for the tools.
