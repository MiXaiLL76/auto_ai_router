# OpenAI

## Configuration

```yaml
credentials:
  - name: "openai_main"
    type: "openai"
    api_key: "sk-proj-xxxxx"
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000
```

## Required Fields

| Field      | Description                                     |
| ---------- | ----------------------------------------------- |
| `api_key`  | OpenAI API key (supports `os.environ/VAR_NAME`) |
| `base_url` | API base URL (`https://api.openai.com`)         |

## Azure OpenAI

For Azure OpenAI, use the same `openai` type with the Azure endpoint:

```yaml
credentials:
  - name: "azure_openai"
    type: "openai"
    api_key: "os.environ/AZURE_OPENAI_KEY"
    base_url: "https://your-resource.openai.azure.com"
    rpm: 100
    tpm: 50000
```

## Per-Model Configuration

Use the `models` section to control per-model behavior:

```yaml
models:
  - model: "gpt-4o-mini"
    rpm: 500
    tpm: 200000

  - model: "gpt-5.3-codex"
    passthrough_responses: true   # forward Responses API requests natively (auto-detected for codex models)

  - model: "gpt-4o"
    passthrough_responses: false  # force Chat Completions conversion even if model name contains "codex"
```

### `passthrough_responses`

Controls whether Responses API (`/v1/responses`) requests for this model are forwarded natively to the provider's `/v1/responses` endpoint instead of being converted to Chat Completions format.

| Value   | Behavior                                                                 |
| ------- | ------------------------------------------------------------------------ |
| `true`  | Always forward as native Responses API (no Chat Completions conversion)  |
| `false` | Always convert to Chat Completions (even if model name contains "codex") |
| omitted | Auto-detect: `true` if model name contains `codex`, `false` otherwise    |

This is useful when deploying non-codex models that natively support `/v1/responses`, or to force conversion for codex-named models behind a Chat Completions-only endpoint.

## Responses API

Requests to `/v1/responses` are automatically detected and handled:

- **Standard models** (gpt-4o, gpt-4o-mini, o1, o3, etc.): converted to Chat Completions format internally, response converted back. Works with Azure OpenAI and all other OpenAI-compatible backends.
- **Codex / passthrough models**: forwarded natively to the provider's `/v1/responses` endpoint without conversion.

See [Responses API documentation](../advanced/responses.md) for full details.

## Web Search Tools

Tool handling follows the outbound endpoint contract:

- Native `/v1/responses` passthrough keeps built-in tools and `tool_choice` unchanged, including `web_search`, `file_search`, `computer_use`, and `code_interpreter`.
- `/v1/chat/completions` converts `web_search` and `web_search_preview` to top-level `web_search_options` for models whose name contains `search-preview`, including `search_context_size` and `user_location`.
- Chat Completions requests for other models preserve `web_search`, `web_search_preview`, and matching `tool_choice` values for upstream validation. Ordinary function tools in the same request also remain unchanged.

```python
# Converted to web_search_options for Chat Completions
client.chat.completions.create(
    model="gpt-4o-search-preview",
    tools=[{"type": "web_search_preview", "search_context_size": "high"}],
    ...,
)

# Built-in tool is preserved for upstream validation
client.chat.completions.create(model="gpt-5.6-sol", tools=[{"type": "web_search"}], ...)
```

Other non-function tool types (e.g. `computer_use`, `code_execution`) are also dropped for OpenAI Chat Completions requests. They are handled by their respective providers (Vertex AI, Anthropic) when those backends are configured.
