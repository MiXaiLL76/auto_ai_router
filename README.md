# Auto AI Router

<p align="center">
  <img src="docs/logo.svg" alt="Auto AI Router Logo" width="120"/>
</p>

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://mixaill76.github.io/auto_ai_router/)
[![license](https://img.shields.io/github/license/MiXaiLL76/auto_ai_router.svg)](https://github.com/MiXaiLL76/auto_ai_router/blob/main/LICENSE)

High-performance proxy router for LLM APIs with automatic load balancing, rate limiting, and fail2ban protection. Routes requests to OpenAI, Vertex AI, Gemini AI Studio, Anthropic, Comet API, and other Auto AI Router instances.

## Key Features

- **Multi-provider support** — OpenAI, Vertex AI, Gemini, Anthropic, Comet API, Proxy chains
- **Round-robin load balancing** — across multiple credentials per model
- **Rate limiting** — per-credential and per-model RPM/TPM controls
- **Fail2ban** — automatic provider banning on repeated errors
- **Prometheus metrics** — request counts, latency, credential status
- **LiteLLM DB integration** — spend logging and API key authentication
- **Streaming** — full SSE support for all providers
- **Environment variables** — secure credential management via `os.environ/VAR_NAME`

## Quick Start

```bash
# Build
git clone https://github.com/MiXaiLL76/auto_ai_router.git
cd auto_ai_router
go build -o auto_ai_router ./cmd/server/

# Run
./auto_ai_router -config config.yaml
```

Or with Docker:

```bash
docker pull ghcr.io/mixaill76/auto_ai_router:latest
docker run -p 8080:8080 -v $(pwd)/config.yaml:/app/config.yaml ghcr.io/mixaill76/auto_ai_router:latest
```

## Yandex Search passthrough (fork)

`POST /v1/search/yandex` forwards the client's request body to Yandex Cloud's
Search API (`{base_url}/v2/web/search`, `WebSearch.Search`), injecting the
router's own `api_key` and `folder_id` — Yandex only ever sees the router's
egress IP, never the caller's, and the request body is validated against an
allowlisted schema (unknown fields, oversized queries, and smart-snippets
mode are all rejected with `400` before Yandex is ever called). Client auth
reuses the existing gateway key (no new key type); usage is billed into the
same Spend Logs pipeline under `search/yandex` / `yandex_search` at a
day/night marked-up rate, budget is reserved before the upstream call and
settled after, and Yandex's own quota (1 RPS + an hourly cap, shared across
every router replica via Redis) is enforced on top of a per-key RPM limit.
Like every other billed operation, each row is written to LiteLLM's Postgres
*and* published to the Kafka spend-log topic, so the request also shows up in
the downstream ClickHouse analytics (`call_type = search/yandex`, synthetic
`model = yandex-search`).

Disabled by default (`yandex_search.enabled: false`). Even when enabled, a
client key, team, or organization must separately opt in via its LiteLLM
metadata — `metadata.air_scopes: ["yandex_search"]` — otherwise it gets
`403`; this keeps the endpoint from becoming available to every client the
moment an admin turns the feature on. **Requires Redis** for the shared
cross-replica quota — with `redis.enabled: false` (or Redis unreachable at
startup) the endpoint stays unwired (`404`) rather than taking the whole
router down; check the startup logs for "endpoint not wired" if
`/v1/search/yandex` 404s unexpectedly.

```yaml
yandex_search:
  enabled: true
  api_key: "os.environ/YANDEX_SEARCH_API_KEY"
  folder_id: "os.environ/YANDEX_SEARCH_FOLDER_ID"
  rpm: 60          # per client key
  rps: 1           # Yandex's own quota for the whole folder_id — verify before launch
  hourly_limit: 10000
  pricing:
    exchange_rate_rub_per_usd: 85
    rate: 1.3      # markup, already applied to cost_per_query below
    pricing_timezone: "Europe/Moscow"
    night_start: "00:00"
    night_end: "08:00"
    # enabled_operations is the real gate — a price in cost_per_query alone
    # never enables an operation (only web_sync has an adapter today).
    enabled_operations: ["web_sync"]
    cost_per_query:
      web_sync:
        day: 0.007463529412
        night: 0.005597647059
      # image_search / smart_snippets_ru / web_async / ...: catalog for
      # future adapters, see config.yaml.example for the full price list.
```

Calling it as a client (same gateway key as every other endpoint):

```sh
curl -X POST https://<gateway-host>/v1/search/yandex \
  -H "Authorization: Bearer <gateway-key>" \
  -H "Content-Type: application/json" \
  -d '{"query":{"searchType":"SEARCH_TYPE_RU","queryText":"погода в москве"}}'
```

The response is Yandex's own `{"rawData": "<base64 XML/HTML>"}`, passed
through unmodified.

## Documentation

Full documentation is available at **[mixaill76.github.io/auto_ai_router](https://mixaill76.github.io/auto_ai_router/)**.

## License

Apache License 2.0 — see [LICENSE](LICENSE) file.
