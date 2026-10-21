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

## Health-check integration (fork)

Общие баны upstream-аккаунтов вынесены в отдельный воркер
(healthcheck-service): решения принимаются по событиям из Kafka и живут в
Redis, а этот роутер работает в режиме «публикуй и слушай»:

- **`account_events`** — при включении каждый исход запроса (успех и ошибка,
  с провайдером и `Retry-After`) публикуется в топик Kafka `account-events`;
  решение о бане роутер локально не принимает (`recordBanSignal` пропускает
  собственные fail2ban-счётчики).
- **Ban reader** — раз в `account_events.reader_interval` роутер читает баны
  воркера из Redis (`redis.health_key_prefix` — отдельный namespace, не общий
  `key_prefix`) и материализует их в локальный fail2ban-кэш: модельные баны
  снимают с ротации конкретные модели, wildcard-баны — весь аккаунт.
  Снапшот атомарен: при сбое чтения баны не снимаются.
- Если Redis недоступен — ридер не подключается, локальные баны остаются
  активными (безопасная деградация); через `fail2ban.error_codes: []`
  локальные решения можно отключить сознательно.

```yaml
account_events:
  enabled: true
  topic: "account-events"
  reader_interval: 2s
redis:
  enabled: true
  addresses: ["valkey.valkey.svc:6379"]
  health_key_prefix: "hc:"   # ключи воркера — в отдельном namespace
```

## Documentation

Full documentation is available at **[mixaill76.github.io/auto_ai_router](https://mixaill76.github.io/auto_ai_router/)**.

## License

Apache License 2.0 — see [LICENSE](LICENSE) file.
