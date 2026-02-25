# Auto AI Router

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://mixaill76.github.io/auto_ai_router/)
[![license](https://img.shields.io/github/license/MiXaiLL76/auto_ai_router.svg)](https://github.com/MiXaiLL76/auto_ai_router/blob/main/LICENSE)

High-performance proxy router for LLM APIs with automatic load balancing, rate limiting, and fail2ban protection. Routes requests to OpenAI, Vertex AI, Gemini AI Studio, Anthropic, and other Auto AI Router instances.

## Key Features

- **Multi-provider support** — OpenAI, Vertex AI, Gemini, Anthropic, Proxy chains
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

## Documentation

Full documentation is available at **[mixaill76.github.io/auto_ai_router](https://mixaill76.github.io/auto_ai_router/)**.

## License

Apache License 2.0 — see [LICENSE](LICENSE) file.
