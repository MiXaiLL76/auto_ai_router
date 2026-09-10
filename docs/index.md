# Auto AI Router

High-performance proxy router for LLM APIs with automatic load balancing, rate limiting, and fail2ban protection.

## Overview

Auto AI Router acts as a unified gateway between your applications and multiple LLM providers. It handles credential rotation, rate limit enforcement, and automatic failover — so your applications only need a single endpoint.

```mermaid
graph LR
    Client["<b>Client</b><br/>(OpenAI format)"]

    Router{{"<b>Auto AI Router</b><br/>- Load balancing<br/>- Rate limiting<br/>- Fail2ban"}}

    subgraph Providers ["Backend Providers"]
        OpenAI[OpenAI]
        Vertex[Vertex AI]
        Anthropic[Anthropic]
        Gemini[Gemini]
        Bedrock[Bedrock]
        Proxy[Proxy]
        AIR[AIR]
    end

    Client <--> Router
    Router --> OpenAI
    Router --> Vertex
    Router --> Anthropic
    Router --> Gemini
    Router --> Bedrock
    Router --> Proxy
    Router --> AIR
```

## Features

- **Multi-provider routing** — OpenAI, Vertex AI, Gemini AI Studio, Anthropic, Comet API
- **AIR chains** — forward to other Auto AI Router instances as fallback
- **Round-robin balancing** — distribute load across multiple credentials
- **Two-level rate limiting** — per-credential RPM/TPM + per-model limits
- **Fail2ban protection** — auto-ban credentials on repeated errors
- **Prometheus metrics** — request counts, latency, credential utilization
- **OpenTelemetry** — export traces, logs, and metrics over OTLP to any collector
- **Health dashboard** — JSON API and HTML dashboard at `/health` and `/vhealth`
- **LiteLLM DB integration** — spend logging, daily aggregation, API key auth
- **SSE streaming** — full streaming support for all providers
- **Secure config** — environment variable resolution via `os.environ/VAR_NAME`

## Getting Started

1. [Installation](getting-started/installation.md) — build from source or use Docker
2. [Configuration](getting-started/configuration.md) — set up providers and credentials
3. [API Usage](getting-started/api.md) — make your first request

## Advanced

- [Load Balancing](advanced/balancing.md) — round-robin, priority tiers, last-resort
- [Session-Sticky Routing](advanced/session_sticky.md) — cache affinity for multi-turn conversations
- [Responses API](advanced/responses.md) — multi-turn support, passthrough models
- [Redis / Valkey](advanced/redis.md) — distributed rate limiting and response store
- [Security](advanced/security.md) — master key, token auth
- [Model Aliases](advanced/model_alias.md) — routing by alias
- [Troubleshooting](advanced/troubleshooting.md)

## LiteLLM Integration

- [LiteLLM Database](litellm-integration/litellm_db.md) — spend logging, daily aggregation, API key auth
- [Model Pricing](litellm-integration/pricing.md) — price file format, cost calculation formulas
