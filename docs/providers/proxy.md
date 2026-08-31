# Proxy

The proxy provider forwards requests to a generic OpenAI-compatible API. For another Auto AI Router instance, prefer [`type: air`](air.md).

## Configuration

```yaml
credentials:
  - name: "proxy_fallback"
    type: "proxy"
    base_url: "https://openai-compatible.example/v1"
    api_key: "sk-upstream-key" # Optional
    rpm: 200
    tpm: 100000
    priority: 999   # last-resort tier (is_fallback: true is a deprecated alias)
```

## Required Fields

| Field      | Description                      |
| ---------- | -------------------------------- |
| `base_url` | URL of the OpenAI-compatible API |

## Optional Fields

| Field         | Description                                                                       |
| ------------- | --------------------------------------------------------------------------------- |
| `api_key`     | Remote master key (if the target requires authentication)                         |
| `priority`    | Selection-order group (lower first). `999` = last-resort. `is_fallback: true` is a deprecated alias for `priority: 999` |

## Usage Contract

Generic OpenAI-compatible APIs usually report `audio_tokens` including `cached_audio_tokens`. AIR subtracts cached audio before billing ordinary audio for these responses.

AIR responses also include an explicit cached-audio usage contract:

```http
Air-Usage-Audio-Tokens: exclude-cached
```

or:

```http
Air-Usage-Audio-Tokens: include-cached
```

Downstream AIR reads this header automatically.

## Migration / Rollout Note

For AIR-to-AIR chained routing, switch credentials to `type: air`. Roll out the upstream AIR first. During the transition, current AIR releases read and write both the current `Air-Proxy-Client` marker and the legacy `X-Aar-Proxy-Client` marker.

## Last-Resort Behavior

A proxy credential at `priority: 999` (the last-resort group; `is_fallback: true` is a
deprecated alias) is only selected after every lower-numbered priority tier for the
requested model is unavailable (rate-limited or banned). It is part of the same weighted
priority cascade as every other tier — there is no separate fallback pool.

### Processing Chain

1. Request arrives for a model (e.g., `gpt-4o`)
2. Router cascades through priority tiers ascending, weighted round-robin within a tier
3. It drops to the next tier only when the current one is fully banned or rate-limited
4. If even the last-resort tier is exhausted → client receives `503 Service Unavailable`

### Example: Router Chain

```yaml
credentials:
  # Primary provider
  - name: "openai_main"
    type: "openai"
    api_key: "sk-..."
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000

  # Fallback: another Auto AI Router instance
  - name: "backup_router"
    type: "air"
    base_url: "http://10.0.1.50:8080"
    api_key: "sk-remote-key"
    priority: 999   # last-resort tier (is_fallback: true is a deprecated alias)
```

When `openai_main` exhausts its rate limits, requests automatically route to `backup_router`.

### Multiple Fallbacks

You can configure multiple fallback proxy/AIR credentials — they are also load-balanced using round-robin:

```yaml
credentials:
  - name: "openai_main"
    type: "openai"
    api_key: "sk-..."
    base_url: "https://api.openai.com"
    rpm: 100

  - name: "backup_1"
    type: "air"
    base_url: "http://router-1:8080"
    priority: 999   # last-resort tier (is_fallback: true is a deprecated alias)
  - name: "backup_2"
    type: "air"
    base_url: "http://router-2:8080"
    priority: 999   # last-resort tier (is_fallback: true is a deprecated alias)
```
