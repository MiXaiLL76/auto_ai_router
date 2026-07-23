# Proxy

The proxy provider forwards requests to another Auto AI Router instance (or any OpenAI-compatible API). This enables router chaining and fallback configurations.

## Configuration

```yaml
credentials:
  - name: "proxy_fallback"
    type: "proxy"
    base_url: "http://backup-router.local:8080"
    api_key: "sk-remote-master-key"  # Optional
    rpm: 200
    tpm: 100000
    is_fallback: true
```

## Required Fields

| Field      | Description                                       |
| ---------- | ------------------------------------------------- |
| `base_url` | URL of the remote router or OpenAI-compatible API |

## Optional Fields

| Field         | Description                                                                       |
| ------------- | --------------------------------------------------------------------------------- |
| `api_key`     | Remote master key (if the target requires authentication)                         |
| `is_fallback` | When `true`, this credential is only used after primary credentials are exhausted |

## Usage Contract

Generic OpenAI-compatible APIs usually report `audio_tokens` including
`cached_audio_tokens`. AIR subtracts cached audio before billing ordinary audio
for these responses.

AIR-normalized responses report `audio_tokens` as non-cached audio only and
expose `cached_audio_tokens` separately. When AIR emits such a response, it sets:

```http
X-Aar-Usage-Audio-Tokens: exclude-cached
```

Downstream AIR proxies read this header automatically, so AIR-to-AIR chains do
not need per-credential usage-format configuration.

`proxy_usage_format: "normalized"` is still accepted as a legacy override for
older upstream AIR deployments that do not emit the header yet, but new configs
should not need it.

## Migration / Rollout Note

For AIR-to-AIR chained routing, roll out the upstream AIR first or at least
ensure the upstream emits `X-Aar-Usage-Audio-Tokens: exclude-cached` on
normalized responses. Once that header is present, downstream proxy credentials
can stay as plain `type: proxy`.

## Fallback Behavior

When `is_fallback: true`, the proxy credential activates only after all primary credentials for the requested model are unavailable (rate-limited or banned).

### Processing Chain

1. Request arrives for a model (e.g., `gpt-4o`)
2. Router tries primary credentials in round-robin order
3. If all primary credentials are exhausted → router tries fallback proxies
4. If fallback proxy is also unavailable → client receives `503 Service Unavailable`

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
    type: "proxy"
    base_url: "http://10.0.1.50:8080"
    api_key: "sk-remote-key"
    is_fallback: true
```

When `openai_main` exhausts its rate limits, requests automatically route to `backup_router`.

### Multiple Fallbacks

You can configure multiple fallback proxies — they are also load-balanced using round-robin:

```yaml
credentials:
  - name: "openai_main"
    type: "openai"
    api_key: "sk-..."
    base_url: "https://api.openai.com"
    rpm: 100

  - name: "backup_1"
    type: "proxy"
    base_url: "http://router-1:8080"
    is_fallback: true

  - name: "backup_2"
    type: "proxy"
    base_url: "http://router-2:8080"
    is_fallback: true
```
