# Proxy

The proxy provider forwards requests to another Auto AI Router instance (or any OpenAI-compatible API). This enables router chaining and fallback configurations.

## Configuration

```yaml
credentials:
  - name: "proxy_fallback"
    type: "proxy"
    base_url: "http://backup-router.local:8080"
    api_key: "sk-remote-master-key"  # Optional
    proxy_usage_format: "normalized" # Use only when upstream audio_tokens excludes cached audio
    rpm: 200
    tpm: 100000
    is_fallback: true
```

## Required Fields

| Field      | Description                                       |
| ---------- | ------------------------------------------------- |
| `base_url` | URL of the remote router or OpenAI-compatible API |

## Optional Fields

| Field                | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| `api_key`            | Remote master key (if the target requires authentication)                         |
| `is_fallback`        | When `true`, this credential is only used after primary credentials are exhausted |
| `proxy_usage_format` | Usage contract: `openai` (default) or `normalized`                                |

`proxy_usage_format` controls cached-audio accounting:

- `openai` means `audio_tokens` includes `cached_audio_tokens`; the router subtracts cached audio before billing ordinary audio. This is the safe default for generic OpenAI-compatible APIs.
- `normalized` means `audio_tokens` already contains only non-cached audio and `cached_audio_tokens` is reported separately. Use this only when the upstream explicitly guarantees that contract, including AIR deployments configured to expose normalized usage.

The setting is explicit because `type: proxy` can target either kind of API; the wire payload alone is ambiguous when both fields are present.

## Migration / Rollout Note

Before rolling this change out to production, review every existing `type: proxy` credential:

- Use `proxy_usage_format: "normalized"` when the upstream is another Auto AI Router instance.
- Use `proxy_usage_format: "openai"` or omit the field when the upstream is a generic OpenAI-compatible API.

The default is `openai` for backward compatibility. Existing AIR-to-AIR chains must be updated explicitly, otherwise cached audio can be subtracted twice from `audio_tokens` and audio input spend can be understated.

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
    proxy_usage_format: "normalized"
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
