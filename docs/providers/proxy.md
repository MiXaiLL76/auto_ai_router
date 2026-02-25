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
