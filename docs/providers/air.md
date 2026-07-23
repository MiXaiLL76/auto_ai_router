# Auto AI Router

Use `type: air` when the upstream is another Auto AI Router instance.

It uses the same HTTP forwarding, `/health` discovery, dynamic model sync, and fallback behavior as `type: proxy`, but declares a narrower product contract: the remote endpoint is AIR, not a generic OpenAI-compatible API.

## Configuration

```yaml
credentials:
  - name: "backup_router"
    type: "air"
    base_url: "http://backup-router.local:8080"
    api_key: "sk-remote-master-key" # Optional
    rpm: 200
    tpm: 100000
    is_fallback: true
```

## Usage Contract

AIR responses carry their cached-audio usage contract in:

```http
X-Aar-Usage-Audio-Tokens: include-cached
```

or:

```http
X-Aar-Usage-Audio-Tokens: exclude-cached
```

Downstream AIR reads this header automatically and bills audio/cache correctly.

If the header is missing, AIR falls back to OpenAI-compatible semantics where `audio_tokens` includes `cached_audio_tokens`. This keeps older or partially rolled out upstreams safe for raw Chat Completions passthrough.

## When to use `proxy`

Use `type: proxy` for generic OpenAI-compatible APIs that are not Auto AI Router instances.

Older `proxy_usage_format` configuration is no longer supported. Use `type: air` for Auto AI Router upstreams.
