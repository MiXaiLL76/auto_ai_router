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
Air-Usage-Audio-Tokens: include-cached
```

or:

```http
Air-Usage-Audio-Tokens: exclude-cached
```

Downstream AIR reads this header automatically and bills audio/cache correctly.

If this header is missing on a `type: air` credential, AIR uses the legacy AIR contract where provider-converted `audio_tokens` already excludes cached audio. Generic `type: proxy` credentials continue to use raw OpenAI-compatible semantics.

For one rolling-upgrade window, AIR sends and accepts both `Air-Proxy-Client` and the legacy `X-Aar-Proxy-Client` marker. The marker only enables internal response fields when the request also authenticates with the upstream router's master key. A marker supplied with an ordinary client key is ignored.

Roll out upstream routers first, then downstream routers. Mixed-version chains remain billable because the explicit `type: air` contract covers an older upstream that does not yet return `Air-Usage-Audio-Tokens`.

## When to use `proxy`

Use `type: proxy` for generic OpenAI-compatible APIs that are not Auto AI Router instances.

Older `proxy_usage_format` configuration is no longer supported. Use `type: air` for Auto AI Router upstreams.
