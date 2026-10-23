# Session-Sticky Credential Routing

Session-sticky routing ensures that requests belonging to the same conversation or session are routed to the **same backend credential** on every turn. This maximises the chance of hitting the provider's prompt cache and reduces redundant token processing costs.

## Why It Matters

Modern LLM providers maintain an internal prompt cache. When consecutive turns of a conversation reach the same model instance behind the same API key, the provider can reuse the cached KV-state of the prefix and charge only for new tokens:

| Scenario               | Without sticky routing | With sticky routing           |
| ---------------------- | ---------------------- | ----------------------------- |
| Turn 1 (10 000 tokens) | cred_A, full cost      | cred_A, full cost             |
| Turn 2 (10 200 tokens) | cred_B, full cost      | cred_A, 200 new tokens billed |
| Turn 3 (10 400 tokens) | cred_A, full cost      | cred_A, 200 new tokens billed |

The savings scale with context length. For long conversations the effective cost can drop by 80–90 %.

Sticky routing also helps providers that require continuity for features like:

- Vertex AI `thoughtSignature` (reasoning continuation)
- Anthropic extended thinking across turns

## How It Works

### Session-Sticky Routing

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as Auto AI Router
    participant Store as SessionStore<br/>(Redis, or in-memory without Redis)
    participant Bal as Balancer
    participant Cred as credential

    Client->>Proxy: POST /v1/chat/completions<br/>{"user": "session-abc", "model": "gemini-2.5-flash", ...}
    Proxy->>Store: Get("session-abc", "gemini-2.5-flash")
    alt binding exists and credential available
        Store-->>Proxy: "vertex_cred_2"
        Proxy->>Bal: NextSpecific("vertex_cred_2", "gemini-2.5-flash")
        Bal-->>Proxy: vertex_cred_2
    else no binding or credential unavailable
        Store-->>Proxy: not found
        Proxy->>Bal: NextForModel("gemini-2.5-flash")
        Bal-->>Proxy: vertex_cred_1 (round-robin)
    end
    Proxy->>Cred: forward request
    Cred-->>Proxy: 200 OK (full response delivered)
    Proxy->>Store: Set("session-abc", "gemini-2.5-flash", "vertex_cred_1")
    Proxy-->>Client: response
```

### Responses API Sticky Routing (via `previous_response_id`)

When using the Responses API with `previous_response_id`, the proxy additionally remembers which credential served each stored response. This gives an even stronger affinity guarantee:

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as Auto AI Router
    participant RS as ResponseStore<br/>(bbolt / Redis)
    participant Store as SessionStore
    participant Bal as Balancer

    Client->>Proxy: POST /v1/responses<br/>{"previous_response_id": "resp-xyz", ...}
    Proxy->>RS: GetEntry("resp-xyz")
    RS-->>Proxy: StoredEntry{CredentialName: "openai_cred_3"}

    Proxy->>Bal: NextSpecific("openai_cred_3", "gpt-4o")
    alt credential available
        Bal-->>Proxy: openai_cred_3
    else credential unavailable
        Proxy->>Store: Get(sessionID, "gpt-4o")
        alt session binding exists
            Store-->>Proxy: "openai_cred_2"
            Proxy->>Bal: NextSpecific("openai_cred_2", "gpt-4o")
        else no binding
            Proxy->>Bal: NextForModel("gpt-4o")
        end
    end

    Proxy-->>Client: response
    Note over Proxy,RS: On success: binding updated,<br/>response saved with CredentialName
```

### Credential Selection Priority

```mermaid
flowchart TD
    Start([Incoming request]) --> HasPrev{Has<br/>previous_response_id?}

    HasPrev -->|yes| LoadEntry[Load StoredEntry<br/>from ResponseStore]
    LoadEntry --> EntryHasCred{Entry has<br/>CredentialName?}
    EntryHasCred -->|yes| TryPref[NextSpecific<br/>preferredCredential]
    TryPref --> PrefOK{Available?}
    PrefOK -->|yes| UsePref([Use preferred credential])
    PrefOK -->|no| TrySession

    HasPrev -->|no| TrySession
    EntryHasCred -->|no| TrySession

    TrySession{sessionID set<br/>and sticky enabled?}
    TrySession -->|yes| LookupBinding[SessionStore.Get<br/>sessionID + modelID]
    LookupBinding --> BindingOK{Credential<br/>available?}
    BindingOK -->|yes| UseBinding([Use session-bound credential])
    BindingOK -->|no| RoundRobin

    TrySession -->|no| RoundRobin[NextForModel<br/>round-robin]
    RoundRobin --> RROk{Found?}
    RROk -->|yes| UseRR([Use round-robin credential])
    RROk -->|no| Fallback[NextFallbackForModel]
    Fallback --> FbOk{Found?}
    FbOk -->|yes| UseFb([Use fallback credential])
    FbOk -->|no| Error([503 No credentials available])
```

### Binding Lifecycle

The binding is **only written on full success** — after the entire response has been delivered to the client without error. This prevents a failed request from locking future traffic to a bad credential.

```mermaid
stateDiagram-v2
    [*] --> Selecting: Incoming request

    Selecting --> Forwarding: credential selected\n(binding NOT updated yet)

    Forwarding --> Success: response fully delivered\n(2xx + no write error)
    Forwarding --> Failure: upstream error\nor client disconnect\nor stream error

    Success --> [*]: setSessionBinding(sessionID, modelID, cred)\nRequestCompleted = true

    Failure --> [*]: clearSessionBinding(sessionID, modelID)\nunless the client disconnected\nor the bound credential was only held
```

Key consequences:

- A binding points to a credential that **successfully completed at least one request** and is therefore likely to have a warm cache.
- The cascade decides, the binding follows. If the bound credential is **busy** (its own RPM/TPM limit is reached) or excluded for this one request (a reasoning-only credential on a plain turn), the turn is served elsewhere and **the binding is kept**: the next turn goes back to it and its warm cache.
- If the bound credential **fails or is banned**, the session moves to the credential that served the turn: a credential that keeps erroring hands its users to the next one.
- Sessions are bound only to credentials in the best priority tier the model has (fallback credentials and lower tiers excluded). A turn served by a lower tier clears any existing binding instead of pinning it to a reserve credential (or to whatever failed and got rescued by the reserve), so the next turn tries the primary tier again from a clean slate.
- A client disconnect (499) does not clear the binding: it says nothing about the credential.
- A binding is refreshed only when a credential serves a turn, so a binding held for a busy credential expires one TTL after it last served.
- Bindings from client API keys are scoped by the hashed key, so two customers sending the same `user` or `prompt_cache_key` do not share a binding. Master-key traffic (AIR-to-AIR hops) keeps the raw session ID.

### Multiple replicas

With `redis.enabled`, bindings are stored in Redis/Valkey under `{redis.key_prefix}session:{sha256(model, session)}` with a native TTL, so every replica of a deployment sees the same binding. Without Redis each replica keeps its own in-memory bindings, and with N replicas behind a load balancer a follow-up turn reaches the replica that holds the binding only about 1/N of the time. Every write is mirrored into a local in-memory copy. Each Redis command is bounded by 200 ms; after an error the store uses the local copy for 5 seconds before trying Redis again, so an outage costs one timeout per window, not one per request.

## Sources of Session ID

The proxy extracts a session ID from the request body in the following priority order:

| Field                           | Location                          |
| ------------------------------- | --------------------------------- |
| `extra_body.litellm_session_id` | nested in `extra_body`            |
| `litellm_session_id`            | top-level field                   |
| `extra_body.chat_id`            | nested in `extra_body`            |
| `extra_body.session_id`         | nested in `extra_body`            |
| `session_id`                    | top-level field                   |
| `user`                          | top-level field (OpenAI standard) |
| `safety_identifier`             | top-level field                   |
| `prompt_cache_key`              | top-level field                   |

The simplest approach — compatible with the standard OpenAI SDK — is to pass `user`:

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[...],
    user="conversation-id-or-user-id",
)
```

For multipart requests the proxy reads `session_id` or `user` from the form fields.

### Header Fallback

If none of the body fields above yield a session ID, the proxy falls back to scanning request
headers whose name matches `^(x-.+-)?session[-_]id$` (case-insensitive), e.g. `Session-Id`,
`Session_Id`, or `X-Codex-Session-Id`. This covers agents/SDKs that attach a session identifier
to a header instead of the JSON body. When multiple matching headers are present, a bare
`Session-Id` takes priority over any `X-...-Session-Id` variant.

Note: a plain `X-Session-Id` header (no segment between `x-` and `session-id`) does **not**
match this pattern by design — use `Session-Id` or add a distinguishing segment (e.g.
`X-Codex-Session-Id`).

## Configuration

### Enable / Disable

Session-sticky routing is **enabled by default**. To disable it:

```yaml
server:
  session_sticky_enabled: false
```

When disabled, no `SessionStore` is created and all routing uses plain round-robin. Behaviour is identical to versions before the feature was introduced.

### TTL

The binding expires after the configured TTL of inactivity. Every turn served by the bound credential refreshes the TTL, so an active conversation keeps its credential affinity. Lookups alone do not extend it. After expiry, the next request starts a new binding via round-robin.

```yaml
server:
  session_sticky_ttl_minutes: 6   # default: 6 minutes
```

Choose a TTL that matches your conversation cadence:

| Use case                      | Recommended TTL                       |
| ----------------------------- | ------------------------------------- |
| Interactive chat (fast turns) | 5–10 minutes (default)                |
| Agentic / slow workflows      | 30–60 minutes                         |
| Batch processing with breaks  | disable sticky or use a very long TTL |

Negative values are rejected at startup.

### Automatic Anthropic Prompt Caching

When routing to an **Anthropic**, **Comet API**, or **Bedrock** credential with an active session, the proxy automatically injects `cache_control: {type: "ephemeral"}` markers into the request body. This converts automatic same-credential affinity into explicit provider-level prompt caching, reducing token costs by up to 90% on long conversations.

**Enabled by default.** To disable:

```yaml
server:
  session_sticky_auto_cache_control: false
```

#### What gets marked

The proxy follows Anthropic's recommended multi-turn caching pattern:

| Position         | What is marked                                            |
| ---------------- | --------------------------------------------------------- |
| System message   | Last content block (stable across all turns)              |
| History boundary | Last content block of the **second-to-last** user message |

```
system: [large context] ← cache_control: ephemeral
user:   turn 1          ← cache_control: ephemeral (history boundary)
asst:   turn 1
user:   current message   (not marked — changes every turn)
```

The current user message is intentionally left unmarked because it changes with every request. Marking the boundary of stable history is sufficient for Anthropic to cache the expensive prefix.

#### Interaction with manual cache_control

If the incoming request **already contains** any `cache_control` field anywhere in its content blocks, auto-injection is skipped entirely. Manual markers always take precedence.

#### Trigger conditions

Auto-injection fires when **all** of the following are true:

- `session_sticky_auto_cache_control: true` (default)
- Credential type is `anthropic`, `cometapi`, or `bedrock`
- A session is active: request has a session ID **or** `previous_response_id` resolved to a known credential

### Full Example

```yaml
server:
  port: 8080
  master_key: "os.environ/MASTER_KEY"
  session_sticky_enabled: true
  session_sticky_ttl_minutes: 10
  session_sticky_auto_cache_control: true   # default — explicit for clarity

credentials:
  - name: "vertex_cred_1"
    type: "vertex-ai"
    credentials_file: "os.environ/VERTEX_KEY_1"
    rpm: 60

  - name: "vertex_cred_2"
    type: "vertex-ai"
    credentials_file: "os.environ/VERTEX_KEY_2"
    rpm: 60

  - name: "vertex_cred_3"
    type: "vertex-ai"
    credentials_file: "os.environ/VERTEX_KEY_3"
    rpm: 60
```

With this config, three Vertex AI credentials share the load in round-robin. Once a session is established on `vertex_cred_2`, all subsequent turns of that conversation go to `vertex_cred_2` (as long as it is available and the TTL has not expired). A new session starts fresh via round-robin.

## Interaction with Responses API

If you use the Responses API (`POST /v1/responses`) with `store: true`, the proxy persists the `CredentialName` alongside the stored response. On the next turn, `previous_response_id` is resolved to that credential before session-sticky or round-robin selection is attempted.

This gives you two independent layers of cache affinity:

1. **Responses API sticky** — direct link to the credential that created the previous response.
2. **Session-sticky** — fallback if the response-level credential is unavailable.

```python
# Turn 1 — no previous_response_id, round-robin selects a credential
r1 = client.responses.create(
    model="gpt-4o",
    input="Hello!",
    store=True,
    user="conv-123",
)

# Turn 2 — proxy prefers the credential from r1 (Responses API sticky)
# then falls back to session-sticky (user="conv-123"), then round-robin
r2 = client.responses.create(
    model="gpt-4o",
    input="Continue our conversation.",
    previous_response_id=r1.id,
    store=True,
    user="conv-123",
)
```

## Edge Cases

| Scenario                             | Behaviour                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------- |
| `session_sticky_enabled: false`      | Plain round-robin, no binding stored                                                      |
| No `user` / `session_id` in request  | No binding attempted, plain round-robin                                                   |
| First request for a session          | Round-robin selects credential; binding written only on success                           |
| Bound credential is banned           | Fallback to the cascade; binding moves to the credential that served                      |
| Bound credential is busy (RPM/TPM)   | Fallback to the cascade; binding kept, next turn returns to it                            |
| Bound credential removed from config | `NextSpecific` returns not-found; fallback to round-robin                                 |
| Request fails (5xx)                  | Binding is cleared by `defer` in `ProxyRequest`                                           |
| Client disconnects (499)             | Binding kept                                                                              |
| Streaming failure mid-stream         | `streamCompleted = false` → binding cleared                                               |
| Same-type retry succeeds on cred B   | Binding moves to B if B is in the model's best tier, else any existing binding is cleared |
| Fallback proxy succeeds              | Same as retry                                                                             |
| TTL expired between turns            | Binding treated as absent; next round-robin result starts a new binding                   |
| One `sessionID`, multiple models     | Independent bindings per `{sessionID, modelID}` pair                                      |

## Performance Notes

- With Redis, a lookup is one `GET` bounded by 200 ms (local copy on error, 5 s back-off); without Redis, `SessionStore` is an in-memory mutex-protected map — lookup is O(1) and sub-microsecond.
- `NextSpecific` acquires a **read lock** on the balancer — it does not block concurrent round-robin selections.
- The background cleanup goroutine runs every 2 minutes and removes expired entries.
- At very high session counts (millions of concurrent sessions) consider using a Redis-backed store for the `ResponseStore` and monitoring memory usage of `SessionStore`.
