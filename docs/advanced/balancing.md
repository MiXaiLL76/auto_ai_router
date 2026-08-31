# Load Balancing

## Round-Robin

Auto AI Router distributes requests across credentials using round-robin balancing. When multiple credentials support the same model, each request goes to the next available credential in rotation.

### Example

With 4 Vertex AI credentials configured for `gemini-2.5-flash`:

```
Request 1 → vertex_cred_1
Request 2 → vertex_cred_2
Request 3 → vertex_cred_3
Request 4 → vertex_cred_4
Request 5 → vertex_cred_1  (cycle repeats)
```

Credentials that are rate-limited or banned are skipped automatically.

## Weighted Round-Robin

By default every credential has a `weight` of `1`, so traffic is split evenly. Set a higher
`weight` to send a proportionally larger share of requests to a credential. The router uses
smooth weighted round-robin (the nginx algorithm): requests are handed out proportionally to
the weights but spread evenly over time, not in bursts.

With weights `100` and `1`, roughly 100 out of every 101 requests go to the first credential
and the rest are sprinkled across the others:

```
weights: ours=100, azure=1
... → ours (×100, interleaved) ... → azure (×1) ...  (per 101-request cycle)
```

Weight can be set per credential (the default for all of its models) and overridden per model,
exactly like `rpm`. Resolution order is: model-level `weight` → credential `weight` → `1`.

```yaml
credentials:
  - name: "ours"
    type: "openai"
    api_key: "os.environ/OUR_KEY"
    base_url: "https://our-endpoint.example.com"
    rpm: 5000
    weight: 100            # default weight for every model on this credential

  - name: "azure"
    type: "openai"
    api_key: "os.environ/AZURE_KEY"
    base_url: "https://azure.example.com"
    rpm: 5000
    # weight omitted → 1

models:
  - name: "gpt-5"
    credential: ours
    weight: 200            # per-model override: push gpt-5 harder to "ours"
  - name: "gpt-5"
    credential: azure
```

Notes:

- **Weight does not bypass limits.** When the high-weight credential hits its `rpm`/`tpm` or
  is banned by fail2ban, it is skipped and the request goes to the next live
  credential — the same failover behavior as plain round-robin.
- **No burst after recovery.** A banned credential does not accumulate weight while it is down,
  so it resumes its normal share on recovery instead of receiving a backlog of requests.
- **Equal weights behave exactly like plain round-robin.** Models where all candidates share the
  same weight (e.g. a model you give no special weight to) keep the default even rotation.

## Multiple Credentials per Model

Configure multiple credentials for the same model to multiply your effective rate limits:

```yaml
credentials:
  - name: "openai_1"
    type: "openai"
    api_key: "os.environ/OPENAI_KEY_1"
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000

  - name: "openai_2"
    type: "openai"
    api_key: "os.environ/OPENAI_KEY_2"
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000

models:
  - name: "gpt-4o"
    credential: openai_1
    rpm: 100
    tpm: 50000
  - name: "gpt-4o"
    credential: openai_2
    rpm: 100
    tpm: 50000
```

This gives you an effective 200 RPM for `gpt-4o`.

## Primary Priority Groups

`priority` (distinct from `fallback_priority` below) groups **primary** credentials into
tiers for the *initial* request selection. Credentials are bucketed by `priority` value
(ascending); the balancer runs weighted round-robin inside the lowest-numbered tier that
still has a live member, and only cascades to the next tier when every member of the
current one is banned or rate-limited.

```yaml
credentials:
  - name: "cheap-a"
    type: "openai"
    api_key: "os.environ/CHEAP_A"
    base_url: "https://a.example.com"
    rpm: 100
    priority: 1        # tier 1 — tried first

  - name: "cheap-b"
    type: "openai"
    api_key: "os.environ/CHEAP_B"
    base_url: "https://b.example.com"
    rpm: 100
    priority: 1        # tier 1 — shares traffic with cheap-a via weighted round-robin

  - name: "expensive"
    type: "openai"
    api_key: "os.environ/EXPENSIVE"
    base_url: "https://c.example.com"
    rpm: 1000
    priority: 2        # tier 2 — only used while both tier-1 credentials are down
```

Credentials that omit `priority` (or set it to `0`) all share the default tier `0`, which
is tried first — so a config that never sets `priority` behaves exactly like the flat
weighted pool described above. `priority` and `fallback_priority` are mutually exclusive
on a single credential, and `is_fallback: true` credentials cannot set `priority`.

For a `proxy`/`air` credential, the per-model priority learned from the upstream's own
`/health` (its upstream credentials' `priority` values) takes precedence over the static
`priority` set here, so a proxy credential's tier reflects what the node it proxies to is
actually configured with.

When one upstream node serves the same model from several priority groups behind a single
proxy credential, that credential **expands into one local candidate per tier**. Each tier
becomes its own primary priority group (using the upstream's priority values), with the
summed RPM/TPM capacity and summed weight of the upstream credentials in that group, and
its own *cumulative* local rate-limit gate: tier 1's gate is tier 1's capacity, tier 2's
gate is tier 1 + tier 2, and so on. A tier the upstream reports **banned** contributes no
capacity to that cumulative gate, and its own per-tier usage counter (from `/health`) is
also checked, so a cheap tier that has recovered upstream is not held closed by load the
upstream is serving from a pricier tier.

Consequences:

- Selection cascades **immediately** off this router's own contribution — once the
  requests this router has sent to the proxy credential reach tier 1's cumulative
  capacity, tier 1 stops being offered and a genuinely lower-priority alternative (this
  router's own tier-2/3 credentials) is preferred over the proxy credential's tier-2
  group. No `/health` poll is needed for that step.
- Only the *cross-fleet* part of the signal — other routers also filling the same
  upstream tier — still arrives via the 30s `/health` poll, so a tier can briefly look
  live here while it is actually full upstream.
- The tier breakdown is re-published on this router's own `/health`, so it survives
  additional chain hops (a router fronting this one re-expands it).
- A single-group upstream (the common case) is unaffected — no expansion, one candidate,
  the scalar `priority` above. The one exception: if that single group is **banned**
  upstream, the ban is published as a one-tier breakdown so this router (and the next one
  in the chain) drops the credential — the scalar `priority` number alone carries no ban
  state.

> `fallback_priority` is **not** folded into primary-tier grouping — it only affects retry
> order (next section). A credential that sets only `fallback_priority` stays in primary
> tier `0`.

## Fallback Priority

Primary credentials (non-fallback) are used for the initial request. By default provider retry
stays within the same provider type: if an OpenAI credential returns `429` or `5xx`, the router
tries another OpenAI credential for the same model.

Set `fallback_priority` when the retry order must be explicit and may cross provider types.
Lower numbers are tried first after the initially selected credential returns a retryable error.
The field is applied to regular primary credentials; `is_fallback: true` credentials stay reserved
for the fallback phase and cannot set `fallback_priority`.

```yaml
credentials:
  - name: "primary-anthropic"
    type: "anthropic"
    api_key: "os.environ/PRIMARY_ANTHROPIC_KEY"
    base_url: "https://anthropic-primary.example.com"
    rpm: 400
    fallback_priority: 10

  - name: "backup-anthropic"
    type: "anthropic"
    api_key: "os.environ/BACKUP_ANTHROPIC_KEY"
    base_url: "https://anthropic-backup.example.com"
    rpm: 500
    fallback_priority: 20

  - name: "bedrock-reserve"
    type: "bedrock"
    api_key: "os.environ/BEDROCK_RESERVE_KEY"
    base_url: "https://bedrock-reserve.example.com"
    rpm: 1000
    fallback_priority: 30
```

With this configuration, if `primary-anthropic` returns a retryable error for `claude`, the router
tries `backup-anthropic` next. If `backup-anthropic` is also unavailable, it tries
`bedrock-reserve`. When the next credential has a credential-specific real model mapping, the
router re-resolves the model before sending the retry request, so an Anthropic model alias can
safely move to a Bedrock credential.

If `fallback_priority` is omitted or set to `0`, the old same-type retry behavior is preserved
when that credential starts the retry chain. When a retry chain starts from a credential with
`fallback_priority > 0`, the router tries all configured priority tiers first, then continues
with regular credentials that do not set `fallback_priority`. Fallback credentials
(`is_fallback: true`) are still used only by the fallback mechanism. See
[Proxy — Fallback Behavior](../providers/proxy.md#fallback-behavior) for details.

## AIR Chain Fallback

When using chained routers (e.g. router01 → router02 as primary, router03 as fallback), fallback works across the chain:

```
router01 receives request
  └─► router02 (primary AIR) → router02 returns 429/5xx
      └─► router01 detects retryable error
          └─► router03 (fallback AIR) → success
```

router01 marks router02 as "tried" immediately on the first attempt, so same-type retries
never re-select router02. After all primary AIR credentials are exhausted, `TryFallbackProxy`
selects the next `is_fallback: true` credential (router03).

```yaml
credentials:
  - name: "router02"
    type: "air"
    base_url: "https://router02.example.com"
    is_fallback: false

  - name: "router03"
    type: "air"
    base_url: "https://router03.example.com"
    is_fallback: true
```
