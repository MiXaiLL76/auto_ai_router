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

`priority` is the single selection-order axis, used for both the initial request and
every retry. Credentials are bucketed by `priority` value (ascending); the balancer runs
weighted round-robin inside the lowest-numbered tier that still has a live member, and
only cascades to the next tier when every member of the current one is banned or
rate-limited.

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
weighted pool described above. `priority: 999` (`FallbackPriorityGroup`) is the
last-resort group.

The retired `is_fallback: true` and `fallback_priority: N` YAML keys are still accepted
as deprecated input aliases: `is_fallback: true` folds to `priority: 999`,
`fallback_priority: N` folds to `priority: N`. The router logs a warning; `is_fallback:
true` combined with a *lower* explicit `priority` is rejected as contradictory.

For a `proxy`/`air` credential, the per-model priority learned from the upstream's own
`/health` (its upstream credentials' `priority` values) takes precedence over the static
`priority` set here, so a proxy credential's tier reflects what the node it proxies to is
actually configured with. Models that the upstream serves **only** from its last-resort
group (`priority: 999` / `is_fallback`) are no longer hidden from a non-`is_fallback`
proxy credential: they are discovered and placed in this router's local last-resort tier.
When the model checker is disabled there is no learned priority, so a proxy credential
fronting an all-last-resort upstream node should itself set `priority: 999` (or
`is_fallback: true`) to keep those models out of the primary pool.

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

## Retry Order

After the initially selected credential returns a retryable error (`429`, `5xx`, auth
error), the router continues the **same priority cascade** over the credentials it has not
tried yet: it walks the ascending priority tiers, weighted round-robin within a tier, and
crosses provider types freely. When the next credential has a credential-specific real
model mapping, the router re-resolves the model before sending the retry request, so an
Anthropic model alias can safely move onto a Bedrock credential.

`max_provider_retries` bounds the same-request retry loop; a proxy-forward path then has
an extended phase bounded by `max_fallback_attempts` that keeps walking the cascade
through any remaining proxy/AIR credential. There is no separate retry knob per
credential — order is entirely `priority`.

> **Deprecated:** `fallback_priority: N` and `is_fallback: true` are still accepted as
> input aliases. `is_fallback: true` becomes `priority: 999`; `fallback_priority: N`
> becomes `priority: N` (which now also groups the *initial* selection into a hard tier,
> not only the retry order). The router logs a warning; migrate to `priority`.

## AIR Chain Last-Resort

When using chained routers (e.g. router01 → router02 in tier 0, router03 at `priority:
999`), the last-resort credential works across the chain:

```
router01 receives request
  └─► router02 (tier 0 AIR) → router02 returns 429/5xx
      └─► router01 detects retryable error, marks router02 tried
          └─► cascade continues → router03 (priority 999 AIR) → success
```

```yaml
credentials:
  - name: "router02"
    type: "air"
    base_url: "https://router02.example.com"

  - name: "router03"
    type: "air"
    base_url: "https://router03.example.com"
    priority: 999
```

Models that router03's upstream serves only from its own last-resort tier are still
discovered by router01 and placed in router01's local last-resort tier — an all-last-resort
upstream node is not hidden. (With the model checker disabled there is no learned tier, so
give the fronting credential `priority: 999` itself.)
