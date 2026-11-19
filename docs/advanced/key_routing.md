# Per-key model routing policies

Use a key routing policy when selected models of one authenticated virtual key must stay within an explicitly approved provider-account pool. For example, a workload may have a contractual requirement to use two approved accounts while the same key's other models retain ordinary routing.

An organization-wide denylist affects unrelated keys. A routing weight is a preference and does not prevent fallback outside a pool. This policy is an additional eligibility constraint, applied before ordinary balancing; it neither grants model access nor changes provider content policies.

## Configuration

The following is a fragment to add to an existing configuration. All names are placeholders. Both credentials and model routes must already exist.

```yaml
litellm_db:
  enabled: true
  is_required: true

key_routing_policies:
  - id: approved-pool-a
    key_hash: os.environ/CLIENT_A_KEY_SHA256
    rules:
      - models:
          - vendor/model-a
          - vendor/model-b
        credential_allowlist:
          - provider-account-a
          - provider-account-b
```

`CLIENT_A_KEY_SHA256` contains the authenticated virtual key's stored SHA-256 fingerprint (`TokenInfo.Token`), supplied through the deployment's secret mechanism. It is not the raw API key, key alias, user ID or organization ID. Literal fingerprints in YAML are rejected. Do not commit the environment value or include it in support exports. The configuration serializer retains only the environment reference, and validation errors do not print its value.

The policy is owned by the operator. Request headers, body metadata, session IDs and client-editable key aliases cannot select or replace it. Authentication through the database must be enabled and required; the master key remains an operator credential, not a virtual-key policy subject. Key rotation requires updating the binding before using the new key.

Configuration is loaded at startup. An infrastructure repository can supply this section and the protected environment variable through its existing deployment process; this feature does not require a new database table or admin API. Configure every public ingress serving the key consistently. There is no cross-router policy propagation in this version.

## Request flow

1. Authenticate the virtual key and apply the existing model and organization access checks.
2. Resolve the requested model and its accepted aliases or organization mapping.
3. If the key and model match a policy, intersect its allowed credentials with the existing scope and organization restrictions.
4. Run ordinary priority/weight selection, shared credential/model rate limits and ban checks inside that pool.
5. Apply the same constraint to exact/sticky picks, retries, fallback tiers and each native Responses WebSocket turn. Check it again at the outbound send boundary.

Unmatched keys and models retain their existing routing. A newly added credential does not enter an existing allowlist automatically. An allowed credential still needs to serve the requested model; the policy does not substitute a different model or override pricing, budgets, access checks, bans or limits. Requests share the existing RPM/TPM counters, rather than receiving fresh quota per policy.

Exact model names and their resolved aliases are supported; wildcards are rejected. This includes alternate route names mapping to the same upstream model on the same credential, including per-credential deployment names. Equivalence follows connected aliases; an identical deployment name on two different credentials alone does not establish equivalence. Both the requested public identity and an organization's mapped route participate in matching.

Rules that match the same request intersect. Disjoint allowlists deliberately leave no eligible route. Catalog refreshes also match a rule's current alias target, while retaining its startup identities and equivalent deployment aliases until restart to avoid broadening access when a catalog entry disappears. Consequently, renaming or temporarily removing a configured alias does not disable its restriction; changing the policy itself is an explicit operator action.

## Unavailable routes and conversations

- No eligible permitted route at initial selection: HTTP 503. Actual rate-limit exhaustion remains HTTP 429. Failure after contacting permitted upstreams follows the existing error handling. There is no fallback into the unrestricted pool.
- A sticky binding outside the pool is ignored. A permitted, healthy binding may be reused.
- For HTTP Responses, an opaque `previous_response_id` without reconstructed history is confined to its original permitted account, including retries. An unknown or excluded account produces HTTP 400 with guidance to resend the conversation. If stored history is available and reconstructed successfully, ordinary selection inside the permitted pool can resume it.
- Native Responses WebSocket connections retain their existing conversation-affinity rules and recheck eligibility on subsequent turns. An existing connection is not permission to bypass the policy or a ban.

## Deliberate boundaries

Only direct provider credentials can be allowlisted. `type: air` and `type: proxy` credentials perform another selection downstream and are rejected at startup and excluded at runtime for restricted requests. A direct credential may still use its existing network `proxy_url`; that does not perform a second model/account selection. AIR cannot attest to routing inside a third-party provider behind a direct API endpoint.

Automatic vision-description helper calls are rejected for matched requests, because they invoke a different model outside the requested model's policy. Native vision requests continue normally. Model listing is unchanged; appearing in `/v1/models` does not promise a currently eligible route. This feature is intended for synchronous model inference through the common request orchestrator, including Chat Completions, Messages, Responses HTTP and native Responses WebSocket; it is not a new authorization system for separately managed asynchronous jobs.

The feature does not change provider filters, refusal handling, key aliases, public model names, organization billing or global cascade priorities. It adds no prompt/response logging. A matched request emits the policy ID and public model in the existing operational logger; choose non-sensitive policy IDs.

## Validation and rollout

Startup rejects missing/malformed fingerprint values, duplicate policy IDs or key bindings, empty rules, unknown credentials, proxy credentials, unroutable model names and models with no permitted direct route. Limits are 1,024 policies, 128 rules per policy, and 128 models/credentials per rule.

Deploy the code without policies first if desired: an absent section preserves existing routing. Then review a minimal policy, inject the protected binding and roll out the same configuration to the ingresses serving that key. Verify the selected account on successful requests and verify that exhausting the pool produces an error rather than a forbidden fallback. Check unaffected keys and models as well. Removing a policy restores ordinary routing, so treat rollback of the configuration as an explicit access decision.

Tests use local mock providers and synthetic key fingerprints. They exercise matching and aliases, overlapping policies, unchanged unrelated requests, sticky selection, retries and pool exhaustion, shared rate limits, organization exclusions, bans, HTTP Responses, streaming, native WebSocket, opaque continuations and internal vision calls. No live provider credentials or customer traffic are used.
