# Credential routing profiles

Use a routing profile when selected models must stay inside an approved provider-account pool. Bind its opaque ID in the authenticated virtual key's database metadata, or in its team's or organization's metadata. The provider-account names stay in server configuration, not in client-readable metadata.

A profile is an additional eligibility constraint. It does not grant model access, change provider content policies, billing, budgets, shared RPM/TPM, bans, priorities or weights. An organization-wide restriction still applies to all its keys; a key binding can narrow access further.

## Configuration and binding

This fragment uses fictional names; credentials and routes must already exist:

```yaml
litellm_db:
  enabled: true
  is_required: true

routing_profiles:
  - id: pool-01
    rules:
      - models: [vendor/model-a, vendor/model-b]
        credential_allowlist: [provider-account-a, provider-account-b]
  - id: pool-02
    rules:
      - models: [vendor/model-a]
        credential_denylist: [provider-account-b]
```

An operator assigns the profile by merging this field into the key's existing database metadata (not the API request body):

```json
{"air_routing_profiles": ["pool-01"]}
```

The same field is supported in team and organization metadata. Bindings at all three levels are combined; matching rules **intersect**, never replace a parent restriction. A new key inherits team/org rules only when it is assigned to that team/org. A key-specific binding must be copied explicitly during rotation. This feature does not infer parent/child keys hidden behind an application's shared key, or restrict all future keys of a user automatically.

No key, key fingerprint, per-client environment variable, database migration or new AIR management endpoint is required. Profile IDs contain 1–64 ASCII letters, digits, dots, underscores or hyphens; use neutral IDs that do not disclose provider or customer names. Bindings are arrays of at most 32 IDs per level. An absent field or empty array adds no restriction at that level. Null, wrong types, invalid IDs and unknown profiles fail closed for the request, including when the server has no profiles configured. Removing a profile from configuration before removing its bindings therefore causes errors rather than reopening general routing.

An allowlist limits a rule to the named accounts; a denylist subtracts accounts. Both may be used together and denial wins. At least one nonempty list is required. A deny-only rule is compiled against the configured direct credentials at startup. A configuration reload/restart is required to include newly configured credentials. Matching disjoint profiles can deliberately leave an empty pool.

## Management security — required before activation

Database metadata is trusted authorization data **only if every management path protects it**. AIR cannot distinguish an operator's database write from a customer's write. Before assigning live bindings:

- Reserve `air_routing_profiles` for operators on key creation, update, clone, rotation, imports and any team/org management APIs. Protect deletion and whole-metadata replacement as well as setting a value.
- Preserve the binding when an owner edits unrelated metadata. Concurrent edits must not overwrite a newly assigned operator policy.
- Do not let owners remove/reassign a key's team/org to escape inherited policies. New keys outside the group do not inherit its restrictions; control their issuance if the requirement covers a whole workload.
- Audit all control planes sharing the database, including a separate LiteLLM UI/API. Restrict unprotected management endpoints to operators until they implement equivalent field-level authorization. Blocking one URL does not secure other direct paths.
- Key-info/list responses may expose the neutral profile ID. They must not contain provider-account lists, actual credentials or secret material. AIR's public generation responses and `/v1/models` do not expose the profile definitions.

The router does not provide these management endpoints or rewrite permissions in an external application. The companion control-plane protection must be deployed before live metadata bindings. Do not treat this AIR PR alone as an end-to-end authorization rollout.

## Request flow

1. Authenticate the virtual key and perform existing model and organization access checks.
2. Read bindings from authenticated database metadata; request headers, body metadata, key aliases and session IDs cannot select or replace a profile.
3. Match the requested model and resolved route, including accepted/public aliases, organization mappings and per-credential deployment aliases.
4. Intersect all matching permitted pools with existing scopes and organization restrictions.
5. Use ordinary selection inside the resulting pool, including initial selection, sticky/exact picks, retries, fallback and native Responses WebSocket turns. Recheck at the outbound send boundary.

Unbound keys and unmatched models retain existing routing. No eligible initial route returns HTTP 503; actual rate-limit exhaustion stays HTTP 429. Already-contacted upstream errors follow existing error handling. There is no fallback outside the pool. Invalid bindings generate a generic diagnostic, without metadata values or account names.

Only direct provider credentials may be allowlisted. `type: air` and `type: proxy` perform another selection downstream; they are rejected in allowlists and excluded for every restricted request, including deny-only rules. A direct credential's normal network `proxy_url` is supported. AIR cannot verify which internal accounts a third-party provider chooses behind its API.

Exact model names are supported; wildcards are rejected. Equivalent deployment names on the **same** credential are connected transitively. The same deployment name on unrelated credentials alone does not establish equivalence. Catalog refreshes include the current alias target while retaining startup identities, so temporarily removing a bridge does not reopen access.

## Updates and conversations

Profiles load at startup. Configure all public ingress instances consistently before assigning bindings. Bindings refresh through the existing authentication-cache TTL/invalidation mechanism; this is not instantaneous revocation and does not cancel an already-running response. Native WebSocket turns reauthenticate, subject to the same cache delay. There is no cross-router profile propagation in this version.

A sticky binding outside the pool is ignored; an eligible healthy binding may be reused. HTTP Responses with an opaque `previous_response_id` and no reconstructed history remains on its original permitted account, including retries. Unknown or excluded accounts return HTTP 400 with guidance to resend history. Reconstructed history permits ordinary selection within the pool. Native WebSocket conversation affinity is preserved; a changed policy may require a new connection rather than moving an active conversation.

Automatic description of an image using a hidden helper model is rejected for restricted requests; native image processing remains available. Async job APIs are outside this feature's scope. `/v1/models` listing and billing are unchanged.

## Rollout

1. Deploy and verify management-field protection on every database writer.
2. Deploy identical profile definitions with required database authentication to every serving AIR ingress.
3. Assign an opaque profile ID to one test key using an operator account.
4. Verify matched/unmatched models and other keys, exhaustion, retries, sticky and streaming. Confirm that owners cannot change/delete bindings or move the key out of its team.
5. Wait for or explicitly invalidate authentication caches when changing bindings. Remove bindings before retiring a profile. Removing the last matching binding intentionally restores normal routing.

No live profiles or client bindings are included in this change.
