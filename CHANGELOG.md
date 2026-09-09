# Changelog

## Unreleased

### Breaking changes

- `GET /v1/models` now uses the same admission rules as inference for the token.
  Temporary bans and current RPM/TPM exhaustion do not remove models from the listing.
  Ordinary listings still respect `client_model_ids`, credential/model visibility,
  organization policy, and billing eligibility; they are not the union of every
  configured model and alias. Master-key listings can include internal route IDs
  that ordinary keys cannot list or invoke.

- `public_model_alias` was removed. Move its entries to `accepted_model_alias`,
  whose active, admitted aliases are now both callable and listed. Previously,
  `public_model_alias` entries were callable and listed, while
  `accepted_model_alias` entries were callable but hidden from discovery.

- Model routing now requires an explicit static or discovered mapping between
  the requested route and a credential. An empty topology no longer permits
  arbitrary requests: unknown models return `404`, and the model list is empty.
  Undiscovered proxy credentials do not match arbitrary models.

- When spend tracking is enabled, a model with no billable credential route is
  rejected during admission with `404` and omitted from the model list. An
  organization model missing its required exact organization tariff also returns
  `404` instead of `503`. A missing billing price detected after credential
  selection, including on retry/fallback, still returns `503` before forwarding.
  ACL rejection and dangling configured organizations continue to return `403`.

- When all serving providers are known to be denied by organization policy,
  the model is omitted from discovery and inference returns `404` at admission,
  instead of attempting routing and eventually returning `429`. For downstream
  providers, this depends on the health metadata described below.

- The `include_model_access_groups` model-listing extension was removed.
  The query parameter no longer produces synthetic provider/model entries.

- A successful DB price sync now replaces the DB price snapshot. Rows deleted
  from the database disappear on the next sync; an empty snapshot clears all
  previous DB overrides. File and DB snapshots are independent, and DB prices
  take precedence for matching keys. File reloads preserve DB overrides.
  Removing a DB override falls back to the file price, if one exists; it does
  not necessarily make the model unpriced.

### Fixes

- Model ACLs respect direct-route and accepted-alias precedence. Shadowed global
  aliases cannot authorize a different route or supply its provider wildcard identity.
- Chained provider scopes no longer multiply aggregate alternatives into each
  leaf path, preventing false `404` responses with organization denylists and
  many differently scoped providers. Local scope restrictions remain enforced.
- Model resolution now shares alias precedence across ordinary, organization,
  and master-key requests. Registered direct routes take precedence over
  conflicting `model_alias` entries, preventing rejection or redirection of an
  otherwise valid direct route.
- Multipart requests such as `/v1/images/edits` preserve the logical routing
  model when forwarded through proxy credentials instead of sending a
  direct-provider model name.
- Model listing accounts for credential-specific real-model prices. Inference
  excludes routes without a resolvable billing price while keeping billable
  alternatives eligible, including fallback credentials.
- Proxy/AIR retries and fallback proxy hops recheck billing prices and update
  the selected credential's real-model and billing metadata. Organization
  requests retain their exact organization tariff across retries.

### Compatibility and rollout notes

- Billing-price resolution retains its public-ID, route-ID, then effective
  real-model-ID precedence. A public billing price or an exact organization
  tariff is sufficient; there is no separate requirement for a provider base
  price. Credential-specific pricing remains supported. Without spend tracking,
  ordinary requests do not require a price; explicit zero-price rows remain valid.
- With `strict_all_team_models_acl: false`, matching inference does not enforce
  token/team/user model allowlists in discovery. Enable strict ACL enforcement
  if those allowlists must restrict both listing and inference. Credential/model
  visibility and organization policies still apply independently.
- Per-model `/health` entries now include `provider_routes`: reachable leaf
  credential names and their scope expressions. Parents use cached health data
  to evaluate organization denylists through intermediate routers, without
  additional network calls for this eligibility check or transient rate-limit
  filtering.
- Complete multi-hop filtering requires intermediate routers to propagate this
  metadata and their health snapshots to refresh. Missing metadata or legacy
  relays with unknown leaf providers retain the previous behavior: a model can
  still be listed and rejected downstream. Existing denylist enforcement at the
  provider is unchanged; this fixes discovery and early admission, not a
  denylist bypass.

### Go API changes

- `router.New` no longer accepts a separate model manager. Model listing is
  obtained through `Proxy.ListModelsForToken`.
- `ModelPriceRegistry.Update` and `MergeDB` were replaced by `ReplaceFilePrices`
  and `ReplaceDBPrices`. `GetPrice` was removed; use `GetPriceAny`, which returns
  both the matched candidate ID and the price.
- `Manager.SetPublicModelAliases` and `ResolvePublicModelAlias` were replaced
  by `SetAcceptedModelAliases` and `ResolveAcceptedModelAlias`.
  `Config.AcceptedModelAlias` is now `AcceptedModelAliases`; the YAML key remains
  `accepted_model_alias`.
- `Manager.ResolveModel` is the shared resolver, and `OrganizationModelResolution`
  was renamed to `ModelResolution`. `Manager.IsEnabled` and the
  `GetAllModelsWithAccessGroups*` methods were removed.
- The proxy's listing-specific price/ACL helpers were removed in favor of
  `ListModelsForToken` and shared inference admission.

### Migration checklist

1. Move `public_model_alias` entries into `accepted_model_alias`.
2. Audit the newly advertised accepted aliases. Removing an alias removes its
   mapping and can break inference callers; it is not merely a visibility change.
   Use `client_model_ids` to define the ordinary client-facing boundary, and do
   not assume master-key listings represent what ordinary keys can see.
3. Ensure every callable model has an explicit static or discovered
   credential mapping.
4. When spend tracking is enabled, ensure every intended request has a
   resolvable billing price; use an explicit zero-price row for free models.
5. Remove `include_model_access_groups` from model-listing requests; synthetic
   provider/model entries are no longer available.
6. Enable `strict_all_team_models_acl` if token/team/user model allowlists must
   be enforced for both listing and inference.
7. Update intermediate routers and allow health snapshots to refresh before
   relying on complete multi-hop denylist filtering in discovery.
8. Update Go integrations for the constructor and method changes above.
