# Model pools and strict key isolation

Deployment `pools` tags and `air_routing_profiles` retain their existing soft
selection behavior. Without an explicit model selector, unmatched models/pools
continue through ordinary routing. This is a preference, not a security boundary.

For strict isolation, an operator adds both fields to the authenticated key's
database metadata:

```json
{
  "air_routing_profiles": ["pool-01"],
  "air_routing_models": ["vendor/model-a", "vendor/model-b"]
}
```

The selector records which models must remain restricted independently of current
pool membership. Without it, removing the last deployment tag is indistinguishable
from a model that was never restricted. No separate profile registry is required.

Configure membership on existing deployments:

```yaml
models:
  - name: vendor/model-a
    credential: account-a
    pools: [pool-01]
  - name: vendor/model-a
    credential: account-b
    pools: [pool-01]
```

These are fictional names. Define each selected model on its intended accounts.
Strict requests require explicit direct credentials: global model tags do not
authorize all accounts. AIR/proxy credentials cannot carry this restriction to
another router and are excluded; ordinary network `proxy_url` is supported.
Pool tags alone neither reserve deployments for a key nor add RPM/TPM capacity.

## Behavior

- The selector is key-specific, taken only from trusted authentication metadata.
  Client request metadata/headers cannot set or remove it. Model selectors on
  teams/organizations are unsupported and rejected.
- Only selected models are restricted. Other models of the same key retain their
  usual routing; the key's profiles are not also applied to them as preferences.
  Existing team/org model ACLs, scopes, denylist and pool preferences still apply.
  Parent pool preferences cannot widen a strict key's permitted set.
- Exact model names only. Explicit aliases, organization mappings, and separately
  registered names for the same upstream model on the same credential are resolved
  before matching. A shared wire name on different accounts alone is not enough.
- Pools listed on a key are alternatives, intersected with existing visibility and
  availability. No permitted deployment means no route, including after a catalog
  refresh or restart removes the model's pool tags.
- Missing/unknown pools fail closed for selected models; unrelated models remain
  usable. Malformed selectors/profile lists or unknown selector model names fail
  closed for the key because their intended scope cannot be interpreted safely.
  Missing `air_routing_models` preserves legacy behavior.
- Initial choice, sticky/exact selection, retries/fallback and native Responses
  WebSocket turns enforce the restriction, with an additional guard before send.
  Shared quotas, bans, weights and billing identities retain their behavior.
- Invalid/unavailable policy or no permitted initial route returns a generic 503;
  rate-limit exhaustion retains 429. Existing upstream error handling is unchanged.
  Credential/pool names are not echoed in policy error responses.
- Opaque `previous_response_id` without recovered history must stay on its original
  permitted account, otherwise 400. Reconstructed history may use another permitted
  account. Hidden vision-description calls are rejected for restricted requests;
  native vision is unchanged.

The catalog index is rebuilt atomically when definitions or aliases change; there
is no per-request full catalog scan. Known identity links are retained for the
process lifetime so dropping an alias bridge on refresh cannot reopen it. Reusing
old model IDs can conservatively preserve restrictions until restart. Keep alias
definitions consistent across instances/restarts: AIR cannot infer a relationship
removed from all configuration. Pool membership is current, not historical.
In-flight requests keep their snapshot.

## Activation and rollback

Protect both metadata fields on every writer before activation, including separate
LiteLLM management APIs. Only operators may assign/change/remove them. Whole-object
and concurrent edits must preserve this boundary; hiding a UI field is insufficient.
Use public model IDs and neutral pool IDs, never account names or secrets. New
independent keys do not inherit another key's policy.

Deploy compatible AIR to every ingress with required database authentication, then
configure pool tags, then bind one approved key. Verify errors, latency, spend and
cache reuse. Edits obey existing auth-cache TTL/invalidation. Membership changes
can change cache reuse; cache accounting is unchanged.

Do not roll back to a version that ignores active model selectors: stop/isolate
affected keys first. Removing the selector explicitly restores soft selection;
removing both fields restores ordinary routing. These are operator decisions.

No live bindings, infrastructure rollout, prompt logging, model-discovery filtering
or async-job routing are added. Providers' internal account routing is outside AIR.
