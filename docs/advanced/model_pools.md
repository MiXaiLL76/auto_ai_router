# Model pools and strict key isolation

Deployment `pools` tags and `air_routing_profiles` are introduced by this change;
before it neither field existed. Profiles alone are soft selection: a model outside
the selected pools continues through ordinary routing. This is a preference, not a
security boundary.

A pool ID this key names that the catalog does not know is a typo in the operator's
own metadata, so it fails closed (503) instead of quietly opening the general
cascade. A field of the wrong shape (a number, object, or a list with a non-string)
fails the same way. The same mistake inherited from a team or organization does not
fail: one bad team profile would otherwise break every member's key.

An unknown pool in soft selection fails the whole key, because there is no way to
tell which models the operator meant. Strict selection scopes the same failure to
the models its selector names; other models of the key keep routing.

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

Models synced from LiteLLM (`litellm_db.load_db_models`) carry their tags in
`model_info.pools` of the deployment's row, since the table has no dedicated
column. A deployment without that entry belongs to no pool.

## Behavior

- A selector on a team or organization is rejected, not ignored: it is never
  inherited, and silently ignoring it would leave an operator believing a whole
  team is isolated when it is not. The rejection is logged with its scope.
- The selector is key-specific, taken only from trusted authentication metadata.
  Client request metadata/headers cannot set or remove it. Model selectors on
  teams/organizations are unsupported and rejected.
- Only selected models are restricted. Other models of the same key retain their
  usual routing; the key's profiles are not also applied to them as preferences.
  Existing team/org model ACLs, scopes, denylist and pool preferences still apply.
  Parent pool preferences cannot widen a strict key's permitted set.
- Exact model names only. Explicit aliases, organization mappings, and separately
  registered names for the same upstream model on the same credential are resolved
  before matching — in soft and strict selection alike, so an alias cannot step
  around a pool. A shared wire name on different accounts alone is not enough.
- Pools listed on a key are alternatives, intersected with existing visibility and
  availability. No permitted deployment means no route, including after a catalog
  refresh or restart removes the model's pool tags.
- Missing/unknown pools fail closed for selected models; a key's own unknown profile
  fails closed even in soft selection. Unrelated models remain usable. A malformed
  selector list or profile list of the wrong shape, and an unknown selected model
  name, fail closed for the key because their intended scope cannot be interpreted
  safely. Missing `air_routing_models` preserves legacy behavior.
- A selector with no profile list at all is the one exception: it names no pool,
  so there is nothing to restrict to. It degrades to ordinary routing and logs a
  warning rather than failing the key. An operator who means to isolate must set
  both fields; a selector alone is treated as a mistake, not as a lockout.
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

The catalog index is rebuilt atomically when definitions or aliases change, and a
per-name reverse index answers a request without scanning the whole catalog. Only
identity links whose names no current alias mapping mentions are retained for the
process lifetime, so dropping an alias bridge cannot reopen a restriction while a
re-pointed alias cannot merge two live models. Reusing an old model ID can
conservatively preserve a restriction until restart. Keep alias definitions
consistent across instances/restarts: AIR cannot infer a relationship removed from
all configuration. Pool membership is current, not historical. In-flight requests
keep their snapshot.

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
