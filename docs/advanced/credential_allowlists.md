# Credential allowlists for LiteLLM identities

An operator can restrict all models used by a LiteLLM key, team or organization to a set of direct credentials. In AIR's server configuration, assign an optional `routing_id` to each credential that will be selectable by this policy.

```yaml
credentials:
  - name: example-private-account
    routing_id: route-01
    type: openai
    base_url: https://example.com/v1
    api_key: os.environ/EXAMPLE_API_KEY
```

Assign IDs once to the credentials you intend to reference. Use `route-` followed by 1–12 digits. IDs must be unique in each AIR configuration and should consistently identify the same account across instances. Add the field to an existing credential entry when adapting a deployment.

In the server-managed LiteLLM metadata for the identity, place the IDs in `allowed_cred`.

```json
{
  "allowed_cred": ["route-01", "route-02"]
}
```

The metadata contains neutral IDs. The association with provider account names lives in server configuration. The routing policy is derived from the authenticated identity. The request body's metadata and headers are independent of that policy.

- An absent, empty or null list leaves that level unrestricted. Nonempty lists intersect across key, team and organization. Malformed values allow zero routes.
- An ID with no matching credential contributes no candidate. An empty effective set or a set with no candidate for a model fails without sending to another account. The policy covers existing and future models subject to existing model permissions.
- Priorities, weights, quotas, bans and organization credential deny rules apply to the permitted set. Sticky bindings, retries and WebSocket selections use the same restriction.
- Direct credentials can use a network `proxy_url`. A restricted request cannot use an AIR or generic proxy credential because the positive restriction is local to this router.
- Restricted identities cannot start asynchronous video jobs because those jobs use a global provider client outside the credential picker. Existing job access retains its existing ownership checks. Standalone non-model features use their own authorization.
- Health and trace keep their existing visibility rules. They do not filter named account diagnostics by `allowed_cred`.

Before setting these fields, deploy a compatible version to every ingress that accepts the affected key, team or organization. Protect `allowed_cred` on every LiteLLM and admin management API and database writer. Only operators may assign, replace or remove it. Preserve it when a customer replaces their ordinary metadata. Neutral IDs protect account names in metadata reads. Write protection is required to prevent a customer from removing an assigned restriction. Apply normal authentication cache invalidation or wait for expiry when editing a binding.
