# Retries and Request Cleanup

When an upstream answers with an error, the router can replay the request on the next credential that serves the same model, and after those run out, on the fallback chain. That fixes errors tied to one credential: a rate limit, an outage, a revoked key. It does not fix an error caused by the request itself. Replaying a bad request turns one client mistake into one failed upstream call per credential.

This page covers two features that keep such requests from multiplying:

- the `retry` block decides which upstream errors are replayed;
- request cleanup fixes common client mistakes before the request leaves the router.

## Retry policy

```yaml
server:
  max_provider_retries: 2      # extra same-type credentials per request (default 2 = 3 attempts)
  max_fallback_attempts: 5     # fallback hops per request chain (default 5)

retry:
  status_codes: [400, 401, 402, 403, 404, 429, "5xx"]   # the default
  non_retryable_markers: []
  bad_request_markers: []
  provider_overrides:
    openai:
      status_codes: [429, "5xx"]
  credential_overrides:
    reseller-pool-1:
      status_codes: [429, "5xx"]
```

The router replays a response only if both of these hold:

1. its status code is in the **effective status code set** of the credential that returned it;
2. its body contains none of the **non-retryable markers**. A `400` must also contain none of the **bad-request markers**.

The client gets every other response unchanged, straight from the first credential that returned it.

### Parameters

| Parameter               | Type          | Default                                 | Description                                                                                                                  |
| ----------------------- | ------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `status_codes`          | list          | `[400, 401, 402, 403, 404, 429, "5xx"]` | Status codes that are retried. Each entry is an exact code (`429`) or a class (`"4xx"`, `"5xx"`). Empty or omitted = default |
| `non_retryable_markers` | list (string) | content-policy texts                    | Body substrings (case-insensitive) that stop the retry for **any** status. Added to the built-in list                        |
| `bad_request_markers`   | list (string) | known "the request is wrong" texts      | Body substrings (case-insensitive) that stop the retry of a **400**. Added to the built-in list                              |
| `provider_overrides`    | map           | `vllm`: `status_codes` without `400`    | Replaces `status_codes` for every credential of a provider type (`vllm`, `openai`, `anthropic`, ...)                         |
| `credential_overrides`  | map           | —                                       | Replaces `status_codes` for one credential, keyed by its `name`                                                              |

### Precedence

The effective status code set of a credential is the first one that exists:

1. its `credential_overrides` entry;
2. the `provider_overrides` entry of its `type`;
3. the top-level `status_codes` (for `vllm`, without `400`).

An override **replaces** the set and does not merge with it. List every code you want retried, including the ones already in `status_codes`. An override with an empty list (`status_codes: []`) disables retries for that credential or provider.

The built-in `vllm` rule is not a fixed list: it is the top-level `status_codes` with `400` removed, so `status_codes: [429]` means `[429]` for `vllm` too. Setting `provider_overrides.vllm` replaces this rule with your list.

Markers apply on top of the status code set, whichever set is in effect. Configured markers are added to the built-in ones, never replace them.

### Why vLLM does not retry 400

Self-hosted vLLM replicas of one model run the same weights, chat template and `--max-model-len`, and have no account or quota behind them. A `400` from one replica comes back the same from all of them. Typical causes:

- a `reasoning_effort` the chat template rejects (`Unexpected reasoning effort minimal`);
- `max_tokens` over the context window;
- a prompt longer than the context window.

So the built-in policy sends such a `400` straight to the client. Every other code of `status_codes` (by default `5xx`, `429` and `404`, e.g. a replica that serves another model) is still retried.

To restore the old behavior:

```yaml
retry:
  provider_overrides:
    vllm:
      status_codes: [400, 404, 429, "5xx"]
```

### Built-in markers

`non_retryable_markers` (any status): `content policy`, `content management policy`, `policy violation`.

`bad_request_markers` (400 only): `penalty is not enabled`, `thinking level is unsupported`, `thinking level minimal is not supported`, `unsupported mime type`, `required oneof field`, `but the supported range is from`.

Keep `bad_request_markers` narrow. Add only texts that describe the request and the model. Anything that can differ between credentials (a model missing from one account, a disabled API, a quota) must stay retryable, because moving to the next credential is exactly what fixes it.

## Request cleanup

Before a request goes to an OpenAI-shaped upstream, the router fixes client params that would only earn a `400`. This covers Chat Completions bodies, including those converted for Vertex, Anthropic and other providers, and Responses bodies forwarded natively. It runs on every attempt (first, retry, fallback). It is skipped for `proxy`/`air` credentials, because the downstream router owns that model's config and does the same on its side.

### Empty tools

`"tools": []` is removed together with `tool_choice` and `parallel_tool_calls`. vLLM and OpenAI reject an empty tools array, although the request means "no tools". A non-empty or missing `tools` is left as is. This needs no configuration.

### Reasoning effort mapping

Some chat templates accept only a few reasoning effort values. Qwen3.8 raises a `400` on anything except `xhigh`, `medium` and `low`. GLM-5.3 silently turns anything except `low` and `high` into `max`. `reasoning_effort_map` on a model rewrites the client's value into one the model accepts:

```yaml
models:
  - name: qwen3-flash
    credential: vllm-qwen-a
    reasoning_effort_map:
      minimal: low
      high: medium
      max: xhigh
      default: low
```

The same map can be written as a JSON string, which is handy for generated configs:

```yaml
    reasoning_effort_map: '{"minimal": "low", "high": "medium", "max": "xhigh", "default": "low"}'
```

The client value is resolved case-insensitively:

1. a key of the map: replaced by its value (`minimal` → `low`);
2. a value of the map: sent unchanged, because the model accepts it (`medium`, `xhigh`, `low`);
3. anything else: replaced by `default` (`none`, `turbo` → `low`), or sent unchanged when `default` is not set.

So you only list the values to rewrite. Accepted values pass because they appear on the right-hand side. If a value must pass through unchanged but is never a target, map it to itself. For example, `none: none` keeps `none` for Qwen, where it turns thinking off.

The map rewrites every place a request can carry the effort:

| Field                                   | API                     |
| --------------------------------------- | ----------------------- |
| `reasoning_effort`                      | Chat Completions        |
| `chat_template_kwargs.reasoning_effort` | Chat Completions (vLLM) |
| `reasoning.effort`                      | Responses               |

A request without a reasoning effort is not touched, and the deployment's server default applies (`--default-chat-template-kwargs`).

The map belongs to a `models` entry. With several entries for the same model (one per credential), set it on each entry. An entry without `credential` applies to every credential of that model that has no map of its own.

!!! note
The map is read from `config.yaml` only. Models loaded from a LiteLLM database do not carry one.

### What is not changed

`max_tokens` and the prompt length are sent as the client set them. A request over the context window gets the upstream `400` back. With the default vLLM policy, that `400` is not retried.
