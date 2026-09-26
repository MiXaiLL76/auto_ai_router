# vLLM

A self-hosted [vLLM](https://docs.vllm.ai/) server is supported through the
dedicated `vllm` credential type. It speaks the OpenAI wire protocol, so requests
and responses pass through unconverted, exactly like `type: openai`. It is a type
of its own so that spend logs and daily aggregates record the provider as `vllm`,
and so that vLLM-only behaviour (per-model default parameters, below) never applies
to other OpenAI-compatible providers.

## Configuration

```yaml
credentials:
  - name: "vllm_ray"
    type: "vllm"
    base_url: "http://vllm.internal:8000/v1"
    rpm: -1
    tpm: -1
```

`base_url` is required. `api_key` is optional: vLLM is usually run without
`--api-key`, and in that case no `Authorization` header is sent at all.

`hosted_vllm` (LiteLLM's name for the provider) is accepted as an alias for `vllm`.

## Responses API

vLLM serves `/v1/responses` natively, so a Responses API request is forwarded to it
unchanged (the real model name replaces the model group name) rather than being converted
into a Chat Completions request. Set `passthrough_responses: false` on a model to convert instead.

## Using a LiteLLM database

Deployments read from the LiteLLM database (`LiteLLM_ProxyModelTable`) with
`custom_llm_provider: hosted_vllm` become `vllm` credentials automatically. See
[LiteLLM Database](../litellm-integration/litellm_db.md#models-from-the-litellm-database)
for how names, aliases and default parameters are imported.

## Default parameters

AIR can fill request-body parameters into `/chat/completions` requests sent to `vllm`
credentials. Use them for what the vLLM server cannot default by itself — per-request
fields such as `skip_special_tokens` or `vllm_xargs`. Anything the engine can default
(`--default-chat-template-kwargs`, `--override-generation-config`) is better set there:
it then also applies to requests that bypass the router.

Defaults come from two places:

- **`config.yaml`** — `default_params` on a `models[]` entry. With `credential` it applies
  to that credential only; without it, to every credential serving the model.

    ```yaml
    models:
      - name: unlimited-ocr
        credential: vllm_ocr
        rpm: -1
        tpm: -1
        default_params:
          skip_special_tokens: false
          vllm_xargs: {ngram_size: 35, window_size: 128}
    ```

    `model`, `messages`, `prompt`, `input` and `stream` cannot be set this way.

- **LiteLLM database** — a deployment's `litellm_params`: `chat_template_kwargs`,
  `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
  `repetition_penalty`, `max_tokens`, `seed`.

How a default meets the request:

- A key the client did not send is added. A key the client sent wins.
- When both the default and the client's value are JSON objects (`chat_template_kwargs`,
  `vllm_xargs`, ...), they are merged key by key, recursively, and the client wins on every
  key it sent. A client sending `vllm_xargs: {window_size: 1024}` still gets the default
  `ngram_size`. **This differs from LiteLLM**, which drops the whole default object once the
  request has the key.
- A client value that is not an object — including an explicit `null` — replaces the
  default as a whole; `null` is the way to opt out of a default object.
- A default `max_tokens` is not added when the client sent `max_completion_tokens`
  (and vice versa).
- For the same credential and model, layers are merged in this order, later winning:
  database → `config.yaml` without credential → `config.yaml` with the credential.
  The config wins over the database; a database sync never drops config defaults.

Embeddings, the passed-through `/v1/responses` and other endpoints never receive defaults.

### Literal `extra_body`

The OpenAI SDKs flatten `extra_body` into the top level of the request before sending it.
Clients that post a literal `"extra_body": {...}` instead (curl, hand-written HTTP) would
have those fields ignored by vLLM, so for `vllm` credentials AIR lifts the keys of an
`extra_body` object to the top level and removes `extra_body`, on every endpoint. A key
present at both levels keeps its top-level value. `model`, `messages`, `prompt`, `input`
and `stream` inside `extra_body` are dropped, not lifted: the request has already been
routed and accounted by its top-level values. This happens before defaults are applied.
