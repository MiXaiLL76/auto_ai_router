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

A LiteLLM deployment can carry default request parameters in its `litellm_params`
(for example `chat_template_kwargs: {enable_thinking: false}`, `temperature`,
`top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty`, `max_tokens`, `seed`). For `vllm` credentials AIR fills these
into `/chat/completions` requests **only when the client did not send the same key**,
which is the precedence LiteLLM uses (the request wins over the deployment). A deployment
`max_tokens` is likewise not added when the client sent `max_completion_tokens`.
Embeddings, the passed-through `/v1/responses` and other endpoints never receive them.

Defaults are read from the LiteLLM database only; there is no YAML setting for them.
