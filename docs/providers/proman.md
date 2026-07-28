# ProMan

ProMan is supported as an Anthropic-compatible provider via the dedicated
`proman` credential type. The router treats it as a provider with its own
compatibility rules, not as a generic `proxy` credential.

## Configuration

Set the credential `type` to `proman`. Required fields are `api_key` and
`base_url`. Use public model names for clients and map them to ProMan model
aliases with model-level `model` values when the names differ.

## Implemented

| Area                         | Status                                                               |
| ---------------------------- | -------------------------------------------------------------------- |
| Provider type                | `proman` is a first-class credential type.                           |
| Chat Completions             | Converted to Anthropic Messages and sent to ProMan.                  |
| Responses API                | Converted to Anthropic Messages for ProMan.                          |
| Streaming                    | Supported through the Anthropic-compatible streaming path.           |
| Model catalog                | Internal upstream model IDs are sanitized from client responses.     |
| Error surface                | ProMan upstream errors are masked before returning to clients.       |
| Header surface               | Internal provider and LiteLLM/LightLLM routing headers are stripped. |
| Unsupported request handling | Known provider-level unsupported blocks are not sent to ProMan.      |

## Supported By Live Tests

| Capability                                                                       | Result                        |
| -------------------------------------------------------------------------------- | ----------------------------- |
| Basic `/v1/messages` request to ProMan                                           | Works.                        |
| System prompt                                                                    | Works.                        |
| Content blocks                                                                   | Works.                        |
| Consecutive user messages                                                        | Works.                        |
| Assistant prefill                                                                | Works in current smoke tests. |
| `stop_sequences`                                                                 | Works.                        |
| Forced tool calls                                                                | Works.                        |
| Streaming                                                                        | Works.                        |
| `/v1/messages/count_tokens`                                                      | Works.                        |
| OpenAI-compatible `/v1/chat/completions`                                         | Works.                        |
| Router `/v1/chat/completions` through ProMan                                     | Works.                        |
| Router `/v1/responses` through ProMan                                            | Works.                        |
| Full chain: client -> LiteLLM -> router -> ProMan -> router -> LiteLLM -> client | Works in smoke tests.         |

## Not Implemented Or Not Claimed

| Capability                                                | Status                                                                                                                                                                                    |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/v1/messages/batches`                                    | Not claimed by this router surface. Direct ProMan can create and run batches, but currently returns `results_url` on `api.anthropic.com`; the same results path works on the ProMan host. |
| Direct ProMan `/v1/responses`                             | Not required. Router converts Responses API requests to Anthropic Messages for ProMan; do not force `passthrough_responses: true` for ProMan models.                                      |
| Native Anthropic Messages endpoint exposed by this router | Not part of the current ProMan integration surface.                                                                                                                                       |
| Long-context guarantees                                   | Not claimed from the current smoke tests.                                                                                                                                                 |
| Large tool schemas                                        | Not fully covered by the current smoke tests.                                                                                                                                             |
| Parallel tool call quality                                | Not fully covered by the current smoke tests.                                                                                                                                             |
| Image/file inputs                                         | Not claimed unless covered by a separate model-specific test.                                                                                                                             |

## ProMan Compatibility Guards

When one of these request shapes is detected for a ProMan credential, the router
does not send the request to ProMan. It tries another primary credential first,
then a fallback proxy. If no compatible route exists, it returns a local 400 with
a neutral error message.

| Request field or shape                            | Router behavior |
| ------------------------------------------------- | --------------- |
| Top-level message content `server_tool_use` block | Skip ProMan.    |

The router does not hard-code ProMan behavior by concrete model name. Model-level
capability drift must be tracked by live/provider fixture results and provider
configuration, not by adding temporary model matrices to the proxy binary.

The router does not add ProMan-specific blocks for `temperature`, `top_p`,
`top_k`, reasoning/thinking parameters, assistant prefill,
`tool_choice: {"type":"none"}`, or text documents. These request shapes passed
the latest direct ProMan compatibility checks and should follow the normal
provider path.

## Masked From Clients

| Upstream data                                                       | Client-visible behavior                         |
| ------------------------------------------------------------------- | ----------------------------------------------- |
| ProMan provider name in errors                                      | Removed.                                        |
| Internal ProMan model IDs                                           | Removed from sanitized client bodies.           |
| Upstream route or credential names                                  | Removed from client headers and error bodies.   |
| AWS Bedrock or internal backend markers in upstream errors          | Removed from client error bodies.               |
| LiteLLM/LightLLM provider-specific fields in ProMan upstream bodies | Removed before the router returns the response. |
| Internal provider headers                                           | Not copied to the client response.              |

## Outside Router Scope

| Area                                                           | Status                                                                                   |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Fields added by a downstream gateway after the router response | Not controlled by the router. Configure that gateway separately.                         |
| Raw operator logs                                              | The router may keep upstream diagnostics in internal logs. Client responses stay masked. |

## Current Status

ProMan is usable behind the router for the tested Anthropic-compatible surface.
It should stay configured as `type: "proman"` so the provider-specific masking
and compatibility guards apply. Do not configure it as a generic `proxy` if the
goal is to hide ProMan-specific errors, headers, and model IDs from clients.
