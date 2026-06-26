# Sosana.art

Sosana.art is supported as an image-only provider for the OpenAI-compatible
Images API. The router accepts `/v1/images/generations` and `/v1/images/edits`,
submits a Sosana Banana async task, polls it, and returns an OpenAI Images
response with `data[].b64_json`.

Chat Completions, Responses API, Embeddings, video, and slides are not routed to
Sosana in this integration.

## Configuration

```yaml
credentials:
  - name: "sosana_images"
    type: "sosana"
    api_key: "os.environ/SOSANA_API_KEY"
    base_url: "https://sosana.art"
    rpm: 60
    tpm: -1

models:
  - name: "nano-banana"
    credential: sosana_images
    rpm: 60
    tpm: -1
```

Sosana Banana tasks are asynchronous and can take longer than short chat
completion requests. For production Sosana credentials, set the router
`request_timeout` and HTTP `write_timeout` to at least `2m`.

## Behavior

- `n` must be `1`.
- Default response format and `response_format: "b64_json"` return
  `data[].b64_json`.
- `response_format: "url"` returns a local 400 because URL responses require
  VSELLM-owned rehosting before they can hide Sosana storage.
- `/v1/images/edits` sends uploaded images as `data:image/...;base64,...`
  values in Sosana `image_urls`.
- Mask images are not supported.

On completion, Sosana returns a public object URL in `result_file_url`. The
router downloads that object without forwarding the Sosana `Authorization`
header, keeps the download bounded to 32 MiB, verifies the body is an image, and
base64-encodes it into the OpenAI-compatible JSON response. The upstream object
URL is not returned to clients.

## Error Masking

Sosana upstream HTTP errors and terminal task errors are masked before they are
returned to clients or written to structured logs. The router preserves the
appropriate HTTP status but replaces provider details with neutral
OpenAI-compatible error bodies.

If Sosana is hidden behind another proxy credential, enable
`mask_upstream_errors: true` on that proxy unless the upstream router is known to
propagate the credential marker used by this router.
