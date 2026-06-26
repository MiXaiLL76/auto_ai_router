# Sosana.art

Sosana.art is supported as an image-only provider for the OpenAI-compatible
Images API. The router accepts `/v1/images/generations` and `/v1/images/edits`,
submits a Sosana Banana async task, polls it, and returns an OpenAI Images
response with `data[].url`.

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
- `response_format: "b64_json"` is accepted, but Sosana results are returned as
  URLs because Sosana provides `result_file_url`.
- `/v1/images/edits` sends uploaded images as `data:image/...;base64,...`
  values in Sosana `image_urls`.
- Mask images are not supported.

## Error Masking

Sosana upstream HTTP errors and terminal task errors are masked before they are
returned to clients or written to structured logs. The router preserves the
appropriate HTTP status but replaces provider details with neutral
OpenAI-compatible error bodies.

If Sosana is hidden behind another proxy credential, enable
`mask_upstream_errors: true` on that proxy unless the upstream router is known to
propagate the credential marker used by this router.
