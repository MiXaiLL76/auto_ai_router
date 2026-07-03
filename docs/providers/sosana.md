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
- Requests selected to a Sosana credential return a local 400 when they require
  controls that Sosana does not support.
- Default response format and `response_format: "b64_json"` return
  `data[].b64_json`.
- `response_format: "url"` returns a local 400 because URL responses require
  VSELLM-owned rehosting before they can hide Sosana storage.
- `/v1/images/edits` accepts PNG input images only, up to 14 files, and sends
  them as `data:image/png;base64,...` values in Sosana `image_urls`.
- Mask images are not supported.
- Output is PNG. `output_format` may be omitted or set to `png`; other formats
  are not routed to Sosana.
- The router sends `prompt_optimization: false` so Sosana returns `MODERATED`
  instead of rewriting moderated prompts into safe alternatives.

On completion, Sosana returns a public object URL in `result_file_url`. The
router downloads that object only from allowed Sosana/CDN hosts, does not follow
redirects, does not forward the Sosana `Authorization` header, keeps the download
bounded to 32 MiB, verifies the body is PNG, and base64-encodes it into the
OpenAI-compatible JSON response. The upstream object URL is not returned to
clients.

## Error Masking

Sosana upstream HTTP errors and terminal task errors are masked before they are
returned to clients. The router preserves the appropriate HTTP status but
replaces provider details with neutral OpenAI-compatible error bodies.

For operator debugging, structured logs may include a truncated textual upstream
error body with `response_body_masked=true`. Raw image bytes and full result
URLs are not logged.

If Sosana is hidden behind another proxy credential, enable
`mask_upstream_errors: true` on that proxy unless the upstream router is known to
propagate the credential marker used by this router.
