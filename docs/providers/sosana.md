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
  - name: "google/gemini-3.1-flash-image-preview"
    model: "banana-2-{image_size}-compliant"
    credential: sosana_images
    rpm: 60
    tpm: -1
```

The credential value is configured as `api_key`. The router sends it to Sosana
as `Authorization: Bearer <api_key>`, matching Sosana's API contract.

The dynamic model template maps `image_size` to Sosana's concrete image models:
`banana-2-1k-compliant`, `banana-2-2k-compliant`, and
`banana-2-4k-compliant`. If `image_size` is omitted, the router uses `1K`.
This integration maps Sosana only for `google/gemini-3.1-flash-image-preview`;
other image families should be served by their native providers or fallback
proxies.

Sosana Banana tasks are asynchronous and can take longer than short chat
completion requests. For production Sosana credentials, set the router
`request_timeout` and HTTP `write_timeout` to at least `2m`.

## Behavior

- `n` must be `1`.
- Requests selected to a Sosana credential are skipped when they require
  controls that Sosana does not support. The router then tries another primary
  credential for the same model and then the configured fallback proxy cascade.
  If no compatible provider is available, the router returns a local 400.
- Default response format and `response_format: "b64_json"` return
  `data[].b64_json`.
- `response_format: "url"` is not routed to Sosana because URL responses require
  VSELLM-owned rehosting before they can hide Sosana storage. Another provider
  may handle it through normal fallback routing.
- `image_size` may be omitted or set to `1K`, `2K`, or `4K`; `0.5K` is not
  routed to Sosana. Pixel `size` values are accepted only when they match the
  documented Gemini `image_size` + `aspect_ratio` table for `1K`, `2K`, or
  `4K`.
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

## Billing

Sosana vendor prices are not used at request time and are not returned to
clients. Successful image requests log `ImageCount=1`; spend is calculated from
the internal price registry or LiteLLM model table using `output_cost_per_image`.
When `image_size` selects a concrete tier, the spend lookup uses the concrete
model first, for example `banana-2-2k-compliant`, and then falls back to the
public model name if no concrete price is configured.

## Error Masking

Sosana upstream HTTP errors and terminal task errors are masked before they are
returned to clients. The router preserves the appropriate HTTP status but
replaces provider details with neutral OpenAI-compatible error bodies.

For operator debugging, structured logs may include a truncated textual upstream
error body with `response_body_masked=true`. Raw image bytes and full result
URLs are not logged.

If Sosana is hidden behind another proxy credential, the upstream router should
propagate the credential marker used by this router so proxy-chain errors can be
masked as Sosana errors too.
