# Images for Text-Only Models (Vision Fallback)

Some models behind AIR accept images (Qwen-VL, Qwen 3.x omni/VL builds, GPT-4o, Gemini),
others do not (GLM, gpt-oss). Clients rarely care: OpenWebUI, IDE agents and SDK scripts
send an `image_url` to whichever model is selected, and a text-only vLLM model answers with
an opaque `400` (`... is not a multimodal model`).

Vision fallback lets you declare which models cannot see images and decide what AIR does
with an image sent to them:

| Mode               | What happens                                                                                                                                        | Cost                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `reject` (default) | `400` with a clear message, before any upstream call. No retries, no fail2ban hits                                                                  | none                         |
| `strip`            | Every image is replaced with a text placeholder; the request goes on                                                                                | none                         |
| `describe`         | Images of the **current turn** are described by a vision model you choose, and the description replaces the image. Older images become placeholders | one extra call per new image |

Models that never declared `supports_vision` are not touched: images pass through as before.

!!! note "vLLM only"
Vision fallback applies only to models served **exclusively** by [`vllm`](../providers/vllm.md)
credentials. A `supports_vision: false` on a model reachable through any other provider
(OpenAI, Anthropic, Vertex, an AIR/proxy peer, ...) — alone or mixed with vLLM
credentials — is ignored, and its requests are forwarded unchanged.

## Configuration

```yaml
models:
  - name: glm-text
    credential: vllm_glm_a
    supports_vision: false          # text-only: vision_fallback applies
  - name: glm-text
    credential: vllm_glm_b
    supports_vision: false
  - name: qwen-vl
    credential: vllm_qwen
    supports_vision: true           # optional; images pass through either way
  - name: gpt-oss
    credential: vllm_gpt_oss
    supports_vision: false

vision_fallback:
  mode: describe                    # reject | strip | describe
  describe_model: qwen-vl           # vision-capable model served by this AIR
  max_images: 4                     # images described per request (default 4)
  max_tokens: 1024                  # max_tokens of each describe call (default 1024)
  timeout: 2m                       # per describe call (default 2m)
  # describe_prompt: "..."          # system prompt of the describe call (a detailed default is built in)
```

### `supports_vision` (per model)

| Value   | Meaning                                                                    |
| ------- | -------------------------------------------------------------------------- |
| omitted | unknown — images are forwarded unchanged (previous behaviour)              |
| `true`  | the model accepts images — forwarded unchanged                             |
| `false` | the model cannot see images — `vision_fallback` applies (vLLM-only models) |

The flag is per model **name**. When the same name is served by several entries and one
of them says `false`, the whole name is treated as text-only, because a request may land on
that deployment.

Models loaded from the [LiteLLM database](../litellm-integration/litellm_db.md) take the flag
from `model_info.supports_vision` (the field LiteLLM itself uses). A `supports_vision` set in
`config.yaml` for the same name wins over the database.

### `vision_fallback` (global)

| Parameter         | Default                                                   | Description                                                                             |
| ----------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `mode`            | `describe` if `describe_model` is set, otherwise `reject` | What to do with images sent to a `supports_vision: false` model                         |
| `describe_model`  | —                                                         | Model used to describe images. Required for `describe`. Must be a model this AIR serves |
| `describe_prompt` | built-in                                                  | System prompt of the describe call                                                      |
| `max_images`      | `4`                                                       | Images of one request that are described; further images become placeholders            |
| `max_tokens`      | `1024`                                                    | `max_tokens` of each describe call                                                      |
| `timeout`         | `2m`                                                      | Timeout of each describe call                                                           |

All values accept `os.environ/VAR`.

## How `describe` works

A conversation is split at the **last model output** (the last `assistant` message; in the
Responses API also the last `reasoning` or `*_call` item). Everything after it is the current
turn.

1. Each image of the current turn (user message or tool result, e.g. a screenshot returned
   by an agent tool) is sent to `describe_model` together with the user's text of that turn:

   ```json
   {
     "model": "qwen-vl",
     "max_tokens": 1024,
     "stream": false,
     "messages": [
       {"role": "system", "content": "Describe this image as precisely and completely as possible: ..."},
       {"role": "user", "content": [
         {"type": "text", "text": "User's question about the image: что на картинке?"},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
       ]}
     ]
   }
   ```

   Several images are described in parallel.

2. The image part is replaced with a text part:

   ```text
   [Image 1, described by qwen-vl because glm-text cannot see images]
   A brown owl sits on a birch branch at night. ...
   ```

3. Images **before** the current turn are replaced with `[image from an earlier turn omitted]`
   without any call. The model already answered about them, and that answer is in the history.

4. The rewritten request goes to the text-only model as usual (streaming included).

The response carries `X-AIR-Vision-Fallback: described` (at least one image was described) or
`stripped` (only placeholders).

### Example

First request — the image is new, so it is described:

```json
{
  "model": "glm-text",
  "messages": [
    {"role": "user", "content": "привет"},
    {"role": "assistant", "content": "Привет! Чем могу помочь?"},
    {"role": "user", "content": [
      {"type": "text", "text": "что на картинке?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
    ]}
  ]
}
```

What `glm-text` receives:

```json
{"role": "user", "content": [
  {"type": "text", "text": "что на картинке?"},
  {"type": "text", "text": "[Image 1, described by qwen-vl because glm-text cannot see images]\nНа изображении сова ..."}
]}
```

Next request — the client sends the whole history again, the image is now old:

```json
{
  "model": "glm-text",
  "messages": [
    {"role": "user", "content": "привет"},
    {"role": "assistant", "content": "Привет! Чем могу помочь?"},
    {"role": "user", "content": [
      {"type": "text", "text": "что на картинке?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
    ]},
    {"role": "assistant", "content": "На картинке изображена сова."},
    {"role": "user", "content": "спасибо"}
  ]
}
```

No describe call is made; the image becomes `[image from an earlier turn omitted]`.

**Trade-off:** a follow-up about a detail that neither the description nor the assistant's
answer mentioned ("what colour are its eyes?") cannot be answered — the model no longer has
the image. The describe prompt asks for an exhaustive description to keep this rare; the
user can always attach the image again.

## Supported APIs

| Endpoint               | Image parts recognized                                                     | Replaced with            |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------ |
| `/v1/chat/completions` | `{"type": "image_url"}` in `user` and `tool` messages                      | `{"type": "text"}`       |
| `/v1/responses`        | `{"type": "input_image"}` in messages and `function_call_output.output`    | `{"type": "input_text"}` |
| `/v1/messages`         | `{"type": "image"}` (base64 or url source), including inside `tool_result` | `{"type": "text"}`       |

For `/v1/responses` with `previous_response_id`, the stored history is prepended first, so its
images are handled like any other earlier-turn image. The stored conversation itself keeps the
original images.

An `input_image` given only as `file_id` cannot be fetched by AIR and becomes a placeholder.

## Billing and limits

The describe call runs through AIR's own pipeline with the caller's headers: the same API key,
the same end-user headers (`X-OpenWebUI-User-Email`, ...). It is therefore authenticated,
rate-limited, load-balanced and billed like a normal request — it shows up as a separate
spend-log row for `describe_model`. The key must be allowed to use `describe_model`.

If a describe call fails (error, timeout, model not allowed), the image becomes
`[image omitted: the image could not be described]` and the original request still proceeds.
The failure is logged at `WARN` as `Vision fallback: describe call failed`.

A describe call never triggers another describe round: if `describe_model` is itself marked
`supports_vision: false`, its images are stripped instead.

## Logs

```text
[INFO] Rewrote image input for a model without vision support model=glm-text outcome=described images=1 described=1 describe_model=qwen-vl
[WARN] Rejected image input for a model without vision support error_code=400 model=glm-text images=1
```

## See also

- [vLLM provider](../providers/vllm.md)
- [Configuration — Models](../getting-started/configuration.md#models)
- Example config: [`examples/vision_fallback.yaml`](https://github.com/MiXaiLL76/auto_ai_router/blob/main/examples/vision_fallback.yaml)
