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
credentials — is ignored, and its requests are forwarded unchanged. The first request with
images to such a model logs a `WARN` once:
`supports_vision: false is ignored: the model is served by a non-vLLM credential`.

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
  max_images: 4                     # images described per request (default 4, 0 = no limit)
  max_tokens: 1024                  # max_tokens of each describe call (default 1024)
  timeout: 2m                       # per describe call (default 2m)
  inject_into_response: true        # write descriptions into the answer (default true)
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
from `model_info.supports_vision` (the field LiteLLM itself uses). The same rule applies across
both sources: a `false` from `config.yaml` or from the database makes the name text-only, a
`true` from one source never hides a `false` from the other.

Aliases (`model_group_alias`, `public_model_alias`, `model_alias`) are resolved first, so the
flag of the target model applies.

### `vision_fallback` (global)

| Parameter              | Default                                                   | Description                                                                                                                                                                |
| ---------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`                 | `describe` if `describe_model` is set, otherwise `reject` | What to do with images sent to a `supports_vision: false` model                                                                                                            |
| `describe_model`       | —                                                         | Model used to describe images. Required for `describe`. Must be a model this AIR serves; a model declared `supports_vision: false` in `config.yaml` is rejected at startup |
| `describe_prompt`      | built-in                                                  | System prompt of the describe call                                                                                                                                         |
| `max_images`           | `4`                                                       | Images of one request that are described; further images become placeholders. `0` = no limit                                                                               |
| `max_tokens`           | `1024`                                                    | `max_tokens` of each describe call                                                                                                                                         |
| `timeout`              | `2m`                                                      | Timeout of each describe call                                                                                                                                              |
| `inject_into_response` | `true`                                                    | Write the descriptions at the start of the answer so later turns restore them (see [Descriptions in the answer](#descriptions-in-the-answer))                              |

All values accept `os.environ/VAR`.

`timeout` may exceed `server.write_timeout`: nothing is written to the client while images are
described, so the write deadline is lifted for that step, and the rest of the request gets a full
`write_timeout` afterwards. Keep the client's own timeout in mind — it sees no bytes until the
describe step is over.

## How `describe` works

A conversation is split at the **last model output** (the last `assistant` message; in the
Responses API also the last `reasoning` or `*_call` item). Everything after it is the current
turn.

1. Each image of the current turn (user message or tool result, e.g. a screenshot returned
   by an agent tool) is sent to `describe_model` on its own. The user's question is **not**
   passed: the description is reused on every later turn, so it must cover the whole image,
   not only what the first question asked about.

   ```json
   {
     "model": "qwen-vl",
     "max_tokens": 1024,
     "stream": false,
     "messages": [
       {"role": "system", "content": "Describe this image as precisely and completely as possible: ..."},
       {"role": "user", "content": [
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

3. Images **before** the current turn are never described again. Each gets the description
   recorded in the answer to its turn (see [Descriptions in the answer](#descriptions-in-the-answer)),
   or `[image from an earlier turn omitted]` when there is none.

4. The rewritten request goes to the text-only model as usual (streaming included), and the
   descriptions of this turn are written at the start of its answer.

The response carries `X-AIR-Vision-Fallback`, a comma-separated list of:

| Value           | Meaning                                                                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `described=N/M` | `N` of the `M` images of the current turn were described; the rest became placeholders (failed call, `max_images`, `file_id`) |
| `restored=K`    | `K` history images got their description back from an earlier answer                                                          |
| `stripped`      | only placeholders: `strip` mode, or history images without a recorded description                                             |

`N < M` is a partial result; `described=0/M` means every describe call failed.

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

No describe call is made. If the assistant answer still starts with the description block
AIR wrote into it, the image gets exactly the text it had on the first turn; otherwise it
becomes `[image from an earlier turn omitted]`.

**Trade-off:** a follow-up about a detail the description does not mention ("what colour
are its eyes?") cannot be answered — the model never saw the image. The describe prompt
asks for an exhaustive description to keep this rare; the user can always attach the image
again.

## Descriptions in the answer

AIR keeps no state. To keep an image's description for later turns, it writes the
descriptions of the current turn at the start of the answer, one collapsible block per image:

```text
<details type="air-vision" n="1" model="qwen-vl">
<summary>Image 1 described by qwen-vl</summary>
A brown owl sits on a birch branch at night. ...
</details>

На картинке изображена сова.
```

Open WebUI renders the block collapsed under its summary and sends the answer text back with
the history. On the next request AIR:

- matches the blocks to the images of the turn they answered, by order (`n` counts the images
  that were described, in order) and puts the same text in place of each image — the prompt is
  identical to the first turn, so vLLM's prefix cache still hits;
- removes the blocks from the assistant messages before the conversation goes upstream. This
  also happens for vision-capable vLLM models, e.g. when the conversation is switched to one:
  it sees the original image and does not need the text. A side effect: text in an earlier
  answer of a vLLM model that happens to match the block format exactly
  (`<details type="air-vision" n="…" model="…">…</details>`) is removed as well.

Like the rest of the feature, this applies only to models served exclusively by vLLM. A
conversation switched to a model of another provider (OpenAI, Anthropic, ...) sends the blocks
upstream unchanged, as ordinary text of the earlier answers.

Where the block goes:

| Endpoint               | Non-streaming                                        | Streaming                                                                           |
| ---------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `/v1/chat/completions` | start of `choices[i].message.content` (every choice) | an extra `delta.content` chunk before the first answer chunk of each choice         |
| `/v1/responses`        | start of the first `output_text`                     | start of the first `response.output_text.delta`, and in the matching `.done` events |
| `/v1/messages`         | start of the first `text` block (after `thinking`)   | start of the first `text` block (after `thinking`)                                  |

An answer without text (tool calls only) still gets the block: Chat Completions as the message
`content`; Responses as an extra `message` item at the end of `output` (when streaming, its
events are inserted before `response.completed`).

The block comes after the reasoning, never inside it: Open WebUI does not send reasoning back
with the history. `usage` is unchanged — the inserted text is not billed as completion tokens.

The block is **not** written when the client asked for machine-readable output —
`response_format` / `text.format` other than `text`, Messages `output_format`, or a forced tool
call (`tool_choice: "required"`, a named function, Messages `any` / `tool`). Set
`inject_into_response: false` to turn it off entirely; history images then always become
placeholders.

Nor is it written for a vLLM model with an explicit `passthrough_messages: true`: `/v1/messages`
then goes to vLLM natively and the answer comes back in Anthropic format. The images are still
described; only the block in the answer (and so the restore on later turns) is missing.

## Supported APIs

| Endpoint               | Image parts recognized                                                     | Replaced with            |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------ |
| `/v1/chat/completions` | `{"type": "image_url"}` in `user` and `tool` messages                      | `{"type": "text"}`       |
| `/v1/responses`        | `{"type": "input_image"}` in messages and `function_call_output.output`    | `{"type": "input_text"}` |
| `/v1/messages`         | `{"type": "image"}` (base64 or url source), including inside `tool_result` | `{"type": "text"}`       |

For `/v1/responses` with `previous_response_id`, the stored history is prepended first, so its
images are handled like any other earlier-turn image. The stored conversation keeps the
original images, and the stored answer keeps the description block, so the descriptions are
restored from the store as well.

An `input_image` given only as `file_id` cannot be fetched by AIR and becomes a placeholder.

## Billing and limits

The describe call runs through AIR's own pipeline with the caller's headers: the same API key,
the same end-user headers (`X-OpenWebUI-User-Email`, ...). It is therefore authenticated,
rate-limited, load-balanced and billed like a normal request — it shows up as a separate
spend-log row for `describe_model`. The key must be allowed to use `describe_model`.

Each described image counts as one request against the key's RPM: a message with 4 images costs
5 requests. When the key's limit is hit, the remaining images become placeholders and the header
shows it (`described=2/4`). The describe calls can also use up the limit the main request needs:
on a key with 5 RPM, a message with 5 images gets its main request rejected with `429`.

The images are described **before** a credential is picked for the main request, so a rejected
main request (`429`, no credential available, upstream error) has already paid for its describe
calls, and a client that retries pays for them again on every retry — AIR keeps no cache of
descriptions. Keep `max_images` low on keys with tight RPM or budget limits.

The describe call has its own `request_id` (it is the key of its spend-log row). It runs inside the
caller's trace, and the `DEBUG` line `Vision fallback: describe call` links the two
(`request_id` = caller, `describe_request_id` = describe call).

If a describe call fails (error, timeout, model not allowed for the key — the caller gets no
error for that, only a placeholder), the image becomes
`[image omitted: the image could not be described]` and the original request still proceeds.
The failure is logged at `WARN` as `Vision fallback: describe call failed`.

A describe call never triggers another describe round. If `describe_model` is itself marked
`supports_vision: false` (e.g. by a `model_info.supports_vision` value synced from the database
after startup), the describe call is rejected, every image becomes a placeholder, and
`vision_fallback.describe_model has supports_vision: false` is logged once at `WARN`.

## Logs

```text
[INFO] Rewrote image input for a model without vision support model=glm-text outcome=described=1/1 images=1 current_turn_images=1 described=1 restored=0 inject_into_response=true describe_model=qwen-vl
[WARN] Rejected image input for a model without vision support error_code=400 model=glm-text images=1
```

## See also

- [vLLM provider](../providers/vllm.md)
- [Configuration — Models](../getting-started/configuration.md#models)
- Example config: [`examples/vision_fallback.yaml`](https://github.com/MiXaiLL76/auto_ai_router/blob/main/examples/vision_fallback.yaml)
