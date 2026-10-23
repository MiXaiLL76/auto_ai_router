# Loop Gate (Looping Generation Detector)

Self-hosted reasoning models (Qwen 3.x and similar on vLLM) sometimes get stuck repeating
the same passage until `max_tokens`: a request runs for 10–15 minutes, ends with
`finish_reason: length`, and the slot and KV cache are busy with garbage the whole time.

The loop gate watches the generated text as it streams through AIR, detects such loops and,
in `enforce` mode, stops them. It never touches the engine, the sampler or the weights: no
logits processors, no `repetition_penalty`.

Based on [loop-gate](https://github.com/vakovalskii/loop-gate-skill) by Valerii Kovalskii (MIT).

| Mode            | What happens on a loop                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `off` (default) | nothing, the detector does not run                                                                         |
| `observe`       | logged and counted; the response is not changed. Use it to measure and calibrate first                     |
| `enforce`       | the upstream generation is aborted, then **restarted** in the same stream or the stream is **cut** (below) |

## How detection works

- The answer text (`content`) and the reasoning text (`reasoning_content` / `reasoning`)
  are counted separately. Tool call arguments are never counted.
- The text is split into words on whitespace. A Han, Hiragana or Katakana character is a
  word of its own (Chinese and Japanese are written without spaces), and a run of more than
  256 bytes without whitespace is split into 256-byte words. AIR counts sequences of
  `ngram` words (32 by default) with a rolling hash, which costs O(1) per word.
- A channel **fires** when some sequence occurred at least `min_repeat` times **and** the
  share of repeated sequences is at least `min_ratio`.
- Nothing fires before `arm_words` words. Short honest answers with lists and tables
  repeat short runs, while real loops show up far later.

A 32-word window matches only literally identical text, values included. An array of
same-shaped JSON objects with different data never matches, so strict JSON-schema output
is not mistaken for a loop. In the reference production data, healthy long answers never
reached a ratio of 0.35, while loops climbed to 0.8–0.99.

## Enforce mode

`enforce` acts once a channel's repeat ratio reaches `cut_min_ratio` (default: `min_ratio`).
AIR closes the upstream connection, so vLLM stops generating, and then:

| Where the loop is                                                                       | What the client gets                                                                                                                                       |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| reasoning, no answer text or tool call streamed yet                                     | **restart**: the request is sent again to the same credential with a new `seed` and a higher `presence_penalty`; the new attempt continues the same stream |
| the answer text (the client is already reading the answer)                              | **cut**: a final chunk with `finish_reason: "length"` and an `air_loop_gate` marker, a usage chunk, then `data: [DONE]`                                    |
| restarts used up (`max_attempts`), past `retry_deadline`, or the restart request failed | **cut**                                                                                                                                                    |

The closing chunk of a cut stream tells the client that `length` came from AIR, not from its
`max_tokens`:

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion.chunk",
  "model": "qwen3.6",
  "choices": [{ "index": 0, "delta": {}, "finish_reason": "length" }],
  "air_loop_gate": { "action": "cut", "channel": "content", "words": 3000, "ratio": 0.93, "attempts": 1 }
}
```

A restarted stream is not marked. Its reasoning keeps the looped part of the abandoned
attempt: the client already received it.

While the new attempt is on its way (the request, then vLLM's prefill until its first chunk),
the client gets an SSE comment (`: air loop gate restart`) every 15 seconds, so a proxy or SDK
read timeout in front of it does not drop the stream. SSE clients skip comments. A
non-streamed request (`force_stream`) has no stream to keep alive: the client waits for the
whole answer anyway.

For a streamed chat completion to vLLM, enforce mode adds
`stream_options.continuous_usage_stats: true` to the request, so a cut or restarted attempt
has the usage the engine reported up to that point (exact, prompt included), not an estimate.
The running usage is replaced with `null` on every chunk with choices before the client and
AIR's accounting see it; the closing usage chunk (`choices: []`) keeps it. A client that asked
for `continuous_usage_stats` itself gets it unchanged.

### Non-streamed requests

With `force_stream: true` (off by default), enforce mode sends a non-streamed chat completion
upstream as a stream (`stream: true`, `stream_options.include_usage`, and for vLLM
`continuous_usage_stats`). AIR reads it through the gate and returns the one
`chat.completion` the client asked for, rebuilt from the chunks: text and reasoning
concatenated, tool calls assembled by index, `content: null` next to a tool call, usage from
the stream, and an error event inside the stream returned with its own status.

The client sees nothing until the answer is complete, so a loop is restarted **wherever** it
is (reasoning or answer). When `max_attempts` or `retry_deadline` run out, the best attempt
(the longest answer) is returned with its looped tail trimmed: the text ends where the
second round of the loop begins. It has `finish_reason: "length"`, the `air_loop_gate`
marker at the top level, and the usage vLLM reported at the cut (an estimate when the
upstream reported none: the local prompt estimate plus 4 tokens per 3 streamed words).

Non-streamed `/v1/responses` (`passthrough_responses: false`) and `/v1/messages` requests
converted to chat completions are gated the same way.

With `force_stream: false` (the default), or in `observe` mode, non-streamed responses are
only scanned after they arrive: firings are logged and counted (at the word a stream would
have fired on), the response is not changed.

!!! warning "Before you turn on `force_stream`"
    Every non-streamed chat completion to the credential then goes upstream as a stream, loop or
    no loop, and the client gets an answer rebuilt by AIR, not the one vLLM returns for a
    non-streamed request:

    - **Tool calls** come from vLLM's streaming tool parser, which is less reliable than the
      non-streamed one for some parsers (`hermes`, `qwen3_coder`, ...): malformed or missing
      `tool_calls` where the non-streamed request had them right.
    - **Rejected options:** vLLM refuses `best_of` and `use_beam_search` with `stream: true`
      (400).
    - **Dropped fields:** the rebuild keeps id, model, created, system_fingerprint, usage,
      and per choice: role, content, reasoning, refusal, tool_calls, logprobs (`content`),
      finish_reason and stop_reason. It drops `prompt_logprobs`, `prompt_token_ids` /
      `token_ids` (`return_token_ids`), `kv_transfer_params`, `message.annotations`,
      `service_tier` and any other vLLM extension.
    - **Response size limit:** `max_response_body_size` applies to the bytes of the stream,
      not of the rebuilt JSON. A stream is several times larger than the answer (an event
      envelope per token, plus the usage on every chunk with `continuous_usage_stats`), so a
      long answer that fits as JSON can fail with "response too large". Raise the limit for
      such credentials, or keep `force_stream` off.
    - Each restart adds the time the loop took to show up, and the client sees nothing
      until the last attempt ends.

    Turn it on per model (`models[].loop_gate.force_stream: true`) for clients that do not
    stream and do not depend on the above, after checking their requests in `observe`.

!!! warning "Side effects"
\- A cut stream ends with a usage chunk (`choices: []`): the last usage the upstream reported in
the stream (exact with vLLM's `continuous_usage_stats`), otherwise the local prompt estimate
plus 4 tokens per 3 streamed words. It is shown to the client only when it asked for
`stream_options.include_usage`, like any usage chunk, and it is what AIR bills.
\- A restart sets `presence_penalty` to the higher of the client's and `retry_presence_penalty`.
A client `seed` becomes `seed + restart number` (attempts stay reproducible but differ); without
one, a random seed is used.
\- Each restart adds the time the loop took to show up (often more than a minute).
\- No restart when the request's images were described by [vision fallback](vision_fallback.md):
the descriptions were injected into the first attempt only. Such streams are cut.

### Billing and rate limits

The client is billed for what the returned attempt reports in `usage`. The abandoned
attempts are not billed, but they used the engine (each one ran the prefill again), so their
prompt and generated tokens count against the credential's and the model's TPM: the running
usage vLLM reported (`continuous_usage_stats`), otherwise the local prompt estimate plus 4
tokens per 3 streamed words. This holds for cut streams too: their usage covers the last
attempt only.

A restart is a request to the same credential: it counts toward the credential's and the
model's RPM (while under the limit: a running stream is never held or refused for RPM),
and its outcome goes to fail2ban and the credential error metric like any attempt.

## Configuration

```yaml
loop_gate:
  mode: enforce # off (default) | observe | enforce; os.environ/VAR works
  providers: [vllm] # credential types the global mode applies to (default: [vllm])
  ngram: 32
  arm_words: 3000
  min_repeat: 5
  min_ratio: 0.35
  cut_min_ratio: 0.35 # enforce acts at this ratio; 0 or omitted = min_ratio
  max_attempts: 3 # first attempt included; 1 = never restart, only cut
  retry_deadline: 10m # no restart later than this after the stream started
  retry_presence_penalty: [1.0, 1.5] # restart 1, restart 2 and later
  force_stream: false # enforce: send non-streamed chat completions upstream as a stream (see the warning above)

credentials:
  - name: vllm_qwen
    type: vllm
    base_url: http://vllm-qwen:8000
    loop_gate: # override for every model of this credential
      min_ratio: 0.4
    models:
      - name: qwen3.6
        loop_gate: # override for this model; wins over the credential block
          arm_words: 2000
      - name: qwen-judge
        loop_gate:
          mode: observe # long reviews that reread the same material: only measure
```

Settings are resolved per request in this order: the global `loop_gate` block, then the
credential's `loop_gate`, then the model's `loop_gate`. Each field set at a later level
replaces the earlier one. A model entry in the top-level `models:` list without a
`credential` applies to every credential that serves it. Every credential and model
combination is validated at startup.

- The global `mode` applies only to credentials whose type is listed in `providers`. A
  paid provider bills the abandoned part of a cut or restarted generation too, so the
  default is self-hosted vLLM only.
- A `mode` set explicitly on a credential or a model applies whatever the provider type.
  Setting thresholds alone does not opt in another provider.
- `cut_min_ratio` must be at least the effective `min_ratio`. A higher value lets a
  borderline answer run longer before it is cut; firings are still logged at `min_ratio`.

### Coverage

What is watched depends on what the credential sends upstream, not on the endpoint the
client called:

| Upstream answer                                                                                   | Streamed                             | Not streamed                                         |
| ------------------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------- |
| chat completion: `/v1/chat/completions`, and `/v1/responses` / `/v1/messages` **converted** to it | restart (reasoning) or cut           | `force_stream`: restart anywhere, trimmed at the end |
| native `/v1/responses` (**passthrough**, the vLLM default)                                        | cut with `response.incomplete`       | observed only                                        |
| native `/v1/messages` (**passthrough**, `passthrough_messages: true`)                             | cut with `stop_reason: "max_tokens"` | observed only                                        |

- A converted stream is restarted like a chat stream: the new attempt goes through the same
  conversion, so the client still sees one `response.created` / `message_start`.
- A passthrough stream is never restarted: a second attempt would start a second response
  or message. It is cut in its own format:
  - Responses API: the output items left open are closed the way a provider closes them
    at the output limit (`response.output_text.done` / `reasoning_text.done` /
    `content_part.done` / ..., then `response.output_item.done` with `status: "incomplete"`
    and the full streamed text), then `response.incomplete` (`status: "incomplete"`,
    `incomplete_details.reason: "max_output_tokens"`, `air_loop_gate` marker). It carries the
    whole output, the other fields of `response.created`, and an estimated `usage`
    (`input_tokens` from the upstream if it reported them, else the local estimate; output
    as 4 tokens per 3 words). The response stored for `previous_response_id` has that output.
  - Messages API: the open content block is closed, then `message_delta` with
    `stop_reason: "max_tokens"`, an estimated `usage.output_tokens` and the
    `air_loop_gate` marker, then `message_stop`.
- Not watched: `responses_only` models, requests forwarded to an `air`/`proxy` peer (gate
  them on the peer). Only choice 0 of a chat completion is watched.

!!! note "Non-streamed passthrough is observed only, by design"
A non-streamed `/v1/responses` or `/v1/messages` request relayed to vLLM as is
(passthrough) is only scanned after the answer arrives, in `enforce` mode too: firings
are logged and counted, the response is not changed and nothing is restarted. Gating it
would mean streaming the native API upstream and rebuilding its response object (every
output item and content block type) in AIR. Such requests are rare (Codex, Claude Code
and most agents stream), so this stays observe-only for now. To have them gated, convert
them instead (`passthrough_responses: false`; `/v1/messages` is converted by default):
converted requests are gated like chat completions.

## What it reports

Each firing logs a `WARN` once per channel and attempt:

```
Loop gate: generation is looping  mode=enforce credential=vllm_qwen model=qwen3.6
  channel=reasoning words=3000 elapsed_s=81.4 top_repeat=71 ratio=0.93 attempt=1 request_id=...
```

`elapsed_s` counts from the start of the upstream stream of that attempt. Enforce mode adds
`Loop gate: generation restarted`, `Loop gate: stream cut` and
`Loop gate: restart recovered the generation`. At the end of each fired attempt an `INFO`
line reports how it ended:

```
Loop gate: looping stream ended  channel=reasoning words_total=36811
  words_after_firing=33811 final_ratio=0.98 finish_reason=length attempt=1 ...
```

`finish_reason` there is the upstream one, or `loop_gate_restart` / `loop_gate_cut`. No
response text is logged.

Prometheus metrics (see [Prometheus](../monitoring/prometheus.md)):

| Metric                                              | Labels                                   |
| --------------------------------------------------- | ---------------------------------------- |
| `auto_ai_router_loop_gate_streams_total`            | `credential`, `model`, `mode`            |
| `auto_ai_router_loop_gate_firings_total`            | `credential`, `model`, `channel`, `mode` |
| `auto_ai_router_loop_gate_words_after_firing_total` | `credential`, `model`, `channel`, `mode` |
| `auto_ai_router_loop_gate_restarts_total`           | `credential`, `model`, `channel`         |
| `auto_ai_router_loop_gate_recovered_total`          | `credential`, `model`                    |
| `auto_ai_router_loop_gate_cuts_total`               | `credential`, `model`, `channel`         |

In `observe` mode `words_after_firing` is roughly what `enforce` would have saved.

## Rolling out

1. Enable `mode: observe` for a day or two.
2. Compare the firings with honest long answers. Firings that end with
   `finish_reason: stop` and a modest `final_ratio` are candidates for false positives.
3. If a model fires falsely, raise its `min_ratio` or `arm_words`, or keep it in `observe`
   with a model-level `mode`.
4. Switch to `mode: enforce` (an `os.environ/` value lets you roll back without a deploy).
   Watch `restarts_total` against `recovered_total` and `cuts_total`.
