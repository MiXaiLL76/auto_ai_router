# Loop Gate (Looping Generation Detector)

Self-hosted reasoning models (Qwen 3.x and similar on vLLM) sometimes get stuck repeating
the same passage until `max_tokens`: a request runs for 10–15 minutes, ends with
`finish_reason: length`, and the slot and KV cache are busy with garbage the whole time.

The loop gate watches the generated text as it streams through AIR and detects such loops.
It never touches the engine, the sampler or the weights: no logits processors, no
`repetition_penalty`.

Based on [loop-gate](https://github.com/vakovalskii/loop-gate-skill) by Valerii Kovalskii (MIT).

!!! note "Current stage: observe only"
`observe` is the only mode for now. AIR logs and counts the loops it sees and changes
nothing in the response. Use it to measure how much of your generation is wasted and to
calibrate the thresholds on your traffic. Cutting and retrying looping generations come
in later stages.

## How detection works

- The answer text (`content`) and the reasoning text (`reasoning_content` / `reasoning`)
  are counted separately. Tool call arguments are never counted.
- The text is split into words on whitespace. AIR counts sequences of `ngram` words
  (32 by default) with a rolling hash, which costs O(1) per word.
- A channel **fires** when some sequence occurred at least `min_repeat` times **and** the
  share of repeated sequences is at least `min_ratio`.
- Nothing fires before `arm_words` words. Short honest answers with lists and tables
  repeat short runs, while real loops show up far later.

A 32-word window matches only literally identical text, values included. An array of
same-shaped JSON objects with different data never matches, so strict JSON-schema output
is not mistaken for a loop. In the reference production data, healthy long answers never
reached a ratio of 0.35, while loops climbed to 0.8–0.99.

## Configuration

```yaml
loop_gate:
  mode: observe # off (default) | observe; os.environ/VAR works
  providers: [vllm] # credential types the global mode applies to (default: [vllm])
  ngram: 32
  arm_words: 3000
  min_repeat: 5
  min_ratio: 0.35

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
          mode: off # long reviews that reread the same material
```

Settings are resolved per request in this order: the global `loop_gate` block, then the
credential's `loop_gate`, then the model's `loop_gate`. Each field set at a later level
replaces the earlier one. A model entry in the top-level `models:` list without a
`credential` applies to every credential that serves it.

- The global `mode` applies only to credentials whose type is listed in `providers`. A
  paid provider bills the abandoned part of a cut or retried generation too, so the default
  is self-hosted vLLM only.
- A `mode` set explicitly on a credential or a model applies whatever the provider type.
  Setting thresholds alone does not opt in another provider.

### Coverage

Only **streamed Chat Completions** (`/v1/chat/completions` with `stream: true`) served
directly by a credential are watched. Non-streamed requests, `/v1/responses` and
`/v1/messages` over vLLM, and requests forwarded to an `air`/`proxy` peer are not watched
yet. Only choice 0 is watched.

## What it reports

Each firing logs a `WARN` once per channel:

```
Loop gate: generation is looping  mode=observe credential=vllm_qwen model=qwen3.6
  channel=reasoning words=3000 elapsed_s=81.4 top_repeat=71 ratio=0.93 request_id=...
```

`elapsed_s` counts from the start of the upstream stream. When the stream ends, an `INFO`
line reports how it ended:

```
Loop gate: looping stream ended  channel=reasoning words_total=36811
  words_after_firing=33811 final_ratio=0.98 finish_reason=length ...
```

No response text is logged.

Prometheus metrics (see [Prometheus](../monitoring/prometheus.md)):

| Metric                                            | Labels                                 |
| ------------------------------------------------- | -------------------------------------- |
| `auto_ai_router_loop_gate_streams_total`          | `credential`, `model`, `mode`          |
| `auto_ai_router_loop_gate_firings_total`          | `credential`, `model`, `channel`, `mode` |
| `auto_ai_router_loop_gate_words_after_firing_total` | `credential`, `model`, `channel`, `mode` |

`words_after_firing` is roughly what cutting the generation at the firing would have saved.

## Calibrating

1. Enable `mode: observe` for a day or two.
2. Compare the firings with honest long answers. Firings that end with
   `finish_reason: stop` and a modest `final_ratio` are candidates for false positives.
3. If a model fires falsely, raise its `min_ratio` or `arm_words`, or switch it off with a
   model-level `mode: off`.
