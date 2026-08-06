# Model Pricing

Auto AI Router supports per-model cost calculation for spend logging. Prices are loaded from a JSON file or remote URL at startup and merged with any prices stored in the LiteLLM database.

## Configuration

```yaml
server:
  model_prices_link: "file://price.json"
```

| Value                                                                                         | Description                   |
| --------------------------------------------------------------------------------------------- | ----------------------------- |
| `file://price.json`                                                                           | Relative path to a local file |
| `file:///data/prices.json`                                                                    | Absolute path                 |
| `https://prices.example.com/default.json`                                                     | Remote HTTPS URL              |
| `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` | LiteLLM's upstream prices     |

The file must be valid JSON and must not exceed 100 MB.

## Price File Format

The file is a JSON object where each key is a model name and each value is a price descriptor:

```json
{
  "gpt-4o-mini": {
    "input_cost_per_token": 1.5e-07,
    "output_cost_per_token": 6e-07
  },
  "gemini-2.5-flash": {
    "input_cost_per_token": 3e-07,
    "output_cost_per_token": 2.5e-06,
    "input_cost_per_audio_token": 1e-06,
    "output_cost_per_reasoning_token": 2.5e-06
  },
  "claude-opus-4-1": {
    "input_cost_per_token": 1.5e-05,
    "output_cost_per_token": 7.5e-05,
    "cache_read_input_token_cost": 1.5e-06,
    "cache_creation_input_token_cost": 1.875e-05,
    "cache_creation_input_token_cost_above_1hr": 3e-05,
    "cache_read_input_token_cost_above_200k_tokens": 3e-06,
    "cache_creation_input_token_cost_above_200k_tokens": 3.75e-05,
    "cache_creation_input_token_cost_above_1hr_above_200k_tokens": 6e-05
  },
  "imagen-4.0-fast-generate-001": {
    "output_cost_per_image": 0.02
  },
  "gpt-4o-search-preview": {
    "input_cost_per_token": 2.5e-06,
    "output_cost_per_token": 1e-05,
    "search_context_cost_per_query": {
      "search_context_size_low": 0.025,
      "search_context_size_medium": 0.0275,
      "search_context_size_high": 0.03
    }
  }
}
```

### Why prices are per 1 token

All per-token prices are expressed as cost per **one token** (not per 1 000 or per 1 million). This matches the format used by LiteLLM's `model_prices_and_context_window.json`, making it straightforward to use the upstream file directly or maintain a custom override file in the same format.

For reference:

- `$1.50 / 1M tokens` → `1.5e-06` (0.0000015)
- `$0.15 / 1M tokens` → `1.5e-07` (0.00000015)

### Available fields

| Field                                                         | Description                                                                  |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `input_cost_per_token`                                        | Regular input tokens                                                         |
| `output_cost_per_token`                                       | Regular output tokens                                                        |
| `input_cost_per_token_above_200k_tokens`                      | Input rate for tokens beyond the 200k threshold                              |
| `output_cost_per_token_above_200k_tokens`                     | Output rate for tokens beyond the 200k threshold                             |
| `input_cost_per_token_above_32k_tokens`                       | Full-session input rate when prompt exceeds 32k tokens                       |
| `output_cost_per_token_above_32k_tokens`                      | Full-session output rate when prompt exceeds 32k tokens                      |
| `input_cost_per_token_above_128k_tokens`                      | Full-session input rate when prompt exceeds 128k tokens                      |
| `output_cost_per_token_above_128k_tokens`                     | Full-session output rate when prompt exceeds 128k tokens                     |
| `input_cost_per_token_above_256k_tokens`                      | Full-session input rate when prompt exceeds 256k tokens                      |
| `output_cost_per_token_above_256k_tokens`                     | Full-session output rate when prompt exceeds 256k tokens                     |
| `input_cost_per_token_above_272k_tokens`                      | Full-session input rate when prompt exceeds 272k tokens                      |
| `output_cost_per_token_above_272k_tokens`                     | Full-session output rate when prompt exceeds 272k tokens                     |
| `input_cost_per_audio_token`                                  | Audio input tokens (falls back to `input_cost_per_token` if absent)          |
| `output_cost_per_audio_token`                                 | Audio output tokens (falls back to `output_cost_per_token` if absent)        |
| `input_cost_per_image_token`                                  | Image input tokens                                                           |
| `output_cost_per_image_token`                                 | Image output tokens                                                          |
| `output_cost_per_reasoning_token`                             | Reasoning/thinking tokens (falls back to `output_cost_per_token`)            |
| `input_cost_per_cached_token`                                 | Cached prompt read cost (alias: `cache_read_input_token_cost`)               |
| `cache_read_input_token_cost`                                 | LiteLLM-compatible alias for `input_cost_per_cached_token`                   |
| `cache_creation_input_token_cost`                             | Prompt cache write cost (falls back to `input_cost_per_token`)               |
| `cache_read_input_token_cost_above_200k_tokens`               | Full-session cache read rate when prompt exceeds 200k tokens                 |
| `cache_creation_input_token_cost_above_200k_tokens`           | Full-session 5m/unclassified cache write rate above 200k                     |
| `cache_creation_input_token_cost_above_1hr`                   | Anthropic 1h cache write rate (falls back to regular cache write rate)       |
| `cache_creation_input_token_cost_above_1hr_above_200k_tokens` | Anthropic 1h cache write rate above 200k                                     |
| `cache_read_input_token_cost_above_32k_tokens`                | Full-session cache read rate when prompt exceeds 32k tokens                  |
| `cache_creation_input_token_cost_above_32k_tokens`            | Full-session cache write rate when prompt exceeds 32k tokens                 |
| `cache_read_input_token_cost_above_128k_tokens`               | Full-session cache read rate when prompt exceeds 128k tokens                 |
| `cache_creation_input_token_cost_above_128k_tokens`           | Full-session cache write rate when prompt exceeds 128k tokens                |
| `cache_read_input_token_cost_above_256k_tokens`               | Full-session cache read rate when prompt exceeds 256k tokens                 |
| `cache_creation_input_token_cost_above_256k_tokens`           | Full-session cache write rate when prompt exceeds 256k tokens                |
| `cache_read_input_token_cost_above_272k_tokens`               | Full-session cache read rate when prompt exceeds 272k tokens                 |
| `cache_creation_input_token_cost_above_272k_tokens`           | Full-session cache write rate when prompt exceeds 272k tokens                |
| `cache_read_input_audio_token_cost`                           | Cached audio input rate (falls back to the selected cache read rate)         |
| `output_cost_per_cached_token`                                | Cached output tokens (falls back to `output_cost_per_token`)                 |
| `output_cost_per_prediction_token`                            | Accepted predicted-output tokens (falls back to `output_cost_per_token`)     |
| `output_cost_per_image`                                       | Cost per generated image (takes priority over `output_cost_per_image_token`) |
| `search_context_cost_per_query`                               | Web Search cost per request/call, keyed by `search_context_size_*`           |
| `web_search_billing_unit`                                     | `per_query` or `per_prompt` Web Search charging mode                         |

## Cost Calculation

All providers return specialised token counts as **subsets** of the totals:

- `prompt_tokens` (Vertex AI, OpenAI) already includes `audio_input_tokens`, `cached_input_tokens`
- `completion_tokens` (all providers) already includes `reasoning_tokens`, `audio_output_tokens`, prediction tokens
- Anthropic reports cache tokens separately; OpenAI-compatible APIs report them in prompt/input token details

To avoid billing the same tokens at two different rates, the calculator first computes **regular** (base-rate) token counts by subtracting all specialised sub-types, then adds each sub-type back at its own rate:

```
regular_input  = prompt_tokens - audio_input_tokens - cached_input_tokens - cache_creation_tokens
regular_output = completion_tokens - audio_output_tokens - reasoning_tokens
                                   - accepted_prediction_tokens - rejected_prediction_tokens

total = regular_input  × input_cost_per_token
      + regular_output × output_cost_per_token
      + audio_input_tokens  × input_cost_per_audio_token
      + audio_output_tokens × output_cost_per_audio_token
      + cached_text_tokens  × cache_read_input_token_cost
      + cached_audio_tokens × cache_read_input_audio_token_cost
      + cache_creation_5m_tokens × cache_creation_input_token_cost
      + cache_creation_1h_tokens × cache_creation_input_token_cost_above_1hr
      + cached_output_tokens   × output_cost_per_cached_token
      + reasoning_tokens            × output_cost_per_reasoning_token
      + accepted_prediction_tokens  × output_cost_per_prediction_token
      + rejected_prediction_tokens  × output_cost_per_token
      + image_count × output_cost_per_image
      + web_search_requests × search_context_cost_per_query[search_context_size]
```

This means every token is billed **exactly once** regardless of how the provider reported it.

### Web Search billing

Web Search is billed as a separate tool cost, not as tokens. The calculator reads LiteLLM-compatible `search_context_cost_per_query` prices and selects one of:

- `search_context_size_low`
- `search_context_size_medium`
- `search_context_size_high`

AIR gets the request size from `web_search_options.search_context_size` or from a `web_search` / `web_search_preview` tool definition. If the request does not specify a size, `medium` is used.

AIR charges only confirmed response usage:

- `usage.server_tool_use.web_search_requests`
- `usage.web_search_requests`
- `response.output[]` or `output[]` items with `type: "web_search_call"`
- Chat Completions `url_citation` annotations when that API contract confirms a search
- Vertex/Gemini `groundingMetadata.webSearchQueries`

Merely enabling a tool does not count as execution. A successful response with no confirmed usage is billed for zero searches. Streaming requests use the final provider usage or completed response output. An incomplete tool event is not billed.

`per_query` multiplies the configured price by the confirmed query count. `per_prompt` clamps any positive count to one charge. LiteLLM Gemini 2.x entries without an explicit unit use `per_prompt`, while Gemini 3.x entries explicitly use `per_query`.

The count and selected context size are written to spend metadata under `usage_object.server_tool_use` and `additional_usage_values.server_tool_use`; the tool cost is written to `cost_breakdown.tool_usage_cost` and `cost_breakdown.web_search_cost`.

### Regular input tokens

Vertex AI and OpenAI include audio and cached tokens **inside** `prompt_tokens`. Anthropic reports cache reads and writes separately on the wire, so AIR first normalises Anthropic usage to an inclusive prompt total. The formula then uses the same semantics for every provider:

- Vertex/OpenAI: `100 prompt − 5 audio − 20 cached = 75 regular`, then +5 audio +20 cached at their rates
- Anthropic wire usage: `100 input + 20 cache read = 120 normalised prompt`; billing uses `120 − 20 cached = 100 regular`, then +20 cached at its rate

### Regular output tokens

All providers include reasoning inside `completion_tokens`:

- OpenAI `o-series`: `completion_tokens_details.reasoning_tokens` is a subset of `completion_tokens`
- Vertex Gemini 2.5+: thinking tokens are included in `candidatesTokenCount`
- Anthropic with extended thinking: thinking tokens are included in `output_tokens`

The subtraction ensures reasoning is billed at `output_cost_per_reasoning_token` (not double-charged at the base output rate as well).

### Tiered pricing (200k threshold)

Some models charge a higher rate once the context exceeds 200 000 tokens. When `input_cost_per_token_above_200k_tokens` is set:

```
below = min(prompt_tokens, 200_000)
above = prompt_tokens - 200_000          # only when prompt_tokens > 200_000

# regular tokens are split proportionally between below/above
regular_above = regular_input × above / prompt_tokens
regular_below = regular_input - regular_above

input_cost = regular_below × input_cost_per_token
           + regular_above × input_cost_per_token_above_200k_tokens
```

The same logic applies to output tokens using `output_cost_per_token_above_200k_tokens`.

Cache prices follow LiteLLM's full-session semantics: when `prompt_tokens > 200_000`, all cache read/write tokens use the matching `*_above_200k_tokens` rate. The 32k/128k/256k/272k full-session cache fields take precedence over the 200k tier whenever configured, with the highest exceeded threshold winning (see "Long-context pricing" below).

### Long-context pricing (32k / 128k / 256k / 272k full-session tiers)

When the prompt exceeds one of these thresholds, the matching `*_above_<N>k_tokens` rate applies to the **full session** rather than only the tokens beyond the threshold — the prompt size selects the tier for regular input, output, cache reads, and cache writes. At exactly the threshold, base rates still apply (the check is strictly "greater than").

272k was added first for models such as GPT-5.6; 32k/128k/256k were added for Alibaba Cloud models (Qwen 3.x, GLM) whose published pricing is a flat rate per input-length bracket (e.g. 0–32k / 32k–128k / 128k–256k / >256k) rather than an incremental rate for the overflow.

A price entry may configure any subset of these four thresholds (e.g. only 32k and 256k, skipping 128k). For each cost component (input, output, cache read, cache write) independently, AIR picks the **highest configured threshold that the prompt exceeds**, in descending order 272k → 256k → 128k → 32k, and skips any threshold whose rate field isn't set. If none of the four apply, cost calculation falls back to the 200k proportional tier described above, then to the base rate.

Example: a model with only `input_cost_per_token_above_128k_tokens` and `input_cost_per_token_above_256k_tokens` set bills a 150k-token prompt at the 128k rate and a 300k-token prompt at the 256k rate; a 100k-token prompt still uses the base rate.

### Specialised token types

| Type                | Formula                                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Audio input         | `audio_input_tokens × input_cost_per_audio_token` (falls back to regular input rate)                                                                |
| Audio output        | `audio_output_tokens × output_cost_per_audio_token` (falls back to regular output rate)                                                             |
| Cached read         | Cached text uses `cache_read_input_token_cost`; cached audio uses `cache_read_input_audio_token_cost` with fallback to the selected cache read rate |
| Cache creation      | 5m and unclassified tokens use `cache_creation_input_token_cost`; 1h tokens use `cache_creation_input_token_cost_above_1hr`; both fall back safely  |
| Reasoning           | `reasoning_tokens × output_cost_per_reasoning_token` (falls back to regular output rate)                                                            |
| Accepted prediction | `accepted_prediction_tokens × output_cost_per_prediction_token` (falls back to regular output rate)                                                 |
| Rejected prediction | `rejected_prediction_tokens × output_cost_per_token` (always at regular output rate)                                                                |
| Images              | `image_count × output_cost_per_image` OR `output_image_tokens × output_cost_per_image_token`                                                        |
| Web Search          | `billable_web_search_count × search_context_cost_per_query[search_context_size]`, with `per_prompt` clamped to one                                  |

## How Prices Are Loaded

Loading is handled by `internal/models/price_loader.go`:

1. The value of `model_prices_link` is inspected to determine the source:
   - Paths starting with `file://` or containing no `://` are read from disk.
   - Paths starting with `http://` or `https://` are fetched via HTTP with a 100 MB limit.
2. The JSON is parsed into a `map[string]*ModelPrice`.
3. Every key is **normalised**: the provider prefix is stripped and the name is lowercased.
   - `"openai/gpt-4-turbo"` → `"gpt-4-turbo"`
   - `"vertex_ai/gemini-2.5-pro"` → `"gemini-2.5-pro"`
   - If two keys normalise to the same string, the last one wins and a warning is logged.
4. The resulting map is stored in a `ModelPriceRegistry` (thread-safe, `sync.RWMutex`).

### DB price merging

When the LiteLLM database is enabled, prices defined in `LiteLLM_ModelTable` are merged on top of the file-based registry via `MergeDB`. Database prices take precedence for any model that appears in both sources. The file-based prices remain intact for all other models.

Cache writes are read from `cache_creation_tokens` or the OpenAI-compatible `cache_write_tokens` alias in both Chat Completions and Responses API usage objects.
Anthropic's `cache_creation_token_details` (`ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`) is preserved in spend-log metadata while the existing aggregate cache-creation token columns remain backward-compatible. Gemini cached-audio counts are taken from `cacheTokensDetails` when the provider supplies a modality breakdown.

### Spend storage contract

AIR keeps LiteLLM's upstream PostgreSQL schema unchanged. `LiteLLM_SpendLogs.spend` and the daily user, team, organization, and end-user tables contain the total cost. Cache and Web Search breakdowns are stored in `LiteLLM_SpendLogs.metadata`.

Kafka and ClickHouse expose the same breakdown as typed fields, including `web_search_requests` and `web_search_cost`. Use that analytics path when structured reconciliation by usage type is required.

### Lookup

When a request completes, the router calls `GetPrice(modelName)` which normalises the name and returns the `*ModelPrice`. If no entry is found, cost calculation is skipped, `spend` is stored as `0`, and the metadata cost breakdown is omitted.
