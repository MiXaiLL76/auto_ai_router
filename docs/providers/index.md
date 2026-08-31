# Providers

Auto AI Router supports multiple LLM providers. Each provider type has its own authentication method and required configuration fields.

## Provider Comparison

| Provider                      | Type        | Required Fields                                                    | Auth Method              |
| ----------------------------- | ----------- | ------------------------------------------------------------------ | ------------------------ |
| [OpenAI](openai.md)           | `openai`    | `api_key`, `base_url`                                              | API Key                  |
| [Anthropic](anthropic.md)     | `anthropic` | `api_key`, `base_url`                                              | API Key                  |
| [Comet API](cometapi.md)      | `cometapi`  | `api_key`, `base_url`                                              | API Key                  |
| [ProMan](proman.md)           | `proman`    | `api_key`, `base_url`                                              | API Key                  |
| [AWS Bedrock](bedrock.md)     | `bedrock`   | `api_key`, `base_url`                                              | Bearer Token             |
| [Vertex AI](vertex.md)        | `vertex-ai` | `project_id`, `location`, `credentials_file` or `credentials_json` | OAuth2 / Service Account |
| [Gemini AI Studio](gemini.md) | `gemini`    | `api_key`, `base_url`                                              | API Key                  |
| [Proxy](proxy.md)             | `proxy`     | `base_url`                                                         | Optional API Key         |
| [Auto AI Router](air.md)      | `air`       | `base_url`                                                         | Optional API Key         |

## Common Fields

All credential types share these fields:

| Field         | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| `name`        | string | Unique identifier for this credential            |
| `rpm`         | int    | Requests per minute limit (-1 = unlimited)       |
| `tpm`         | int    | Tokens per minute limit (-1 = unlimited)         |
| `auth_type`   | string | Optional auth override (`bearer` or `x-api-key`) |
| `priority`    | int    | Selection-order group; `999` = last-resort (`is_fallback` is a deprecated alias) |
