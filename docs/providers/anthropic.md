# Anthropic

## Configuration

```yaml
credentials:
  - name: "anthropic_main"
    type: "anthropic"
    api_key: "sk-ant-xxxxx"
    base_url: "https://api.anthropic.com"
    rpm: 60
    tpm: 100000
```

## Required Fields

| Field      | Description                                        |
| ---------- | -------------------------------------------------- |
| `api_key`  | Anthropic API key (supports `os.environ/VAR_NAME`) |
| `base_url` | API base URL (`https://api.anthropic.com`)         |
