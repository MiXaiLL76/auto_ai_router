# OpenAI

## Configuration

```yaml
credentials:
  - name: "openai_main"
    type: "openai"
    api_key: "sk-proj-xxxxx"
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000
```

## Required Fields

| Field | Description |
|---|---|
| `api_key` | OpenAI API key (supports `os.environ/VAR_NAME`) |
| `base_url` | API base URL (`https://api.openai.com`) |

## Azure OpenAI

For Azure OpenAI, use the same `openai` type with the Azure endpoint:

```yaml
credentials:
  - name: "azure_openai"
    type: "openai"
    api_key: "os.environ/AZURE_OPENAI_KEY"
    base_url: "https://your-resource.openai.azure.com"
    rpm: 100
    tpm: 50000
```
