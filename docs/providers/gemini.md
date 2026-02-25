# Gemini AI Studio

## Configuration

```yaml
credentials:
  - name: "gemini_studio"
    type: "gemini"
    api_key: "os.environ/GEMINI_API_KEY"
    base_url: "https://generativelanguage.googleapis.com"
    rpm: 60
    tpm: -1
```

## Required Fields

| Field      | Description                                                |
| ---------- | ---------------------------------------------------------- |
| `api_key`  | Google AI Studio API key (supports `os.environ/VAR_NAME`)  |
| `base_url` | API base URL (`https://generativelanguage.googleapis.com`) |

## API Key Setup

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create an API key
3. Set it in your environment or config:

```bash
export GEMINI_API_KEY="AIza..."
```
