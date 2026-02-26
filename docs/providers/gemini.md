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

## OpenAI-Compatible API

Gemini AI Studio uses the same converter as [Vertex AI](vertex.md#openai-compatible-api). All parameter mappings, tool types, thinking configuration, content types, structured output, streaming, and token counting work identically.

See the [Vertex AI documentation](vertex.md) for the full reference on:

- [Supported parameters](vertex.md#supported-parameters)
- [Tool calling](vertex.md#tool-calling)
- [Reasoning / Thinking](vertex.md#reasoning--thinking)
- [Content types](vertex.md#content-types) (text, images, audio, video)
- [Audio output](vertex.md#audio-output)
- [Structured output](vertex.md#structured-output)
- [Image generation](vertex.md#image-generation)
- [Streaming](vertex.md#streaming)
