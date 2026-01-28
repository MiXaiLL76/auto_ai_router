# Auto AI Router Tests

Comprehensive pytest test suite for auto_ai_router functionality.

## Setup

1. Install test dependencies:

```bash
pip install -r tests/requirements.txt
```

2. Set environment variables:

```bash
export ROUTER_MASTER_KEY="your-master-key"
export ROUTER_BASE_URL="http://localhost:8080/v1"
```

Or create a `.env` file in the project root:

```
ROUTER_MASTER_KEY=your-master-key
ROUTER_BASE_URL=http://localhost:8080/v1
```

## Running Tests

### All tests

```bash
pytest tests/
```

### Specific test categories

```bash
# OpenAI functionality tests
pytest tests/openai/

# Vertex AI functionality tests
pytest tests/vertex/

# Integration tests
pytest tests/test_integration.py

# Only fast tests (exclude slow image generation)
pytest tests/ -m "not slow"
```

### Specific test files

```bash
# Basic OpenAI tests
pytest tests/openai/test_basic.py

# Gemini text tests
pytest tests/vertex/test_gemini.py

# Image generation tests
pytest tests/vertex/test_imagen.py
pytest tests/vertex/test_gemini_image.py
```

### With verbose output

```bash
pytest tests/ -v
```

## Test Structure

- `tests/openai/` - OpenAI API compatibility tests
  - `test_basic.py` - Chat completions, embeddings, streaming
- `tests/vertex/` - Vertex AI specific tests
  - `test_gemini.py` - Gemini text generation
  - `test_imagen.py` - Imagen image generation
  - `test_gemini_image.py` - Gemini image generation/editing
- `tests/test_integration.py` - Router integration tests
- `tests/conftest.py` - Shared fixtures and configuration

## Test Categories

Tests are marked with categories:

- `@pytest.mark.slow` - Long-running tests (image generation)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.openai` - OpenAI functionality
- `@pytest.mark.vertex` - Vertex AI functionality
- `@pytest.mark.image` - Image generation tests

## Environment Variables

Required:

- `ROUTER_MASTER_KEY` - Master key for authentication
- `ROUTER_BASE_URL` - Router base URL (default: http://localhost:8080/v1)

## Notes

- Image generation tests may take longer to complete
- Tests create temporary files that are automatically cleaned up
- Some tests require specific models to be available in your router configuration
