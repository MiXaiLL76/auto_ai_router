"""
Anthropic Claude Responses API tests
Tests /v1/responses endpoint: basic, reasoning, streaming, multi-turn
"""

import pytest
from test_helpers import TestModels, ContentValidator


ANTHROPIC_RESPONSES_MODELS = TestModels.ANTHROPIC_MODELS


def _collect_responses_output_text(response) -> str:
    """Extract concatenated text from Responses API output items."""
    parts = []
    for item in (response.output or []):
        content_blocks = getattr(item, "content", None) or []
        for block in content_blocks:
            if getattr(block, "type", None) == "output_text":
                parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def _collect_responses_stream_text(stream) -> tuple[str, int]:
    """Collect text and event count from a Responses API event stream.

    Iterates SSE events from client.responses.create(stream=True).
    Text is accumulated from response.output_text.delta events.
    """
    text = ""
    chunk_count = 0
    for event in stream:
        chunk_count += 1
        if getattr(event, "type", None) == "response.output_text.delta":
            text += getattr(event, "delta", "") or ""
    return text, chunk_count


class TestAnthropicResponsesAPIBasic:
    """Basic Responses API functionality."""

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_basic_text_response(self, openai_client, model):
        """Non-streaming Responses API returns non-empty output."""
        response = openai_client.responses.create(
            model=model,
            input="What is the capital of France? Answer in one word.",
            max_output_tokens=50,
        )

        assert response is not None
        assert response.id and response.id.startswith("resp_")
        assert response.status == "completed"
        assert response.output and len(response.output) > 0

        text = _collect_responses_output_text(response)
        assert len(text) > 0, f"Expected non-empty output text, got: {response.output}"
        ContentValidator.assert_contains_any(text.lower(), ["paris"])

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_usage_fields_populated(self, openai_client, model):
        """Responses API returns non-zero usage."""
        response = openai_client.responses.create(
            model=model,
            input="Say 'hi'.",
            max_output_tokens=20,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0, "input_tokens must be > 0"
        assert response.usage.output_tokens > 0, "output_tokens must be > 0"

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_max_output_tokens_respected(self, openai_client, model):
        """max_output_tokens limits response length."""
        response = openai_client.responses.create(
            model=model,
            input="Write a very long essay about ancient Rome.",
            max_output_tokens=20,
        )

        assert response.status == "incomplete"
        assert response.incomplete_details.reason == "max_output_tokens"
        assert 0 < response.usage.output_tokens <= 20

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_system_instructions(self, openai_client, model):
        """instructions field is passed to the model as system prompt."""
        response = openai_client.responses.create(
            model=model,
            input="How are you?",
            instructions="You are a pirate. Always say 'Arrr' at least once.",
            max_output_tokens=100,
        )

        text = _collect_responses_output_text(response)
        assert len(text) > 0
        assert "arr" in text.lower(), f"System instruction (pirate) not followed. Got: {text}"


class TestAnthropicResponsesAPIReasoning:
    """Responses API with extended thinking / reasoning."""

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_reasoning_returns_answer(self, openai_client, model):
        """Responses API with reasoning returns correct factual answer."""
        response = openai_client.responses.create(
            model=model,
            input="Is 17 a prime number? Answer yes or no.",
            reasoning={"effort": "low"},
            max_output_tokens=5000,
        )

        assert response is not None
        assert response.status == "completed"
        assert response.output and len(response.output) > 0

        text = _collect_responses_output_text(response)
        assert len(text) > 0, (
            f"Expected non-empty output. Got output items: {response.output}"
        )
        ContentValidator.assert_contains_any(text.lower(), ["yes"])

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_reasoning_usage_non_zero(self, openai_client, model):
        """Reasoning requests report non-zero usage."""
        response = openai_client.responses.create(
            model=model,
            input="What is 2 + 2?",
            reasoning={"effort": "low"},
            max_output_tokens=5000,
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0, "input_tokens must be > 0 for reasoning request"
        assert response.usage.output_tokens > 0, "output_tokens must be > 0 for reasoning request"

    @pytest.mark.parametrize("effort", ["low", "medium"])
    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_reasoning_effort_levels(self, openai_client, model, effort):
        """Both low and medium reasoning effort produce valid responses."""
        response = openai_client.responses.create(
            model=model,
            input="Is the sky blue? Answer yes or no.",
            reasoning={"effort": effort},
            max_output_tokens=5000,
        )

        text = _collect_responses_output_text(response)
        assert len(text) > 0, f"Empty response for effort={effort}"
        ContentValidator.assert_contains_any(text.lower(), ["yes"])


class TestAnthropicResponsesAPIStreaming:
    """Responses API streaming mode."""

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_streaming_returns_chunks(self, openai_client, model):
        """Streaming Responses API emits multiple events."""
        stream = openai_client.responses.create(
            model=model,
            input="Count from 1 to 5.",
            max_output_tokens=100,
            stream=True,
        )
        text, chunk_count = _collect_responses_stream_text(stream)

        assert chunk_count > 0, "Should receive at least one streaming event"
        assert len(text) > 0, "Should have collected text content"
        ContentValidator.assert_contains_any(text, ["1", "2", "3"])

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_streaming_factual_answer(self, openai_client, model):
        """Streaming Responses API returns correct factual content."""
        stream = openai_client.responses.create(
            model=model,
            input="What is the capital of Germany? One word.",
            max_output_tokens=50,
            stream=True,
        )
        text, chunk_count = _collect_responses_stream_text(stream)

        assert chunk_count > 0
        ContentValidator.assert_contains_any(text.lower(), ["berlin"])

    @pytest.mark.parametrize("model", ANTHROPIC_RESPONSES_MODELS)
    def test_streaming_with_reasoning(self, openai_client, model):
        """Streaming Responses API with reasoning returns correct answer."""
        stream = openai_client.responses.create(
            model=model,
            input="Is 17 a prime number? Answer yes or no.",
            reasoning={"effort": "low"},
            max_output_tokens=5000,
            stream=True,
        )
        text, chunk_count = _collect_responses_stream_text(stream)

        assert chunk_count > 0, "Should receive streaming events"
        assert len(text) > 0, "Should have non-empty text"
        ContentValidator.assert_contains_any(text.lower(), ["yes"])
