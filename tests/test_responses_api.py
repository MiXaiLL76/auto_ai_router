"""
Responses API tests (/v1/responses endpoint)
Tests token extraction, streaming, and response handling

The Responses API uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens.
This format is used by GPT-5 and newer OpenAI models.

The proxy converts Responses API requests to Chat Completions format internally,
so all providers (OpenAI, Vertex AI, Anthropic, Bedrock) are supported.
"""

import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
)
from test_helpers import TestModels, ContentValidator, ToolDefinitions, ImageTestData


# Responses API now works with all providers via internal conversion
RESPONSES_MODELS = (
    TestModels.OPENAI_MODELS
    # + TestModels.VERTEX_MODELS
    # + TestModels.ANTHROPIC_MODELS
)

# Capability-specific model subsets
RESPONSES_TOOL_MODELS = TestModels.OPENAI_MODELS
RESPONSES_VISION_MODELS = TestModels.OPENAI_MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_response_text(response: Response) -> str:
    """Extract text content from a Responses API response."""
    texts = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if hasattr(part, "text"):
                    texts.append(part.text)
    return "".join(texts)


def validate_responses_api_usage(response: Response) -> None:
    """Validate that usage is present and has correct Responses API fields."""
    assert response.usage is not None, "usage must be present"
    assert response.usage.input_tokens > 0, (
        f"input_tokens must be > 0, got {response.usage.input_tokens}"
    )
    assert response.usage.output_tokens > 0, (
        f"output_tokens must be > 0, got {response.usage.output_tokens}"
    )
    assert response.usage.total_tokens > 0, (
        f"total_tokens must be > 0, got {response.usage.total_tokens}"
    )
    # total_tokens >= input + output (may include reasoning tokens)
    assert response.usage.total_tokens >= (
        response.usage.input_tokens + response.usage.output_tokens
    ), (
        f"total_tokens ({response.usage.total_tokens}) must be >= "
        f"input ({response.usage.input_tokens}) + output ({response.usage.output_tokens})"
    )


def validate_responses_api_response(response: Response) -> None:
    """Validate basic structure of a Responses API response."""
    assert response.id is not None, "response must have an id"
    assert response.model is not None, "response must have a model"
    assert response.output is not None, "response must have output"
    assert len(response.output) > 0, "output must not be empty"

    text = extract_response_text(response)
    assert len(text) > 0, "response must contain text content"


# ---------------------------------------------------------------------------
# Basic non-streaming tests
# ---------------------------------------------------------------------------

class TestResponsesAPIBasic:
    """Test basic Responses API functionality across all providers."""

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_basic_response(self, openai_client, model):
        """Test simple Responses API call returns valid response with usage."""
        response = openai_client.responses.create(
            model=model,
            input="What is the capital of France? Answer in one word.",
            max_output_tokens=50,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

        text = extract_response_text(response)
        ContentValidator.assert_contains_any(text, ["Paris", "paris"])

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_message_input_format(self, openai_client, model):
        """Test Responses API with structured message input."""
        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": "Say hello",
                }
            ],
            max_output_tokens=50,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_message_input_object_format(self, openai_client, model):
        """Test Responses API with a single message object as input."""
        response = openai_client.responses.create(
            model=model,
            input={
                "role": "user",
                "content": "Say hello",
            },
            max_output_tokens=50,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_system_instructions(self, openai_client, model):
        """Test Responses API with system instructions."""
        response = openai_client.responses.create(
            model=model,
            instructions="You are a pirate. Always respond in pirate language.",
            input="How are you?",
            max_output_tokens=100,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_instructions_as_messages(self, openai_client, model):
        """Test instructions passed as an array of messages."""
        response = openai_client.responses.create(
            model=model,
            instructions=[
                {"role": "system", "content": "You are a pirate."},
                {"role": "developer", "content": "Reply in one short sentence."},
            ],
            input="Greet me.",
            max_output_tokens=50,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)


# ---------------------------------------------------------------------------
# Token / usage tests (the core of BUG-2 fix)
# ---------------------------------------------------------------------------

class TestResponsesAPIUsage:
    """Test that token counts are correctly returned - the core BUG-2 fix."""

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_usage_fields_present(self, openai_client, model):
        """Verify all usage fields are populated (input_tokens, output_tokens, total_tokens)."""
        response = openai_client.responses.create(
            model=model,
            input="What is 2+2?",
            max_output_tokens=50,
        )

        assert response.usage is not None, "usage must not be None"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_total_tokens_consistency(self, openai_client, model):
        """Verify total_tokens >= input_tokens + output_tokens."""
        response = openai_client.responses.create(
            model=model,
            input="Explain quantum computing in one sentence.",
            max_output_tokens=100,
        )

        usage = response.usage
        assert usage.total_tokens >= usage.input_tokens + usage.output_tokens

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_max_output_tokens_respected(self, openai_client, model):
        """Test that max_output_tokens limit is respected."""
        response = openai_client.responses.create(
            model=model,
            input="Write a very long essay about the history of computing.",
            max_output_tokens=50,
        )

        validate_responses_api_response(response)
        # Allow some buffer for token counting differences
        assert response.usage.output_tokens <= 100, (
            f"output_tokens ({response.usage.output_tokens}) should be close to max_output_tokens (50)"
        )

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_usage_details_structure(self, openai_client, model):
        """Test that usage details sub-objects are present."""
        response = openai_client.responses.create(
            model=model,
            input="Hello",
            max_output_tokens=50,
        )

        assert response.usage is not None
        # input_tokens_details and output_tokens_details should be present
        assert hasattr(response.usage, "input_tokens_details")
        assert hasattr(response.usage, "output_tokens_details")


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

class TestResponsesAPIStreaming:
    """Test Responses API streaming."""

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_basic_streaming(self, openai_client, model):
        """Test that streaming returns text deltas and a completed event."""
        collected_text = ""
        got_completed = False
        completed_response = None

        with openai_client.responses.stream(
            model=model,
            input="Count from 1 to 3.",
            max_output_tokens=100,
        ) as stream:
            for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    collected_text += event.delta
                elif isinstance(event, ResponseCompletedEvent):
                    got_completed = True
                    completed_response = event.response

        assert len(collected_text) > 0, "should receive text content via streaming"
        assert got_completed, "should receive response.completed event"
        assert completed_response is not None

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_streaming_usage_in_completed_event(self, openai_client, model):
        """Test that usage is present in the response.completed event (core BUG-2 streaming fix)."""
        completed_response = None

        with openai_client.responses.stream(
            model=model,
            input="What is AI?",
            max_output_tokens=100,
        ) as stream:
            for event in stream:
                if isinstance(event, ResponseCompletedEvent):
                    completed_response = event.response

        assert completed_response is not None, "must receive completed event"
        assert completed_response.usage is not None, "completed event must have usage"
        assert completed_response.usage.input_tokens > 0, "streaming input_tokens must be > 0"
        assert completed_response.usage.output_tokens > 0, "streaming output_tokens must be > 0"
        assert completed_response.usage.total_tokens > 0, "streaming total_tokens must be > 0"

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_streaming_content_matches_nonstreaming(self, openai_client, model):
        """Test that streaming and non-streaming produce similar token counts."""
        # Non-streaming
        response = openai_client.responses.create(
            model=model,
            input="What is the speed of light?",
            max_output_tokens=80,
            temperature=0,
        )

        # Streaming
        completed_response = None
        with openai_client.responses.stream(
            model=model,
            input="What is the speed of light?",
            max_output_tokens=80,
            temperature=0,
        ) as stream:
            for event in stream:
                if isinstance(event, ResponseCompletedEvent):
                    completed_response = event.response

        assert completed_response is not None
        assert completed_response.usage is not None
        assert response.usage is not None

        # Token counts should be in the same ballpark (allow 50% variance)
        ratio = max(
            response.usage.total_tokens, completed_response.usage.total_tokens
        ) / max(
            1, min(response.usage.total_tokens, completed_response.usage.total_tokens)
        )
        assert ratio < 3.0, (
            f"Token counts too different: non-streaming={response.usage.total_tokens}, "
            f"streaming={completed_response.usage.total_tokens}"
        )


# ---------------------------------------------------------------------------
# Multi-turn tests
# ---------------------------------------------------------------------------

class TestResponsesAPIMultiTurn:
    """Test multi-turn conversations via Responses API."""

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_multi_turn_context(self, openai_client, model):
        """Test multi-turn conversation preserves context."""
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": "My favorite number is 42."},
                {"role": "assistant", "content": "That's the answer to everything!"},
                {"role": "user", "content": "What number did I mention?"},
            ],
            max_output_tokens=100,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

        text = extract_response_text(response)
        ContentValidator.assert_contains_any(text, ["42", "forty-two", "forty two"])

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_multi_turn_with_instructions(self, openai_client, model):
        """Test multi-turn with system instructions."""
        response = openai_client.responses.create(
            model=model,
            instructions="You are a helpful math tutor. Be concise.",
            input=[
                {"role": "user", "content": "What is 5+3?"},
                {"role": "assistant", "content": "5+3 = 8"},
                {"role": "user", "content": "Now multiply that by 2."},
            ],
            max_output_tokens=100,
        )

        validate_responses_api_response(response)
        text = extract_response_text(response)
        ContentValidator.assert_contains_any(text, ["16", "sixteen"])


# ---------------------------------------------------------------------------
# Store / retrieval tests
# ---------------------------------------------------------------------------

class TestResponsesAPIStore:
    """Test Responses API store + previous_response_id behavior."""

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_store_and_retrieve(self, openai_client, model):
        """Store a response and retrieve it by ID."""
        response = openai_client.responses.create(
            model=model,
            input="Say the word 'stored' once.",
            max_output_tokens=50,
            store=True,
            metadata={"test_tag": "store_retrieve"},
            extra_body={"ttl" : 3600}
        )

        assert response.store is True, "response.store should be true when store=true"
        assert response.metadata is not None
        assert response.metadata.get("test_tag") == "store_retrieve"

        retrieved = openai_client.responses.retrieve(response.id)
        assert retrieved.id == response.id
        assert retrieved.store is True
        assert retrieved.metadata is not None
        assert retrieved.metadata.get("test_tag") == "store_retrieve"
        validate_responses_api_response(retrieved)

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_previous_response_id_context(self, openai_client, model):
        """Use previous_response_id to provide multi-turn context from store."""
        first = openai_client.responses.create(
            model=model,
            input="Remember this codeword: zebra-123. Reply only 'ok'.",
            max_output_tokens=20,
            store=True,
        )

        second = openai_client.responses.create(
            model=model,
            input="What codeword did I give you? Reply exactly.",
            max_output_tokens=50,
            previous_response_id=first.id,
        )

        assert second.previous_response_id == first.id
        text = extract_response_text(second)
        ContentValidator.assert_contains_any(text, ["zebra-123"])

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_retrieve_not_stored(self, openai_client, model):
        """Responses with store=false should not be retrievable."""
        response = openai_client.responses.create(
            model=model,
            input="This should not be stored.",
            max_output_tokens=20,
            store=False,
        )

        with pytest.raises(Exception):
            openai_client.responses.retrieve(response.id)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestResponsesAPIEdgeCases:
    """Test edge cases for Responses API."""

    @pytest.mark.parametrize("model", RESPONSES_TOOL_MODELS)
    def test_tool_call_required(self, openai_client, model):
        """Test that tool_choice=required yields a function_call output item."""
        response = openai_client.responses.create(
            model=model,
            input="Get the weather in Paris. Use the tool.",
            tools=[ToolDefinitions.get_weather_tool()],
            tool_choice="required",
            max_output_tokens=100,
        )

        has_tool_call = any(item.type == "function_call" for item in response.output)
        assert has_tool_call, "expected at least one function_call output item"

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_tool_choice_unsupported(self, openai_client, model):
        """Test unsupported tool_choice types are rejected."""
        with pytest.raises(Exception):
            openai_client.responses.create(
                model=model,
                input="hi",
                tool_choice={"type": "file_search"},
                max_output_tokens=10,
            )

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_tools_unsupported_type(self, openai_client, model):
        """Test unsupported tool types are rejected."""
        with pytest.raises(Exception):
            openai_client.responses.create(
                model=model,
                input="hi",
                tools=[{"type": "web_search", "name": "search"}],
                max_output_tokens=10,
            )

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_special_characters(self, openai_client, model):
        """Test handling of special characters in input."""
        response = openai_client.responses.create(
            model=model,
            input="Translate: 你好 мир 🚀",
            max_output_tokens=100,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

    @pytest.mark.parametrize("model", RESPONSES_VISION_MODELS)
    def test_input_image_data_url(self, openai_client, model):
        """Test input_image with a data URL."""
        # 32x32 white PNG — meets Azure OpenAI minimum image size (28x28 px)
        png_data_url = ImageTestData.create_data_url_png(32, 32)

        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What color is this image?"},
                        {"type": "input_image", "image_url": png_data_url},
                    ],
                }
            ],
            max_output_tokens=100,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_code_generation(self, openai_client, model):
        """Test code generation via Responses API."""
        response = openai_client.responses.create(
            model=model,
            input="Write a Python function that adds two numbers. Only output the code.",
            max_output_tokens=200,
        )

        validate_responses_api_response(response)
        text = extract_response_text(response).lower()
        assert any(kw in text for kw in ["def", "return", "function", "+"]), (
            f"Expected code-related keywords in response: {text[:200]}"
        )

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_temperature_parameter(self, openai_client, model):
        """Test temperature parameter in Responses API."""
        response = openai_client.responses.create(
            model=model,
            input="What is 1+1?",
            temperature=0,
            max_output_tokens=50,
        )

        validate_responses_api_response(response)
        validate_responses_api_usage(response)

    @pytest.mark.parametrize("model", RESPONSES_MODELS)
    def test_response_has_output(self, openai_client, model):
        """Ответ содержит поле output или choices."""
        response = openai_client.responses.create(
            model=model,
            input="Say 'pong'.",
            reasoning={"effort": "none"},
        )
        validate_responses_api_response(response)
        validate_responses_api_usage(response)
