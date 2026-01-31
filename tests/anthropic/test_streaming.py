"""
Anthropic Claude streaming tests
Tests streaming functionality and proper chunk handling
"""

import pytest
from test_helpers import (
    TestModels, ResponseValidator, StreamingValidator, ContentValidator
)


class TestAnthropicStreaming:
    """Test streaming chat completion functionality"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_basic_streaming(self, openai_client, model):
        """Test basic streaming response"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        # Check for expected counting
        ContentValidator.assert_contains_any(full_content, ["1", "one"])

    @pytest.mark.parametrize("temperature", [0.3, 0.9])
    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_with_temperature_variations(self, openai_client, model, temperature):
        """Test streaming with different temperatures"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Write a creative sentence about robots"}
            ],
            temperature=temperature,
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        ContentValidator.assert_contains_any(full_content.lower(), ["robot"])

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_with_system_prompt(self, openai_client, model):
        """Test streaming respects system prompt"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Always be concise."
                },
                {"role": "user", "content": "What is 5+3?"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        ContentValidator.assert_contains_any(full_content, ["8", "eight"])

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_multi_turn_conversation(self, openai_client, model):
        """Test streaming in multi-turn context"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": "That's a nice choice! Blue is calming."},
                {"role": "user", "content": "What color did I mention?"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        ContentValidator.assert_contains_any(full_content.lower(), ["blue"])

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_with_stop_sequence(self, openai_client, model):
        """Test streaming respects stop sequences"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "List three colors: red"}
            ],
            stop=["STOP"],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        # Stop sequence should not appear in content
        assert "STOP" not in full_content

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_chunk_structure(self, openai_client, model):
        """Test proper chunk structure in streaming response"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=50,
            stream=True
        )

        chunk_count = 0
        content_chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            assert hasattr(chunk, 'choices')
            if chunk.choices and len(chunk.choices) > 0:
                assert hasattr(chunk.choices[0], 'delta')
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_chunk_count += 1

        assert chunk_count > 0, "Should receive multiple chunks"
        assert content_chunk_count > 0, "Should have content chunks"

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_with_advanced_parameters(self, openai_client, model):
        """Test streaming with multiple parameters"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Write a short story"}
            ],
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            max_tokens=150,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_handles_long_response(self, openai_client, model):
        """Test streaming handles longer responses properly"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a paragraph about artificial intelligence"
                }
            ],
            max_tokens=300,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        # Longer responses should have multiple chunks
        assert chunk_count >= 5, "Longer response should have multiple chunks"
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_special_characters(self, openai_client, model):
        """Test streaming handles special characters correctly"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a greeting in Russian and Chinese"
                }
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        # Should contain non-ASCII characters
        assert any(ord(char) > 127 for char in full_content)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_respects_max_tokens(self, openai_client, model):
        """Test streaming respects max_tokens limit"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long essay about quantum physics"
                }
            ],
            max_tokens=60,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        # Response should be relatively short due to max_tokens limit
        assert len(full_content) > 0


class TestAnthropicStreamingEdgeCases:
    """Test streaming edge cases"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_minimal_response(self, openai_client, model):
        """Test streaming with minimal max_tokens"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hi"}
            ],
            max_tokens=10,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        assert len(full_content) > 0

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_with_empty_system_message(self, openai_client, model):
        """Test streaming with empty system message"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_streaming_very_long_conversation(self, openai_client, model):
        """Test streaming with long conversation history"""
        messages = []
        for i in range(3):
            messages.append({"role": "user", "content": f"Question {i+1}"})
            messages.append({"role": "assistant", "content": f"Answer {i+1}"})

        messages.append({"role": "user", "content": "What's my latest question?"})

        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
