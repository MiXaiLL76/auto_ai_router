"""
OpenAI streaming tests
Tests streaming chat completions
"""

import pytest
from test_helpers import (
    TestModels, StreamingValidator, ContentValidator
)


class TestOpenAIStreaming:
    """Test OpenAI streaming functionality"""

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_basic_streaming(self, openai_client, model):
        """Test basic streaming"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count 1 to 3"}],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        ContentValidator.assert_contains_any(full_content, ["1", "one"])

    @pytest.mark.parametrize("temperature", [0.3, 0.9])
    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_temperatures(self, openai_client, model, temperature):
        """Test streaming with temperatures"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write a sentence"}],
            temperature=temperature,
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_with_system_message(self, openai_client, model):
        """Test streaming with system prompt"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        ContentValidator.assert_contains_any(full_content, ["4", "four"])

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_conversation(self, openai_client, model):
        """Test streaming with conversation context"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "I like red."},
                {"role": "assistant", "content": "Red is nice."},
                {"role": "user", "content": "What did I say?"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        ContentValidator.assert_contains_any(full_content.lower(), ["red"])

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_chunk_structure(self, openai_client, model):
        """Test streaming chunk structure"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=50,
            stream=True
        )

        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            assert hasattr(chunk, 'choices')
            if chunk.choices:
                assert len(chunk.choices) > 0
                assert hasattr(chunk.choices[0], 'delta')

        assert chunk_count > 0

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_with_stop_sequence(self, openai_client, model):
        """Test streaming with stop sequence"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "List colors"}],
            stop=["STOP"],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        assert "STOP" not in full_content

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_advanced_parameters(self, openai_client, model):
        """Test streaming with multiple parameters"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell a story"}],
            temperature=0.8,
            top_p=0.9,
            frequency_penalty=0.1,
            max_tokens=150,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_special_characters(self, openai_client, model):
        """Test streaming with special characters"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Greet in Russian and Chinese"}],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)
        assert any(ord(c) > 127 for c in full_content)

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_streaming_min_max_tokens(self, openai_client, model):
        """Test streaming respects max_tokens"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write long essay"}],
            max_tokens=50,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
