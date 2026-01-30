"""
Vertex AI Gemini streaming tests
Tests streaming with Gemini models
"""

import pytest
from test_helpers import TestModels, StreamingValidator


class TestVertexStreaming:
    """Test Vertex Gemini streaming"""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming_basic(self, openai_client, model):
        """Test basic streaming"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count 1 to 3"}],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        assert len(full_content) > 0

    @pytest.mark.parametrize("temperature", [0.3, 0.9])
    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming_with_temperature(self, openai_client, model, temperature):
        """Test streaming with temperatures"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write a sentence"}],
            temperature=temperature,
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming_chunk_structure(self, openai_client, model):
        """Test chunk structure"""
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

        assert chunk_count > 0

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming_with_system(self, openai_client, model):
        """Test streaming with system message"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is AI?"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming_multilingual(self, openai_client, model):
        """Test streaming with multilingual content"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Greet in Russian and Arabic"}
            ],
            max_tokens=150,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        assert len(full_content) > 0
        # May contain thinking before multilingual response

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming_long_response(self, openai_client, model):
        """Test streaming with longer response"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write about AI"}],
            max_tokens=300,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        assert len(full_content) > 0
