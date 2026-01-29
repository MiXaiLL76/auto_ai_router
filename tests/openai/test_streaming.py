"""
OpenAI streaming tests for auto_ai_router
"""

import pytest


class TestOpenAIStreaming:
    """OpenAI streaming functionality tests"""

    @pytest.mark.parametrize("model", [
        "gpt-4o-mini",
    ])
    def test_basic_streaming(self, openai_client, model):
        """Test basic streaming with different models"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            max_tokens=150,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        chunk_count = 0
        usage_found = False

        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
            # Check for usage in final chunk
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_found = True
                assert chunk.usage.total_tokens > 0

        assert chunk_count > 0
        assert len(full_content) > 0
        assert any(str(i) in full_content for i in range(1, 6))
        assert usage_found, "Usage information not found in streaming response"

    @pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0])
    def test_streaming_with_temperature(self, openai_client, temperature):
        """Test streaming with different temperature values"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Write a short creative sentence about robots"}
            ],
            temperature=temperature,
            max_tokens=130,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert len(full_content) > 0
        assert "robot" in full_content.lower()

    def test_streaming_with_system_message(self, openai_client):
        """Test streaming with system message"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Hope this helps!'"},
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=150,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert len(full_content) > 0
        assert "4" in full_content
        assert "hope this helps" in full_content.lower()

    def test_streaming_conversation(self, openai_client):
        """Test streaming with conversation context"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": "That's nice! Blue is a calming color."},
                {"role": "user", "content": "What did I just tell you about my favorite color?"}
            ],
            max_tokens=130,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert len(full_content) > 0
        assert "blue" in full_content.lower()

    def test_streaming_with_stop_sequences(self, openai_client):
        """Test streaming with stop sequences"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "List three colors: red, green"}
            ],
            max_tokens=150,
            stop=["STOP"],
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert len(full_content) > 0
        # Should not contain STOP since it's a stop sequence
        assert "STOP" not in full_content

    def test_streaming_chunk_structure(self, openai_client):
        """Test streaming chunk structure"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=120,
            stream=True,
            stream_options={"include_usage": True}
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Check that we get content in some chunks
        content_chunks = [
            chunk for chunk in chunks
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

        # Check structure of content chunks
        for chunk in content_chunks:
            assert hasattr(chunk, 'choices')
            assert len(chunk.choices) > 0
            assert hasattr(chunk.choices[0], 'delta')

    def test_streaming_error_handling(self, openai_client):
        """Test streaming with potential error conditions"""
        # Test with very small max_tokens
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Write a long essay about artificial intelligence"}
            ],
            max_tokens=150,  # Very small limit
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        # Should still get some content even with small limit
        assert chunk_count > 0
        assert len(full_content) > 0
