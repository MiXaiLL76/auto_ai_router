"""
OpenAI streaming tests for auto_ai_router
"""

import pytest


class TestOpenAIStreaming:
    """OpenAI streaming functionality tests"""

    @pytest.mark.parametrize("model", [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo"
    ])
    def test_basic_streaming(self, openai_client, model):
        """Test basic streaming with different models"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            max_tokens=50,
            stream=True
        )

        full_content = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert chunk_count > 0
        assert len(full_content) > 0
        assert any(str(i) in full_content for i in range(1, 6))

    @pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0])
    def test_streaming_with_temperature(self, openai_client, temperature):
        """Test streaming with different temperature values"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Write a short creative sentence about robots"}
            ],
            temperature=temperature,
            max_tokens=30,
            stream=True
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
            max_tokens=50,
            stream=True
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
            max_tokens=30,
            stream=True
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
                {"role": "user", "content": "List three colors: red, green, STOP"}
            ],
            max_tokens=100,
            stop=["STOP"],
            stream=True
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
            max_tokens=20,
            stream=True
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Check first chunk structure
        first_chunk = chunks[0]
        assert hasattr(first_chunk, 'choices')
        assert len(first_chunk.choices) > 0
        assert hasattr(first_chunk.choices[0], 'delta')

        # Check that we get content in some chunks
        content_chunks = [
            chunk for chunk in chunks 
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

    def test_streaming_error_handling(self, openai_client):
        """Test streaming with potential error conditions"""
        # Test with very small max_tokens
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Write a long essay about artificial intelligence"}
            ],
            max_tokens=5,  # Very small limit
            stream=True
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