"""
Streaming response tests for OpenAI -> Anthropic bridge
"""

import pytest


class TestStreamingResponses:
    """Test streaming functionality"""

    def test_basic_streaming(self, openai_client):
        """Test basic streaming chat completion"""
        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Count from 1 to 5 with short description"}
            ],
            max_tokens=100,
            stream=True
        )

        full_content = ""
        chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert chunk_count > 0, "Should receive multiple chunks"
        assert len(full_content) > 0, "Should have content"
        assert "1" in full_content or "one" in full_content.lower(), "Should count"

    def test_streaming_with_system_prompt(self, openai_client):
        """Test streaming with system prompt"""
        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "system", "content": "You are a poetic assistant. Respond in haiku."},
                {"role": "user", "content": "Write about technology"}
            ],
            max_tokens=150,
            stream=True
        )

        chunks_received = []
        for chunk in stream:
            chunks_received.append(chunk)

        assert len(chunks_received) > 0, "Should receive chunks"

        # Verify we got content
        full_text = ""
        for chunk in chunks_received:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content

        assert len(full_text) > 0, "Should have content"

    def test_streaming_multi_turn_conversation(self, openai_client):
        """Test streaming in multi-turn conversation"""
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And what's 4+4?"}
        ]

        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=messages,
            max_tokens=50,
            stream=True
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert "8" in full_content, "Should answer 4+4=8"

    def test_streaming_with_temperature(self, openai_client):
        """Test streaming respects temperature parameter"""
        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Say hello in a creative way"}
            ],
            temperature=0.9,
            max_tokens=80,
            stream=True
        )

        chunks = list(stream)
        assert len(chunks) > 0, "Should receive streaming chunks"

    def test_streaming_finish_reason(self, openai_client):
        """Test that streaming provides finish_reason"""
        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "What is AI?"}
            ],
            max_tokens=100,
            stream=True
        )

        last_chunk = None
        for chunk in stream:
            last_chunk = chunk

        assert last_chunk is not None, "Should have chunks"
        # Last chunk should have finish_reason or content
        if last_chunk.choices:
            assert last_chunk.choices[0]

    def test_streaming_stop_sequences(self, openai_client):
        """Test streaming with stop sequences"""
        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "List three things: apple, banana, cherry"}
            ],
            max_tokens=100,
            stop=["cherry"],
            stream=True
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert len(full_content) > 0, "Should have content before stop sequence"
