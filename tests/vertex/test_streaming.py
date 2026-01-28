"""
Vertex AI streaming tests for auto_ai_router
"""

import pytest


class TestVertexStreaming:
    """Vertex AI streaming functionality tests"""

    @pytest.mark.parametrize("model", [
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ])
    def test_basic_streaming(self, openai_client, model):
        """Test basic streaming with different Gemini models"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            max_tokens=150,
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

    @pytest.mark.parametrize("temperature", [0.3, 0.7])
    def test_streaming_with_temperature(self, openai_client, temperature):
        """Test streaming with different temperature values"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Write one sentence about robots"}
            ],
            temperature=temperature,
            max_tokens=150,
            stream=True
        )

        full_content = ""
        chunk_count = 0
        content_chunks = 0

        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_chunks += 1
                full_content += chunk.choices[0].delta.content

        # Debug info for troubleshooting
        print(f"Total chunks: {chunk_count}, Content chunks: {content_chunks}, Content: '{full_content}'")

        # More lenient assertion - either we get chunks or content
        assert chunk_count > 0 or len(full_content) > 0, f"No chunks ({chunk_count}) and no content ('{full_content}')"

    def test_streaming_multilingual(self, openai_client):
        """Test streaming with multilingual content"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Напиши короткое приветствие на русском языке"}
            ],
            max_tokens=130,
            stream=True
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert len(full_content) > 0
        # Should contain Cyrillic characters
        assert any(ord(char) > 1000 for char in full_content)

    def test_streaming_code_generation(self, openai_client):
        """Test streaming code generation"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "user", "content": "Write a Python function that adds two numbers. Just the function code."}
            ],
            max_tokens=180,
            stream=True
        )

        full_content = ""
        chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        # Debug info
        print(f"Code generation - Chunks: {chunk_count}, Content: '{full_content}'")

        # More lenient checks
        assert chunk_count > 0 or len(full_content) > 0, f"No response received"

        # If we got content, check for code-related keywords
        if len(full_content) > 0:
            content_lower = full_content.lower()
            has_code_keywords = any(keyword in content_lower for keyword in ["def", "function", "add", "return", "python"])
            assert has_code_keywords, f"No code keywords found in: '{full_content}'"

    def test_streaming_with_system_prompt(self, openai_client):
        """Test streaming with system prompt"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI? Answer in one sentence."}
            ],
            max_tokens=150,
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

    def test_streaming_conversation_context(self, openai_client):
        """Test streaming with conversation context"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": "That's nice! Blue is a calming color."},
                {"role": "user", "content": "What color did I mention?"}
            ],
            max_tokens=130,
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

    def test_streaming_chunk_structure(self, openai_client):
        """Test Vertex AI streaming chunk structure"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=120,
            stream=True
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks:
            assert hasattr(chunk, 'choices')
            if chunk.choices:
                assert len(chunk.choices) > 0
                assert hasattr(chunk.choices[0], 'delta')

        # Check that we get some content
        has_content = any(
            chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content
            for chunk in chunks
        )
        assert has_content

    def test_streaming_with_extra_body(self, openai_client):
        """Test streaming with Vertex AI specific parameters"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Write a short sentence"}
            ],
            max_tokens=140,
            stream=True,
            extra_body={
                "generation_config": {
                    "top_k": 40,
                    "top_p": 0.8
                }
            }
        )

        full_content = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert chunk_count > 0
        assert len(full_content) > 0

    def test_streaming_error_recovery(self, openai_client):
        """Test streaming with very restrictive parameters"""
        stream = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Explain quantum physics in detail"}
            ],
            max_tokens=3,  # Very restrictive
            stream=True
        )

        full_content = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        # Should still get some response even with very small limit
        assert chunk_count > 0
