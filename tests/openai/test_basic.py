"""
OpenAI API tests for auto_ai_router
"""

import pytest
import numpy as np


class TestOpenAIBasic:
    """Basic OpenAI functionality tests"""

    @pytest.mark.parametrize("model,temperature,max_tokens", [
        ("gpt-4o-mini", 0.7, 100),
        ("gpt-4o-mini", 0.3, 150),
    ])
    def test_chat_completion(self, openai_client, model, temperature, max_tokens):
        """Test basic chat completion with different parameters"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        assert model in response.model
        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0
        assert len(response.choices) == 1
        assert response.choices[0].message.content

    @pytest.mark.parametrize("model,input_text", [
        ("text-embedding-3-small", "The quick brown fox jumps over the lazy dog"),
        ("text-embedding-3-small", "Machine learning is a subset of artificial intelligence"),
    ])
    def test_single_embedding(self, openai_client, model, input_text):
        """Test single text embedding"""
        response = openai_client.embeddings.create(
            model=model,
            input=input_text
        )

        assert model in response.model
        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0
        assert response.usage.total_tokens > 0

    def test_batch_embeddings(self, openai_client):
        """Test batch text embeddings"""
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language",
            "The weather is nice today",
            "Embeddings convert text to numerical vectors"
        ]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        assert response.model == "text-embedding-3-small"
        assert len(response.data) == len(texts)
        assert response.usage.total_tokens > 0

        # Test similarity calculation
        embeddings = [data.embedding for data in response.data]
        sim1 = self._cosine_similarity(embeddings[0], embeddings[3])  # ML vs Embeddings
        sim2 = self._cosine_similarity(embeddings[0], embeddings[2])  # ML vs Weather
        assert sim1 > sim2  # Related texts should have higher similarity

    def test_streaming_response(self, openai_client):
        """Test streaming chat completion"""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Tell me a very short story about a robot."}
            ],
            temperature=0.8,
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

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class TestOpenAIAdvanced:
    """Advanced OpenAI functionality tests"""

    @pytest.mark.parametrize("params", [
        {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["END", "STOP"]
        },
        {
            "temperature": 0.3,
            "max_tokens": 200,
            "top_p": 0.8,
            "seed": 42,
            "user": "test_user_123"
        }
    ])
    def test_chat_with_advanced_params(self, openai_client, params):
        """Test chat completion with advanced parameters"""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short poem about AI"}
            ],
            **params
        )

        assert response.usage.completion_tokens > 0
        assert response.choices[0].message.content

    def test_conversation_context(self, openai_client):
        """Test multi-turn conversation"""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that remembers context."},
                {"role": "user", "content": "I'm planning a trip to Japan. What are the must-visit places?"},
                {"role": "assistant", "content": "For Japan, I'd recommend Tokyo for modern culture, Kyoto for traditional temples, Osaka for food, and Mount Fuji for natural beauty."},
                {"role": "user", "content": "What's the best time to visit the places you mentioned?"}
            ],
            temperature=0.7,
            max_tokens=250
        )

        assert response.usage.total_tokens > 0
        assert response.choices[0].message.content
