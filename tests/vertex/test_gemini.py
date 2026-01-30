"""
Vertex AI Gemini tests for auto_ai_router
"""

import pytest
import requests
import json


class TestGeminiText:
    """Gemini text generation tests"""

    @pytest.mark.parametrize("model,temperature,max_tokens", [
        ("gemini-2.5-pro", 0.8, 250),
        ("gemini-2.5-flash", 0.7, 200),
        ("gemini-2.5-pro", 0.3, 300),
    ])
    def test_text_generation(self, openai_client, model, temperature, max_tokens):
        """Test basic text generation with Gemini"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative writing assistant."},
                {"role": "user", "content": "Write a short poem about artificial intelligence in 4 lines."}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        assert model in response.model
        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0
        assert response.choices[0].message.content
class TestGeminiAdvanced:
    """Advanced Gemini functionality tests"""

    @pytest.mark.parametrize("params", [
        {
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "seed": 42,
            "user": "test_user_123",
            "stop": ["END", "STOP"],
            "extra_body": {
                "generation_config": {
                    "top_k": 40,
                    "temperature": 0.8,
                    "seed": 123
                }
            }
        }
    ])
    def test_all_parameters(self, master_key, base_url, params):
        """Test comprehensive OpenAI parameters with Gemini"""
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short poem about AI"}
            ],
            **params
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=200
        )

        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert result["choices"][0]["message"]["content"]

    def test_conversation_context(self, openai_client):
        """Test multi-turn conversation with Gemini"""
        response = openai_client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that remembers context."},
                {"role": "user", "content": "I'm planning a trip to Japan. What are the must-visit places?"},
                {"role": "assistant", "content": "For Japan, I'd recommend Tokyo for modern culture, Kyoto for traditional temples, Osaka for food, and Mount Fuji for natural beauty."},
                {"role": "user", "content": "What's the best time to visit the places you mentioned?"}
            ],
            temperature=0.7,
            max_tokens=250
        )

        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0
        assert response.choices[0].message.content
