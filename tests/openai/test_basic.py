"""
OpenAI API tests - chat completions and embeddings
Tests OpenAI routing and parameter conversion
"""

import pytest
from test_helpers import (
    TestModels, ResponseValidator, ContentValidator, VectorMath
)


class TestOpenAIChatBasic:
    """Test basic OpenAI chat completions"""

    @pytest.mark.parametrize("temperature", [0.3, 0.7])
    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_chat_with_temperatures(self, openai_client, model, temperature):
        """Test chat with different temperatures"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=temperature,
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)
        ContentValidator.assert_contains_any(
            response.choices[0].message.content,
            ["Paris", "paris"]
        )

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_basic_chat_completion(self, openai_client, model):
        """Test basic chat completion"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_multi_turn_conversation(self, openai_client, model):
        """Test multi-turn conversation"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "My favorite number is 7."},
                {"role": "assistant", "content": "That's a lucky number!"},
                {"role": "user", "content": "What number did I mention?"}
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        ContentValidator.assert_contains_any(
            response.choices[0].message.content,
            ["7", "seven"]
        )

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_advanced_parameters(self, openai_client, model):
        """Test advanced chat parameters"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Tell a short story"}
            ],
            temperature=0.8,
            top_p=0.95,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)
        assert response.usage.completion_tokens > 0

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_stop_sequences(self, openai_client, model):
        """Test stop sequences"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "List colors"}
            ],
            stop=["STOP"],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        assert "STOP" not in response.choices[0].message.content

class TestOpenAIEdgeCases:
    """Test edge cases"""

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_max_tokens_limit(self, openai_client, model):
        """Test max_tokens limit is respected"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Write a very long essay about AI"}
            ],
            max_tokens=50
        )

        ResponseValidator.validate_chat_response(response)
        assert response.usage.completion_tokens <= 80  # Buffer for token counting

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_special_characters(self, openai_client, model):
        """Test special characters handling"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Translate: 你好 мир 🚀"}
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.OPENAI_MODELS)
    def test_code_generation(self, openai_client, model):
        """Test code generation"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Write Python function to add numbers"}
            ],
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)
        content = response.choices[0].message.content.lower()
        assert any(kw in content for kw in ["def", "function", "return"])
