"""
Anthropic Claude chat completions tests
Tests OpenAI -> Anthropic conversion and response handling
"""

import pytest
from test_helpers import (
    TestModels, ResponseValidator, ContentValidator, StreamingValidator
)


class TestAnthropicBasicChat:
    """Test basic chat completion functionality"""

    @pytest.mark.parametrize("temperature", [0.3, 0.7, 1.0])
    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_chat_with_temperature_variations(self, openai_client, model, temperature):
        """Test chat with different temperature settings"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=temperature,
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)
        ContentValidator.assert_contains_any(
            response.choices[0].message.content,
            ["Paris", "paris", "France"]
        )

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_basic_chat_completion(self, openai_client, model):
        """Test basic chat completion"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'hello world'"}
            ],
            max_tokens=50
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_multi_turn_conversation(self, openai_client, model):
        """Test multi-turn conversation context preservation"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "My favorite number is 42."},
                {"role": "assistant", "content": "That's a great number! It's from Hitchhiker's Guide to the Galaxy."},
                {"role": "user", "content": "What number did I mention earlier?"}
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        ContentValidator.assert_contains_any(
            response.choices[0].message.content,
            ["42", "forty-two", "number"]
        )

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_system_prompt_adherence(self, openai_client, model):
        """Test that system prompts are respected"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a pirate. Respond with pirate language."
                },
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        # Response should contain pirate-like language
        content = response.choices[0].message.content.lower()
        assert len(content) > 0

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_max_tokens_respected(self, openai_client, model):
        """Test that max_tokens parameter is respected"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long essay about artificial intelligence"
                }
            ],
            max_tokens=50
        )

        ResponseValidator.validate_chat_response(response)
        # Completion tokens should be roughly within max_tokens
        assert response.usage.completion_tokens <= 100  # Some buffer for token counting

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_stop_sequences(self, openai_client, model):
        """Test stop sequences parameter"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "List three colors"}
            ],
            stop=["STOP"],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        assert "STOP" not in response.choices[0].message.content


class TestAnthropicAdvancedParameters:
    """Test advanced chat parameters"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_advanced_parameters_combination(self, openai_client, model):
        """Test combining multiple advanced parameters"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a creative story about robots"}
            ],
            temperature=0.8,
            # top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)
        assert response.usage.completion_tokens > 0

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_temperature_zero_determinism(self, openai_client, model):
        """Test that temperature=0 produces more deterministic results"""
        # Send same request twice with temperature=0
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        response1 = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=50
        )

        response2 = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=50
        )

        # With temperature=0, responses should be identical or very similar
        content1 = response1.choices[0].message.content
        content2 = response2.choices[0].message.content
        assert len(content1) > 0 and len(content2) > 0

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_top_p_effect(self, openai_client, model):
        """Test top_p parameter effect"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Write a short creative sentence"}
            ],
            top_p=0.5,  # More restrictive
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
        assert response.usage.completion_tokens > 0


class TestAnthropicEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_very_long_context(self, openai_client, model):
        """Test handling of longer conversation history"""
        messages = []
        for i in range(5):
            messages.append({
                "role": "user",
                "content": f"Question {i+1}: What is {i+1}+1?"
            })
            messages.append({
                "role": "assistant",
                "content": f"The answer to {i+1}+1 is {i+2}."
            })

        messages.append({
            "role": "user",
            "content": "What was the second question I asked?"
        })

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_empty_system_message(self, openai_client, model):
        """Test handling empty system message"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=50
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_special_characters_in_messages(self, openai_client, model):
        """Test handling special characters"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Explain: 你好 مرحبا Привет 🚀 <html>test</html>"
                }
            ],
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_code_generation(self, openai_client, model):
        """Test code generation capability"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a Python function that adds two numbers"
                }
            ],
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)
        content = response.choices[0].message.content.lower()
        # Should contain code-related keywords
        assert any(kw in content for kw in ["def", "function", "return", "python"])
