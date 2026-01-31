"""
Anthropic Claude multimodal tests
Tests image handling and vision capabilities
"""

import pytest
from test_helpers import (
    TestModels, ResponseValidator, ImageTestData, StreamingValidator
)


class TestAnthropicImageUrl:
    """Test image URL handling"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_url_basic(self, openai_client, model):
        """Test sending image via URL"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        }
                    ]
                }
            ],
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_url_with_text_before(self, openai_client, model):
        """Test image URL with preceding text"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this famous artwork."},
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        }
                    ]
                }
            ],
            max_tokens=250
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_url_with_text_after(self, openai_client, model):
        """Test image URL with following text"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "What style of art is this?"}
                    ]
                }
            ],
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_url_with_temperature(self, openai_client, model):
        """Test image handling with temperature parameter"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "Describe this image"}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicBase64Image:
    """Test base64 encoded image handling"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_base64_image_basic(self, openai_client, model):
        """Test sending base64 encoded image"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{ImageTestData.get_red_pixel_base64()}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_base64_image_data_url_format(self, openai_client, model):
        """Test base64 image with proper data URL format"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{ImageTestData.get_red_pixel_base64()}"
                            }
                        },
                        {"type": "text", "text": "What do you see?"}
                    ]
                }
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicMultipleImages:
    """Test handling multiple images in a single message"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_multiple_images_same_url(self, openai_client, model):
        """Test sending same image multiple times"""
        image_url = ImageTestData.get_van_gogh_url()

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what you see in these images"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {"type": "text", "text": "Are they identical?"}
                    ]
                }
            ],
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_with_surrounding_text(self, openai_client, model):
        """Test image surrounded by multiple text blocks"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is a famous painting."},
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "Tell me about the artist and style."}
                    ]
                }
            ],
            max_tokens=250
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicImageConversation:
    """Test image handling in multi-turn conversation"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_in_conversation_flow(self, openai_client, model):
        """Test image appearing in conversation context"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "I'll show you an image"},
                {
                    "role": "assistant",
                    "content": "Sure, I'm ready to analyze an image for you."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "What is this?"}
                    ]
                }
            ],
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_multiple_images_in_conversation(self, openai_client, model):
        """Test multiple messages with images"""
        image_url = ImageTestData.get_van_gogh_url()

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": "I see a painting."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this the same or different?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_with_followup_question(self, openai_client, model):
        """Test asking follow-up questions about image"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "What do you see?"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "I see a starry night scene with a village."
                },
                {
                    "role": "user",
                    "content": "What colors dominate the painting?"
                }
            ],
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicImageStreaming:
    """Test streaming with image content"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_streaming_basic(self, openai_client, model):
        """Test streaming with image"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "Describe this painting"}
                    ]
                }
            ],
            max_tokens=200,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_streaming_with_temperature(self, openai_client, model):
        """Test streaming with image and temperature"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "Write creative description"}
                    ]
                }
            ],
            temperature=0.9,
            max_tokens=200,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        StreamingValidator.assert_valid_streaming_response(full_content, chunk_count)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_streaming_concise(self, openai_client, model):
        """Test streaming with image and tight max_tokens"""
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "What is this in 2 sentences?"}
                    ]
                }
            ],
            max_tokens=80,
            stream=True
        )

        full_content, chunk_count = StreamingValidator.collect_streaming_content(stream)
        assert chunk_count > 0
        assert len(full_content) > 0


class TestAnthropicImageEdgeCases:
    """Test edge cases with images"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_only_no_text(self, openai_client, model):
        """Test image without accompanying text"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        }
                    ]
                }
            ],
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_minimal_text_with_image(self, openai_client, model):
        """Test image with minimal text prompt"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "?"}
                    ]
                }
            ],
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_image_with_low_max_tokens(self, openai_client, model):
        """Test image analysis with very limited tokens"""
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": ImageTestData.get_van_gogh_url()}
                        },
                        {"type": "text", "text": "One word description"}
                    ]
                }
            ],
            max_tokens=30
        )

        ResponseValidator.validate_chat_response(response)
