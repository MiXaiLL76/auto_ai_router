"""
Multimodal tests (images) for OpenAI -> Anthropic bridge
"""

import base64
import pytest


class TestMultimodal:
    """Test multimodal capabilities (images, etc.)"""

    def test_image_url(self, openai_client):
        """Test sending image via URL"""
        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
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
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )

        assert response.choices[0].message.content is not None
        content = response.choices[0].message.content.lower()
        # Should recognize it's a famous painting
        assert any(word in content for word in ["van gogh", "starry", "night", "painting", "artwork"])

    def test_base64_image(self, openai_client):
        """Test sending image as base64 data URL"""
        # Create a simple 1x1 red pixel PNG as base64
        red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{red_pixel_png}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        assert response.choices[0].message.content is not None
        # Response should mention something about color
        assert len(response.choices[0].message.content) > 0

    def test_multiple_images(self, openai_client):
        """Test sending multiple images in one message"""
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in these images"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": "Are they the same?"
                        }
                    ]
                }
            ],
            max_tokens=200
        )

        assert response.choices[0].message.content is not None
        content = response.choices[0].message.content.lower()
        # Should recognize both images and potentially mention they're the same
        assert any(word in content for word in ["same", "identical", "similar", "starry", "painting"])

    def test_image_with_context(self, openai_client):
        """Test image with surrounding text context"""
        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is a famous artwork."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
                            }
                        },
                        {"type": "text", "text": "Can you identify the artist and describe the style?"}
                    ]
                }
            ],
            max_tokens=250
        )

        assert response.choices[0].message.content is not None
        content = response.choices[0].message.content.lower()
        # Should mention the artist and style
        assert any(word in content for word in ["van gogh", "impressionist", "post-impressionist", "swirl"])

    def test_text_and_image_mixed_conversation(self, openai_client):
        """Test conversation mixing text and images"""
        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {
                    "role": "user",
                    "content": "I'll show you an image"
                },
                {
                    "role": "assistant",
                    "content": "Sure, I'm ready to analyze an image for you."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What art movement does this belong to?"
                        }
                    ]
                }
            ],
            max_tokens=200
        )

        assert response.choices[0].message.content is not None
        content = response.choices[0].message.content.lower()
        # Should mention art movement
        assert any(word in content for word in ["impressionist", "post-impressionist", "movement", "style"])

    def test_image_streaming(self, openai_client):
        """Test streaming with image content"""
        stream = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this painting in 3 sentences"
                        }
                    ]
                }
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

        assert chunk_count > 0, "Should receive chunks"
        assert len(full_content) > 0, "Should have content"
